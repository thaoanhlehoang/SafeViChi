"""
Bộ chuẩn hóa chống né lọc — Giai đoạn 1 (rule-based + từ điển ViLexNorm).

Thiết kế đối phó 11 loại perturbation từ bộ sinh biến thể:

  Perturbation               │ Bước xử lý               │ Chiến lược
  ───────────────────────────┼───────────────────────────┼──────────────────────
  surface_obfuscation        │ B2 lookalike + B3 sep     │ rule deterministic
  boundary_variation         │ B3 separator removal      │ rule (chỉ xóa .-_)
  expressive_lengthening     │ B4 collapse elongation    │ regex
  diacritic_variation        │ B5 dictionary             │ tra cứu (an toàn)
  abbreviation_clipping      │ B5 dictionary             │ tra cứu
  intentional_spelling       │ B5 dictionary             │ tra cứu
  phonetic_spelling          │ B5 dictionary             │ tra cứu
  dialectal_writing          │ B5 dictionary             │ tra cứu
  slang_lexical              │ B5 dictionary             │ tra cứu (chỉ 1:1)
  context_dependent          │ ❌ BỎ QUA                 │ quá rủi ro
  typographical_noise        │ ❌ BỎ QUA                 │ quá rủi ro

Nguyên tắc vàng: THÀ KHÔNG SỬA, CHỨ ĐỪNG SỬA SAI.
  - Lookalike: chỉ chuyển số→chữ khi số NẰM TRONG từ có chữ cái
  - Dictionary: chỉ thay thế khi từ gốc CHÍNH XÁC match (case-insensitive)
  - Elongation: chỉ collapse khi >= 3 ký tự liên tiếp giống nhau
  - KHÔNG xử lý: context_dependent, typographical_noise (dễ sửa sai)
  - KHÔNG xử lý: boundary_variation dạng 1→nhiều từ (cần neural)

Pipeline 5 bước:
  B1. Chuẩn hóa Unicode NFC
  B2. Ánh xạ lookalike (số/ký hiệu → chữ, chỉ trong context chữ cái)
  B3. Xóa separator chèn thêm (chỉ .-_ giữa ký tự chữ, KHÔNG xóa space)
  B4. Gộp ký tự lặp (>= 3 lần → 1 lần)
  B5. Tra từ điển teencode (JSON trích từ ViLexNorm)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional


# ============================================================
# Load từ điển
# ============================================================

_DICT_DIR = Path(__file__).parent
_TEENCODE_DICT_PATH = _DICT_DIR / "teencode_dict.json"


def _load_teencode_dict() -> dict[str, str]:
    """Load từ điển teencode từ file JSON."""
    if not _TEENCODE_DICT_PATH.exists():
        return {}
    with open(_TEENCODE_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TEENCODE_MAP: dict[str, str] = _load_teencode_dict()


# ============================================================
# Bước 2: Lookalike — AN TOÀN
# Chỉ chuyển số→chữ khi token PHẦN LỚN là chữ cái (alpha > digits).
# VD: "m4y" (2 alpha, 1 digit) → "may" ✓
#     "400k" (1 alpha, 3 digit) → giữ nguyên ✓
# ============================================================

LOOKALIKE_MAP = {
    "4": "a", "3": "e", "1": "i", "0": "o",
    "5": "s", "9": "g", "8": "b", "@": "a",
}

_TOKEN_RE = re.compile(r"\S+")


def _map_lookalike_token(token: str) -> str:
    """
    Chuyển lookalike trong 1 token.
    Hai guard:
    1. Token bắt đầu bằng số → đây là số đo (40km, 40km/h, 1TB) → giữ nguyên.
    2. Majority rule: chỉ chuyển khi num_alpha > num_convertible.
    """
    # Guard 1: số đo (40km, 40km/h, 1TB...) — token bắt đầu bằng digit
    if token and token[0].isdigit():
        return token

    num_alpha = sum(1 for c in token if c.isalpha())
    num_convertible = sum(1 for c in token if c in LOOKALIKE_MAP)

    # Guard 2: không có gì chuyển, hoặc token chủ yếu là số (hoặc hòa) → bỏ qua
    if num_convertible == 0 or num_alpha <= num_convertible:
        return token

    return "".join(LOOKALIKE_MAP.get(ch, ch) for ch in token)


def map_lookalike_chars(text: str) -> str:
    """
    Bước 2: Ánh xạ lookalike theo từng token.
    An toàn: giữ nguyên khi token chủ yếu là số (VD: "400k", "300k").
    """
    def replace_token(match):
        return _map_lookalike_token(match.group())
    return _TOKEN_RE.sub(replace_token, text)


# ============================================================
# Bước 3: Separator removal — BẢO THỦ
# Chỉ xóa dấu . - _ nằm giữa 2 KÝ TỰ CHỮ CÁI.
# KHÔNG xóa khoảng trắng (tránh merge 2 từ).
# VD: "n.g.u" → "ngu" ✓ , "3.14" → giữ nguyên ✓
# ============================================================

# Chỉ xóa khi các ký tự [.\-_·~] bị kẹp giữa 2 chữ cái tiếng Việt.
# VD: "n.g.u" -> "ngu", "c_h_ử_i" -> "chửi", "a~b" -> "ab"
SEPARATOR_PATTERN = re.compile(r"(?<=[a-zA-ZÀ-ỹ])[.\-_·~](?=[a-zA-ZÀ-ỹ])")


def remove_inserted_separators(text: str) -> str:
    """Bước 3: Xóa separator chèn giữa chữ cái."""
    return SEPARATOR_PATTERN.sub("", text)


# ============================================================
# Bước 4: Collapse elongation — BẢO THỦ
# Gộp khi >= 3 ký tự GIỐNG NHAU liên tiếp: "nàoooo" → "nào"
# Xử lý thêm: ký tự có dấu + base letter lặp: "quáaaaa" → "quá"
#   (vì 'á' ≠ 'a' nên regex đơn thuần không bắt được)
# Giữ nguyên 2 ký tự giống nhau (VD: "oo" trong "xoong" là hợp lệ).
# ============================================================

# Chỉ gộp KÝ TỰ CHỮ CÁI lặp >= 3 lần. Không gộp số (100000) hay dấu câu (...).
ELONGATION_PATTERN = re.compile(r"([a-zA-ZÀ-ỹ])\1{2,}")


def _strip_base(char: str) -> str:
    """Bỏ dấu thanh/phụ để lấy ký tự gốc. VD: á → a, ồ → o."""
    import unicodedata
    s = char.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def collapse_elongation(text: str) -> str:
    """
    Bước 4: Gộp ký tự lặp.
    - Bước 4a: gộp >= 3 ký tự giống hệt → 1 ("ooooo" → "o")
    - Bước 4b: bỏ base letter thừa sau ký tự có dấu ("quáa" → "quá")
      Chỉ bỏ khi ký tự thừa là base letter của ký tự trước (VD: a là base của á)
    """
    # 4a: collapse ký tự giống hệt
    text = ELONGATION_PATTERN.sub(r"\1", text)

    # 4b: strip trailing base letter sau ký tự có dấu
    # VD: "quáa" → chữ cuối "a" là base của "á" (chữ áp cuối) → bỏ "a"
    if len(text) >= 2:
        chars = list(text)
        result = [chars[0]]
        for i in range(1, len(chars)):
            prev = chars[i - 1]
            curr = chars[i]
            # Nếu curr là base letter của prev (và prev có dấu) → bỏ curr
            if (prev != curr
                    and curr.isalpha()
                    and prev.isalpha()
                    and _strip_base(prev) == _strip_base(curr)
                    and prev != _strip_base(prev)):  # prev phải CÓ dấu
                continue
            result.append(curr)
        text = "".join(result)

    return text


# ============================================================
# Bước 5: Tra từ điển — AN TOÀN
# Chỉ thay thế khi từ CHÍNH XÁC khớp (case-insensitive).
# Giữ nguyên viết hoa chữ đầu nếu từ gốc viết hoa.
# Giữ nguyên dấu câu dính liền từ (VD: "ko," → "không,").
# ============================================================

# Dấu câu có thể dính ở đầu/cuối từ
_LEADING_PUNCT_RE = re.compile(r"^([^\w]*)")
_TRAILING_PUNCT_RE = re.compile(r"([^\w]*)$")


def _transfer_case(original: str, replacement: str) -> str:
    """Chuyển kiểu viết hoa từ original sang replacement."""
    if not original or not replacement:
        return replacement
    if original[0].isupper() and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def restore_teencode(text: str) -> str:
    """
    Bước 5: Tra từ điển, thay thế teencode → dạng chuẩn.
    Tách dấu câu dính liền trước khi tra → gắn lại sau.
    """
    if not TEENCODE_MAP:
        return text

    words = text.split()
    result = []

    for w in words:
        # Tách dấu câu đầu/cuối
        lead = _LEADING_PUNCT_RE.match(w).group(1)
        trail = _TRAILING_PUNCT_RE.search(w).group(1)
        core = w[len(lead):len(w) - len(trail)] if trail else w[len(lead):]

        if not core:
            result.append(w)
            continue

        lookup = core.lower()
        if lookup in TEENCODE_MAP:
            replacement = _transfer_case(core, TEENCODE_MAP[lookup])
            result.append(lead + replacement + trail)
        else:
            result.append(w)

    return " ".join(result)


# ============================================================
# Bước 1: Unicode NFC
# ============================================================

def normalize_unicode(text: str) -> str:
    """Bước 1: Chuẩn hóa Unicode về dạng NFC."""
    return unicodedata.normalize("NFC", text)


# ============================================================
# Pipeline chính
# ============================================================

def normalize(text: str) -> str:
    """
    Chạy toàn bộ 5 bước chuẩn hóa.

    Thứ tự quan trọng:
      B1 NFC → B2 lookalike → B3 separator → B4 elongation → B5 dictionary
    Lý do: lookalike trước separator vì "m4.y" cần → "ma.y" → "may".
    Dictionary sau cùng vì cần input đã sạch separator + elongation.
    """
    text = normalize_unicode(text)           # B1
    text = map_lookalike_chars(text)          # B2
    text = remove_inserted_separators(text)   # B3
    text = collapse_elongation(text)          # B4
    text = restore_teencode(text)             # B5
    return text


# ============================================================
# Tiện ích bổ sung
# ============================================================

def get_dict_size() -> int:
    """Trả về số entry trong từ điển hiện tại."""
    return len(TEENCODE_MAP)


def reload_dict() -> None:
    """Reload từ điển từ file (dùng khi vừa chạy build_dictionary.py)."""
    global TEENCODE_MAP
    TEENCODE_MAP = _load_teencode_dict()


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print(f"Từ điển: {get_dict_size()} entries\n")

    test_cases = [
        # --- surface_obfuscation ---
        ("m4y n9u qu4",           "lookalike số→chữ"),
        ("n.g.u q.u.á",          "separator chèn thêm"),
        ("m4.y n9.u",            "lookalike + separator"),
        # --- expressive_lengthening ---
        ("nàoooo đẹppppp",       "kéo dài ký tự"),
        ("quáaaaa",              "kéo dài 1 từ"),
        # --- abbreviation_clipping ---
        ("ko bit j",             "viết tắt"),
        ("cx đc thui",           "viết tắt nhiều từ"),
        # --- diacritic_variation ---
        ("ngu qua",              "thiếu dấu"),
        # --- phonetic_spelling ---
        ("zậy fải k",            "viết theo phát âm"),
        # --- AN TOÀN: không sửa nhầm ---
        ("400k",                 "số thuần → giữ nguyên"),
        ("3.14",                 "số thập phân → giữ nguyên"),
        ("tôi thức dậy",        "từ hợp lệ → giữ nguyên"),
        ("con mèo",             "câu bình thường → giữ nguyên"),
        # --- dấu câu dính liền ---
        ("ko, tôi ko.",         "teencode + dấu câu"),
    ]

    for text, desc in test_cases:
        out = normalize(text)
        changed = " ✓" if out != text else " (giữ nguyên)"
        print(f"  {desc:30s} │ {text:25s} → {out}{changed}")
