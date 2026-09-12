"""
Bộ chuẩn hóa tiếng Việt chống né lọc (rule-based kết hợp từ điển teencode).

Xử lý 5 bước:
1. Chuẩn hóa Unicode NFC
2. Ánh xạ ký tự đồng dạng (lookalike)
3. Xóa ký tự phân cách xen kẽ chữ cái (.-_~·)
4. Rút gọn ký tự lặp kéo dài
5. Tra từ điển khôi phục teencode
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_DICT_DIR = Path(__file__).parent
_TEENCODE_DICT_PATH = _DICT_DIR / "teencode_dict.json"


def _load_teencode_dict() -> dict[str, str]:
    if not _TEENCODE_DICT_PATH.exists():
        return {}
    with open(_TEENCODE_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TEENCODE_MAP: dict[str, str] = _load_teencode_dict()

# Ánh xạ số/ký hiệu thay thế cho chữ cái tương tự
LOOKALIKE_MAP = {
    "4": "a", "3": "e", "1": "i", "0": "o",
    "5": "s", "9": "g", "8": "b", "@": "a",
}

_TOKEN_RE = re.compile(r"\S+")


def _map_lookalike_token(token: str) -> str:
    # Giữ nguyên số đo hoặc thông số kỹ thuật bắt đầu bằng số (ví dụ: 40km, 1TB)
    if token and token[0].isdigit():
        return token

    num_alpha = sum(1 for c in token if c.isalpha())
    num_convertible = sum(1 for c in token if c in LOOKALIKE_MAP)

    # Chỉ thay thế khi token chủ yếu là chữ cái
    if num_convertible == 0 or num_alpha <= num_convertible:
        return token

    return "".join(LOOKALIKE_MAP.get(ch, ch) for ch in token)


def map_lookalike_chars(text: str) -> str:
    """Thay thế ký tự đồng dạng trong từng token."""
    def replace_token(match):
        return _map_lookalike_token(match.group())
    return _TOKEN_RE.sub(replace_token, text)


# Chỉ xóa ký tự phân cách khi nó nằm giữa hai chữ cái tiếng Việt (ví dụ: n.g.u -> ngu, giữ nguyên 3.14)
SEPARATOR_PATTERN = re.compile(r"(?<=[a-zA-ZÀ-ỹ])[.\-_·~](?=[a-zA-ZÀ-ỹ])")


def remove_inserted_separators(text: str) -> str:
    """Xóa ký tự phân cách xen giữa các chữ cái."""
    return SEPARATOR_PATTERN.sub("", text)


ELONGATION_PATTERN = re.compile(r"([a-zA-ZÀ-ỹ])\1{2,}")


def _strip_base(char: str) -> str:
    """Lấy ký tự gốc không dấu (á -> a, ồ -> o)."""
    s = char.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def collapse_elongation(text: str) -> str:
    """
    Rút gọn ký tự lặp kéo dài (>= 3 lần) và bỏ ký tự gốc lặp phía sau ký tự có dấu.
    """
    text = ELONGATION_PATTERN.sub(r"\1", text)

    # Xóa ký tự gốc thừa sau ký tự có dấu (ví dụ: quáa -> quá)
    if len(text) >= 2:
        chars = list(text)
        result = [chars[0]]
        for i in range(1, len(chars)):
            prev = chars[i - 1]
            curr = chars[i]
            if (prev != curr
                    and curr.isalpha()
                    and prev.isalpha()
                    and _strip_base(prev) == _strip_base(curr)
                    and prev != _strip_base(prev)):
                continue
            result.append(curr)
        text = "".join(result)

    return text


_LEADING_PUNCT_RE = re.compile(r"^([^\w]*)")
_TRAILING_PUNCT_RE = re.compile(r"([^\w]*)$")


def _transfer_case(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    if original[0].isupper() and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


def restore_teencode(text: str) -> str:
    """Tra từ điển và thay thế từ teencode về dạng chuẩn."""
    if not TEENCODE_MAP:
        return text

    words = text.split()
    result = []

    for w in words:
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


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize(text: str) -> str:
    """
    Chạy tuần tự 5 bước chuẩn hóa:
    NFC -> Ký tự đồng dạng -> Ký tự phân cách -> Rút gọn lặp -> Từ điển teencode
    """
    text = normalize_unicode(text)
    text = map_lookalike_chars(text)
    text = remove_inserted_separators(text)
    text = collapse_elongation(text)
    text = restore_teencode(text)
    return text


def get_dict_size() -> int:
    return len(TEENCODE_MAP)


def reload_dict() -> None:
    global TEENCODE_MAP
    TEENCODE_MAP = _load_teencode_dict()


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"Từ điển: {get_dict_size()} mục\n")

    test_cases = [
        ("m4y n9u qu4", "Ký tự đồng dạng số sang chữ"),
        ("n.g.u q.u.á", "Ký tự phân cách xen kẽ"),
        ("m4.y n9.u", "Kết hợp đồng dạng và phân cách"),
        ("nàoooo đẹppppp", "Kéo dài ký tự"),
        ("quáaaaa", "Kéo dài sau ký tự có dấu"),
        ("ko bit j", "Viết tắt teencode"),
        ("cx đc thui", "Nhiều từ viết tắt"),
        ("ngu qua", "Thiếu dấu"),
        ("zậy fải k", "Viết theo phát âm"),
        ("400k", "Số và đơn vị giữ nguyên"),
        ("3.14", "Số thập phân giữ nguyên"),
        ("tôi thức dậy", "Câu bình thường giữ nguyên"),
        ("ko, tôi ko.", "Teencode kèm dấu câu"),
    ]

    for text, desc in test_cases:
        out = normalize(text)
        status = " (thay đổi)" if out != text else " (giữ nguyên)"
        print(f"  {desc:35s} | {text:25s} -> {out}{status}")
