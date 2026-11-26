"""
Bộ sinh biến thể né lọc có tham số cường độ.

Gồm 6 nhóm biến thể:
  1. remove_diacritics       - loại bỏ dấu
  2. teencode_substitute     - biến thể teencode
  3. insert_separators       - chèn ký tự phân tách
  4. lookalike_substitute    - thay ký tự đồng dạng
  5. unicode_homoglyph       - ký tự Unicode nhìn giống
  6. spelling_noise          - nhiễu chính tả có chủ đích

Mỗi hàm nhận tham số `intensity` (0.0 - 1.0): tỷ lệ ký tự/từ bị tác động,
dùng để vẽ đường cong suy giảm hiệu năng theo cường độ (Mục 4.3).

Nhãn gốc được giữ nguyên cho mọi biến thể sinh ra.
"""

from __future__ import annotations
import random
import unicodedata

random.seed(42)  # tái lập kết quả khi sinh benchmark

SEPARATORS = [".", "-", "_", " "]

# Vài ký tự Cyrillic/Hy Lạp trông giống hệt Latin, chỉ là ví dụ minh họa,
# nên mở rộng bảng này cho benchmark thật.
HOMOGLYPH_MAP = {
    "a": "а",  # Cyrillic а (U+0430)
    "e": "е",  # Cyrillic е (U+0435)
    "o": "о",  # Cyrillic о (U+043E)
}

LOOKALIKE_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "g": "9", "b": "8"}


def remove_diacritics(text: str, intensity: float = 1.0) -> str:
    """Nhóm 1: bỏ dấu thanh/dấu phụ. đ/Đ xử lý riêng vì không phân rã qua NFD."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    chars = []
    for ch in decomposed:
        if unicodedata.combining(ch) and random.random() < intensity:
            continue  # bỏ dấu kết hợp theo xác suất = intensity
        chars.append(ch)
    return unicodedata.normalize("NFC", "".join(chars))


def teencode_substitute(text: str, intensity: float = 1.0, teencode_map: dict | None = None) -> str:
    """Nhóm 2: thay từ chuẩn bằng biến thể teencode theo từ điển."""
    teencode_map = teencode_map or {}
    words = text.split()
    out = []
    for w in words:
        variants = teencode_map.get(w.lower())
        if variants and random.random() < intensity:
            out.append(random.choice(variants) if isinstance(variants, list) else variants)
        else:
            out.append(w)
    return " ".join(out)


def insert_separators(text: str, intensity: float = 1.0) -> str:
    """Nhóm 3: chèn ký tự phân tách ngẫu nhiên giữa các ký tự trong từ."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and random.random() < intensity:
            out.append(random.choice(SEPARATORS))
    return "".join(out)


def lookalike_substitute(text: str, intensity: float = 1.0) -> str:
    """Nhóm 4: thay chữ cái bằng số/ký hiệu hình dạng tương tự."""
    return "".join(
        LOOKALIKE_MAP[ch] if ch.lower() in LOOKALIKE_MAP and random.random() < intensity else ch
        for ch in text
    )


def unicode_homoglyph(text: str, intensity: float = 1.0) -> str:
    """Nhóm 5: thay ký tự bằng ký tự trông giống hệt từ bảng mã Unicode khác."""
    return "".join(
        HOMOGLYPH_MAP[ch] if ch in HOMOGLYPH_MAP and random.random() < intensity else ch
        for ch in text
    )


def spelling_noise(text: str, intensity: float = 1.0) -> str:
    """Nhóm 6: nhiễu chính tả có chủ đích - đảo/lặp/bỏ ký tự."""
    chars = list(text)
    i = 0
    out = []
    while i < len(chars):
        ch = chars[i]
        if chars[i].isalpha() and random.random() < intensity:
            action = random.choice(["duplicate", "drop", "swap"])
            if action == "duplicate":
                out.append(ch)
                out.append(ch)
            elif action == "drop":
                pass  # bỏ ký tự
            elif action == "swap" and i + 1 < len(chars):
                out.append(chars[i + 1])
                out.append(ch)
                i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out)


VARIANT_FUNCTIONS = {
    "remove_diacritics": remove_diacritics,
    "teencode_substitute": teencode_substitute,
    "insert_separators": insert_separators,
    "lookalike_substitute": lookalike_substitute,
    "unicode_homoglyph": unicode_homoglyph,
    "spelling_noise": spelling_noise,
}


def generate_variant(text: str, variant_type: str, intensity: float = 1.0, **kwargs) -> str:
    """Điểm vào chung: sinh 1 biến thể theo tên nhóm + cường độ."""
    fn = VARIANT_FUNCTIONS[variant_type]
    return fn(text, intensity=intensity, **kwargs)


if __name__ == "__main__":
    sample = "mày ngu quá"
    for name in VARIANT_FUNCTIONS:
        if name == "teencode_substitute":
            continue  # cần truyền teencode_map riêng
        print(f"{name:22s} -> {generate_variant(sample, name, intensity=0.8)}")
