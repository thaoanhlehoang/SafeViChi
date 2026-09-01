"""
Lớp chuẩn hóa chống né lọc (rule-based).

Tương ứng Mục 3.4 trong docs/plan.md. Gồm 4 bước, theo đúng thứ tự:
  1. Chuẩn hóa Unicode (NFC)
  2. Ánh xạ ký tự đồng dạng về ký tự Latin gốc
  3. Loại bỏ ký tự phân tách chèn thêm (regex)
  4. Khôi phục dấu thanh và giải teencode (tra từ điển biến thể -> dạng chuẩn)

Không dùng mạng nơ-ron ở phiên bản demo (xem Mục 7 của plan cho hướng
mở rộng bằng neural denoiser, tham khảo ViSoLex).
"""

from __future__ import annotations
import re
import unicodedata


# TODO: mở rộng bảng ánh xạ này, có thể bootstrap từ ViLexNorm
# (https://arxiv.org/pdf/2401.16403) thay vì gõ tay toàn bộ.
LOOKALIKE_MAP = {
    "4": "a",
    "3": "e",
    "1": "i",
    "0": "o",
    "5": "s",
    "9": "g",
    "8": "b",
}

# TODO: mở rộng từ điển teencode -> dạng chuẩn.
TEENCODE_MAP = {
    "ko": "không",
    "k": "không",
    "j": "gì",
    "wa": "quá",
    "wá": "quá",
    "bit": "biết",
    "iu": "yêu",
    "vs": "với",
}

SEPARATOR_PATTERN = re.compile(r"(?<=\w)[.\-_ ](?=\w)")


def normalize_unicode(text: str) -> str:
    """Bước 1: chuẩn hóa Unicode về dạng NFC."""
    return unicodedata.normalize("NFC", text)


def map_lookalike_chars(text: str) -> str:
    """Bước 2: ánh xạ ký tự đồng dạng (số/ký hiệu) về chữ cái Latin gốc."""
    return "".join(LOOKALIKE_MAP.get(ch, ch) for ch in text)


def remove_inserted_separators(text: str) -> str:
    """Bước 3: loại bỏ ký tự phân tách chèn thêm giữa các ký tự trong từ."""
    # TODO: cẩn thận không xóa nhầm khoảng trắng hợp lệ giữa các TỪ.
    # Cần thuật toán tinh hơn regex đơn giản này cho bản chính thức.
    return SEPARATOR_PATTERN.sub("", text)


def restore_teencode(text: str) -> str:
    """Bước 4: tra từ điển để giải teencode / khôi phục dấu."""
    words = text.split()
    return " ".join(TEENCODE_MAP.get(w.lower(), w) for w in words)


def normalize(text: str) -> str:
    """Chạy toàn bộ 4 bước chuẩn hóa theo đúng thứ tự trong plan."""
    text = normalize_unicode(text)
    text = map_lookalike_chars(text)
    text = remove_inserted_separators(text)
    text = restore_teencode(text)
    return text


if __name__ == "__main__":
    sample = "m4y n9u qu4"
    print(f"Input : {sample}")
    print(f"Output: {normalize(sample)}")
