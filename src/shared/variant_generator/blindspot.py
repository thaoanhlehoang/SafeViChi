"""Sinh biến thể né lọc nằm NGOÀI vùng phủ của bộ chuẩn hóa 5 bước.

Khác với controlled.py (12 rule, đã dùng để build 75K dataset và cũng là các
rule mà normalizer được thiết kế để xử lý), module này cố tình khai thác đúng
những gì normalizer KHÔNG đụng tới:

  Kỹ thuật                  │ Vì sao normalizer bó tay
  ──────────────────────────┼──────────────────────────────────────────────
  cyrillic_homoglyph        │ B2 LOOKALIKE_MAP chỉ map số/@ → chữ, không có
                            │ bảng Cyrillic; а(U+0430) là isalpha() nên B4
                            │ regex [a-zA-ZÀ-ỹ] cũng không khớp
  zero_width                │ B3 chỉ xóa . - _ ; ZWSP/ZWNJ không nằm trong đó
  exotic_separator          │ B3 chỉ xóa . - _ ; * / | + ~ ^ thì không
  space_injection           │ B3 CỐ TÌNH không xóa khoảng trắng (sợ merge từ)
  double_elongation         │ B4 chỉ gộp khi >= 3 ký tự lặp; đúng 2 thì giữ
  fullwidth                 │ B1 dùng NFC, không phải NFKC — fullwidth sống sót

Mọi câu sinh ra PHẢI qua kiểm tra normalize(x) == x trước khi dùng: nếu
normalizer sửa được thì đó không còn là điểm mù, phải xếp vào nhóm dữ liệu
đã-chuẩn-hóa thay vì nhóm này (xem is_blindspot).
"""

from __future__ import annotations

import random
import re

BLINDSPOT_VERSION = "1.0.0"

# Cyrillic/Hy Lạp trông giống hệt Latin trong hầu hết font
CYRILLIC_HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о", "c": "с", "p": "р",
    "y": "у", "x": "х", "k": "к", "s": "ѕ", "i": "і",
    "A": "А", "E": "Е", "O": "О", "C": "С", "P": "Р",
    "X": "Х", "K": "К", "B": "В", "H": "Н", "M": "М",
}

ZERO_WIDTH_CHARS = ("​", "‌", "‍", "﻿")

# Separator mà B3 của normalizer không xóa (B3 chỉ xử lý . - _)
EXOTIC_SEPARATORS = ("*", "/", "|", "+", "~", "^", "'", "`", ":")

_WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]{3,}")


def _pick_word_spans(text: str, rng: random.Random, max_words: int) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not spans:
        return []
    rng.shuffle(spans)
    return sorted(spans[:max_words])


def cyrillic_homoglyph(text: str, rng: random.Random, intensity: float = 0.4) -> str:
    out = []
    for ch in text:
        if ch in CYRILLIC_HOMOGLYPHS and rng.random() < intensity:
            out.append(CYRILLIC_HOMOGLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


def zero_width(text: str, rng: random.Random, max_words: int = 3) -> str:
    spans = _pick_word_spans(text, rng, max_words)
    if not spans:
        return text
    out = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        word = text[start:end]
        cut = rng.randrange(1, len(word))
        out.append(word[:cut] + rng.choice(ZERO_WIDTH_CHARS) + word[cut:])
        prev = end
    out.append(text[prev:])
    return "".join(out)


def exotic_separator(text: str, rng: random.Random, max_words: int = 3) -> str:
    spans = _pick_word_spans(text, rng, max_words)
    if not spans:
        return text
    out = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        word = text[start:end]
        cut = rng.randrange(1, len(word))
        out.append(word[:cut] + rng.choice(EXOTIC_SEPARATORS) + word[cut:])
        prev = end
    out.append(text[prev:])
    return "".join(out)


def space_injection(text: str, rng: random.Random, max_words: int = 2) -> str:
    spans = _pick_word_spans(text, rng, max_words)
    if not spans:
        return text
    out = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        word = text[start:end]
        cut = rng.randrange(1, len(word))
        out.append(word[:cut] + " " + word[cut:])
        prev = end
    out.append(text[prev:])
    return "".join(out)


def double_elongation(text: str, rng: random.Random, max_words: int = 3) -> str:
    """Lặp ĐÚNG 2 lần — B4 chỉ gộp khi >= 3 nên bản này sống sót."""
    spans = _pick_word_spans(text, rng, max_words)
    if not spans:
        return text
    out = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        word = text[start:end]
        # chỉ nhân đôi ký tự KHÔNG dấu, và ký tự sau nó phải khác nó
        candidates = [
            i for i, c in enumerate(word)
            if c.isascii() and c.isalpha() and (i + 1 >= len(word) or word[i + 1] != c)
            and (i == 0 or word[i - 1] != c)
        ]
        if candidates:
            i = rng.choice(candidates)
            word = word[: i + 1] + word[i] + word[i + 1:]
        out.append(word)
        prev = end
    out.append(text[prev:])
    return "".join(out)


def fullwidth(text: str, rng: random.Random, intensity: float = 0.35) -> str:
    """ASCII → fullwidth (U+FF01..U+FF5E). NFC không đụng tới (chỉ NFKC mới đổi)."""
    out = []
    for ch in text:
        if "!" <= ch <= "~" and ch.isalpha() and rng.random() < intensity:
            out.append(chr(ord(ch) + 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


TECHNIQUES = {
    "cyrillic_homoglyph": cyrillic_homoglyph,
    "zero_width": zero_width,
    "exotic_separator": exotic_separator,
    "space_injection": space_injection,
    "double_elongation": double_elongation,
    "fullwidth": fullwidth,
}


def is_blindspot(text: str, normalize_fn) -> bool:
    """True khi normalizer KHÔNG sửa được gì — tức đúng là điểm mù."""
    return normalize_fn(text) == text


def generate_blindspot(
    text: str,
    normalize_fn,
    rng: random.Random,
    max_attempts: int = 6,
) -> tuple[str, list[str]] | None:
    """Sinh 1 biến thể điểm mù. Trả None nếu không đạt sau max_attempts lần.

    Mỗi lần thử áp 1-2 kỹ thuật ngẫu nhiên rồi kiểm tra normalize(x) == x;
    chỉ trả về khi (a) văn bản thực sự đổi so với gốc, và (b) normalizer
    không sửa được gì.
    """
    names = list(TECHNIQUES)
    for _ in range(max_attempts):
        chosen = rng.sample(names, k=rng.choice((1, 2)))
        out = text
        for name in chosen:
            out = TECHNIQUES[name](out, rng)
        if out != text and is_blindspot(out, normalize_fn):
            return out, chosen
    return None
