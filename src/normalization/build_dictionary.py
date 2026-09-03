"""
Trích xuất từ điển teencode → dạng chuẩn từ bộ dữ liệu ViLexNorm.

Cách hoạt động:
  1. Đọc train.csv (cột original, normalized)
  2. Chỉ xử lý các dòng có cùng số từ (alignment 1:1)
  3. Strip dấu câu + emoticon trước khi so sánh
  4. Bỏ qua dạng kéo dài ký tự (xử lý bằng regex riêng)
  5. Đếm tần suất, lọc noise (>= min_freq lần)
  6. Loại các từ mơ hồ phụ thuộc ngữ cảnh
  7. Loại các entry mà original trùng với 1 từ tiếng Việt hợp lệ khác
  8. Nếu 1 từ map sang nhiều dạng chuẩn → chọn cái phổ biến nhất,
     nhưng nếu tỷ lệ top-1 < 70% tổng → loại luôn (quá mơ hồ)
  9. Xuất ra teencode_dict.json

Chạy:
    python src/normalization/build_dictionary.py
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# Cấu hình
# ============================================================
RAW_DATA_DIR = Path("data/raw/vilexnorm")
OUTPUT_PATH = Path("src/normalization/teencode_dict.json")
MIN_FREQ = 2        # chỉ giữ cặp xuất hiện >= 2 lần
DOMINANCE = 0.70     # nếu ứng viên top-1 chiếm < 70% tổng → loại (quá mơ hồ)

# Dấu câu + emoticon fragments cần strip khỏi đầu/cuối từ
PUNCT_CHARS = set(string.punctuation + "…""''）（）【】")
EMOTICON_RE = re.compile(r"^[:;=xX><]+[)(DPpOo3/\\|]+$")

# Từ mơ hồ: đúng/sai tùy ngữ cảnh → loại khỏi từ điển để tránh sửa nhầm.
# Nguyên tắc: nếu từ gốc LÀ 1 từ tiếng Việt hợp lệ trong ngữ cảnh khác
# thì KHÔNG nên tự động thay thế.
AMBIGUOUS_BLOCKLIST = {
    # 1 ký tự: quá ngắn, dễ trùng tên riêng / biến / ký hiệu
    "r", "d", "e", "a", "i", "b", "c", "g", "h", "l", "m", "n", "o", "s",
    "t", "v", "x", "y", "z", "u", "p", "f", "w", "q",
    # Từ có nghĩa riêng hợp lệ
    "dậy",    # thức dậy ≠ vậy
    "hay",    # hay (tính từ) ≠ hãy
    "kia",    # kia (đại từ) — đôi khi bị map nhầm
    "coi",    # coi (xem) — phương ngữ hợp lệ
    "hơi",    # hơi (trạng từ)
    "roi",    # roi (danh từ) ≠ rồi
    "hen",    # hen (bệnh hen) ≠ hẹn
    "no",     # no (no bụng) ≠ nó
    "con",    # con (danh từ)
    "cho",    # cho (động từ)
    "chua",   # chua (vị chua) ≠ chưa
    "ma",     # ma (danh từ) ≠ mà
    "mua",    # mua (mua bán) ≠ múa
    "dam",    # dam (đầm) ≠ dám/đám
    "bang",   # bang (tiểu bang) ≠ bằng/bảng
    "long",   # long (long lanh) ≠ lòng/lông
    "nha",    # nha (nha sĩ) ≠ nhà/nha
    "mang",   # mang (mang vác) ≠ măng
    "co",     # co (co rúm) ≠ có
    "cung",   # cung (cung cấp) ≠ cũng
    "ban",    # ban (ban bệ) ≠ bạn/bản
    "lam",    # lam (màu lam) ≠ làm
    "to",     # to (to lớn) ≠ tớ
    "la",     # la (la hét) ≠ là
    "lo",     # lo (lo lắng) ≠ lờ
}


# ============================================================
# Helpers
# ============================================================

def strip_punct(word: str) -> str:
    """Bỏ dấu câu ở đầu và cuối từ, giữ nguyên phần giữa."""
    start = 0
    end = len(word)
    while start < end and word[start] in PUNCT_CHARS:
        start += 1
    while end > start and word[end - 1] in PUNCT_CHARS:
        end -= 1
    return word[start:end]


def is_emoticon(word: str) -> bool:
    """Kiểm tra từ có phải emoticon text không. VD: :)), =)), :D, @@"""
    stripped = word.strip()
    if not stripped:
        return False
    # Chỉ gồm ký tự đặc biệt lặp lại
    if all(c in ":;)(DPpOo@=#>/\\|3><^~*" for c in stripped):
        return True
    return bool(EMOTICON_RE.match(stripped))


def is_elongated_pair(orig: str, norm: str) -> bool:
    """
    Kiểm tra orig có phải là dạng kéo dài ký tự của norm không.
    VD: "nàoooo" → "nào", "quáaaa" → "quá"
    """
    if len(orig) <= len(norm) or not norm:
        return False
    last_char = orig[-1]
    repeat_count = 0
    for ch in reversed(orig):
        if ch == last_char:
            repeat_count += 1
        else:
            break
    if repeat_count < 2:
        return False
    trimmed = orig[:len(orig) - repeat_count + 1]
    return trimmed == norm


def is_only_diacritic_diff(orig: str, norm: str) -> bool:
    """
    Kiểm tra 2 từ chỉ khác nhau ở dấu thanh/phụ.
    VD: "ngu" vs "ngủ", "qua" vs "quá"
    Dùng để GIỮA LẠI entry — đây là dạng diacritic_variation, an toàn để sửa.
    """
    import unicodedata
    def strip_diacritics(s):
        s = s.replace("đ", "d").replace("Đ", "D")
        nfkd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfkd if not unicodedata.combining(c))
    return strip_diacritics(orig) == strip_diacritics(norm) and orig != norm


# ============================================================
# Core logic
# ============================================================

def load_data(data_dir: Path) -> pd.DataFrame:
    """Đọc train.csv."""
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {train_path}")
    df = pd.read_csv(train_path)
    print(f"Đọc được {len(df)} dòng từ {train_path}")
    return df


def extract_word_pairs(df: pd.DataFrame) -> Counter:
    """
    So sánh từng cặp từ giữa original và normalized (alignment 1:1).
    Strip dấu câu, bỏ emoticon, bỏ elongation.
    """
    pair_counter = Counter()
    same_length = 0
    diff_length = 0
    elongated_skipped = 0
    emoticon_skipped = 0

    for _, row in df.iterrows():
        orig = str(row["original"]).strip()
        norm = str(row["normalized"]).strip()

        orig_tokens = orig.split()
        norm_tokens = norm.split()

        if len(orig_tokens) != len(norm_tokens):
            diff_length += 1
            continue

        same_length += 1

        for o_word, n_word in zip(orig_tokens, norm_tokens):
            # Bỏ emoticon
            if is_emoticon(o_word) or is_emoticon(n_word):
                emoticon_skipped += 1
                continue

            # Strip dấu câu
            o_clean = strip_punct(o_word).lower()
            n_clean = strip_punct(n_word).lower()

            if not o_clean or not n_clean or o_clean == n_clean:
                continue

            # Bỏ elongation (xử lý bằng regex rule riêng)
            if is_elongated_pair(o_clean, n_clean):
                elongated_skipped += 1
                continue

            pair_counter[(o_clean, n_clean)] += 1

    total = same_length + diff_length
    print(f"\n--- Thống kê alignment ---")
    print(f"Cùng số từ (dùng được):  {same_length}/{total} ({same_length/total*100:.1f}%)")
    print(f"Khác số từ (bỏ qua):    {diff_length}/{total} ({diff_length/total*100:.1f}%)")
    print(f"Elongation (bỏ):         {elongated_skipped}")
    print(f"Emoticon (bỏ):            {emoticon_skipped}")
    print(f"Tổng cặp từ khác nhau:  {len(pair_counter)}")

    return pair_counter


def build_dictionary(pair_counter: Counter, min_freq: int, dominance: float) -> dict[str, str]:
    """
    Xây từ điển 1-1 từ Counter.
    Lọc: freq < min_freq, ambiguous blocklist, dominance < threshold.
    """
    # Nhóm: orig → [(norm, freq), ...]
    orig_to_norms: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)

    for (orig, norm), freq in pair_counter.items():
        if freq < min_freq:
            continue
        if orig in AMBIGUOUS_BLOCKLIST:
            continue
        orig_to_norms[orig].append((norm, freq))

    dictionary = {}
    conflicts = 0
    dropped_ambiguous = 0
    conflict_details = []

    for orig, candidates in orig_to_norms.items():
        candidates.sort(key=lambda x: x[1], reverse=True)
        total_freq = sum(f for _, f in candidates)
        best_norm, best_freq = candidates[0]

        if len(candidates) > 1:
            conflicts += 1
            conflict_details.append((orig, candidates))

            # Nếu ứng viên top-1 không chiếm đa số → quá mơ hồ, bỏ
            if best_freq / total_freq < dominance:
                dropped_ambiguous += 1
                continue

        dictionary[orig] = best_norm

    print(f"\n--- Kết quả xây từ điển ---")
    print(f"Tổng entry sau lọc (freq >= {min_freq}): {len(dictionary)}")
    print(f"Từ trong blocklist bị loại:              {len(AMBIGUOUS_BLOCKLIST)}")
    print(f"Từ có nhiều ứng viên:                    {conflicts}")
    print(f"  → Loại vì quá mơ hồ (top1 < {dominance:.0%}):  {dropped_ambiguous}")

    if conflict_details[:15]:
        print(f"\n--- Mẫu từ có nhiều ứng viên ---")
        for orig, candidates in conflict_details[:15]:
            total = sum(f for _, f in candidates)
            cands = ", ".join(f'"{n}" ({f}/{total})' for n, f in candidates)
            print(f'  "{orig}" → {cands}')

    return dictionary


def show_sample(dictionary: dict[str, str], n: int = 40):
    """In mẫu."""
    print(f"\n--- Mẫu {n} entry ---")
    for i, (k, v) in enumerate(sorted(dictionary.items())):
        if i >= n:
            break
        print(f"  {k:20s} → {v}")


def show_stats(dictionary: dict[str, str]):
    """Thống kê."""
    lens = [len(k) for k in dictionary]
    print(f"\n--- Phân bố độ dài từ gốc ---")
    for bucket in [1, 2, 3, 4]:
        label = f"{bucket} ký tự" if bucket < 4 else "4+ ký tự"
        count = sum(1 for l in lens if (l == bucket if bucket < 4 else l >= bucket))
        print(f"  {label}: {count}")


def main():
    df = load_data(RAW_DATA_DIR)
    pair_counter = extract_word_pairs(df)
    dictionary = build_dictionary(pair_counter, min_freq=MIN_FREQ, dominance=DOMINANCE)
    show_sample(dictionary)
    show_stats(dictionary)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\n✓ Đã lưu từ điển ({len(dictionary)} entries) vào {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
