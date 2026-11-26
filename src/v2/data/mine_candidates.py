"""Lọc thô ứng viên DE_DOA, QUAY_ROI, LUA_DAO từ kho câu đã gộp (merged.jsonl) bằng từ khóa.

Đây chỉ là bước khoanh vùng để LLM gán sơ bộ (prelabel.py --input ...) rồi người xác nhận. Khớp từ khóa
KHÔNG phải nhãn. So khớp trên chữ thường đã bỏ dấu, nên bắt được cả câu gõ không dấu.

Tùy chọn --voz: lấy thêm CÂU THÔ của VOZ-HSD (không lấy nhãn, đúng quy tắc trong CLAUDE.md), nguon="voz".

Chạy:
    python -m src.v2.data.mine_candidates
    python -m src.v2.data.mine_candidates --voz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from src.v2.data.merge import OUT_PATH as MERGED_PATH, VIHSD_DIR, SPLITS, is_lfs_pointer, make_record

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "data" / "v2" / "interim" / "ung_vien.jsonl"
VOZ_SOURCE = "tarudesu/VOZ-HSD"

# Từ khóa viết không dấu, chữ thường. Ưu tiên cụm nhiều chữ để bớt khớp nhầm.
TU_KHOA: dict[str, tuple[str, ...]] = {
    "DE_DOA": (
        "biet nha may", "biet nha mi", "den nha may", "toi nha may", "tao danh", "danh chet", "dap chet",
        "giet may", "giet chet", "cho may chet", "may chet", "xu may", "xu dep", "chem chet", "dot nha",
        "tung anh", "tung clip", "tung video", "gui clip", "phat tan", "dang len group", "dang len nhom",
        "coi chung tao", "liu hon", "cho day", "ra sau truong", "cong truong", "chan duong", "chan dau",
        "khong tra tien", "tra tien di", "khong tra no", "bao nha truong", "goi cho bo me may",
        "tao biet truong may", "an dao", "an dam", "dap cho",
    ),
    "QUAY_ROI": (
        "gui anh", "anh nong", "anh nude", "anh khoa than", "khoa than", "khong mac gi", "di khach san",
        "nha nghi", "ngu voi", "qua dem", "o lai dem", "seen khong rep", "seen ma khong", "khong rep",
        "sao khong tra loi", "sao chan anh", "chan anh", "nick khac", "vong 1", "vong mot", "vong 3",
        "nguc", "mong to", "beo nhu", "map nhu", "xau nhu", "ngoai tinh", "cap ke", "ngu voi sep",
        "dung ke voi ai", "giu bi mat", "gap rieng",
    ),
    "LUA_DAO": (
        "chuyen khoan", "ck truoc", "chuyen truoc", "dat coc", "nap the", "ma the", "the cao", "otp",
        "ma xac nhan", "dang nhap", "xac minh tai khoan", "trung thuong", "nhan qua", "phi ship",
        "phi nhan", "viec nhe luong cao", "ctv", "cong tac vien", "kiem tien online", "lam nhiem vu",
        "dau tu", "lai suat", "hoa hong", "vay nhanh", "vay tien", "giai ngan", "kim cuong free",
        "robux", "quan huy", "giveaway", "minigame", "hoc bong", "le phi", "giu suat", "de thi",
    ),
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def _compile(keywords: Iterable[str]) -> re.Pattern:
    alternatives = sorted({re.escape(k) for k in keywords}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(alternatives) + r")(?!\w)")


PATTERNS = {label: _compile(words) for label, words in TU_KHOA.items()}


def match_keywords(text: str) -> dict[str, list[str]]:
    """{nhãn ứng viên: [từ khóa khớp]} cho một câu."""
    plain = strip_accents(text)
    hits = {}
    for label, pattern in PATTERNS.items():
        found = sorted(set(pattern.findall(plain)))
        if found:
            hits[label] = found
    return hits


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_voz_raw(directory: Path = VIHSD_DIR) -> Iterator[dict]:
    """Câu thô VOZ-HSD. KHÔNG giữ nhãn VOZ (nhan_goc để trống)."""
    seen: set[str] = set()
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists() or is_lfs_pointer(path):
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if VOZ_SOURCE not in line:
                    continue
                row = json.loads(line)
                if row.get("source_dataset") != VOZ_SOURCE:
                    continue
                key = f"{row['source_split']}:{row['source_row_id']}"
                text = row.get("original_text") or ""
                if key in seen or not text.strip():
                    continue
                seen.add(key)
                record = make_record(text=text, nguon="voz", nguon_id=key, nhan_goc="", split_goc="")
                yield record


def mine(records: Iterable[dict]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    seen_text: set[str] = set()
    by_label: Counter = Counter()
    by_source_label: Counter = Counter()
    by_keyword: Counter = Counter()
    total = 0
    for record in records:
        total += 1
        if record["text"] in seen_text:
            continue
        hits = match_keywords(record["text"])
        if not hits:
            continue
        seen_text.add(record["text"])
        result = dict(record)
        result["ung_vien"] = sorted(hits)
        result["tu_khoa"] = hits
        out.append(result)
        for label, words in hits.items():
            by_label[label] += 1
            by_source_label[f"{record['nguon']}/{label}"] += 1
            by_keyword.update(f"{label}:{w}" for w in words)
    stats = {
        "so_cau_da_quet": total,
        "so_ung_vien": len(out),
        "theo_nhan_ung_vien": dict(by_label.most_common()),
        "theo_nguon_va_nhan": dict(sorted(by_source_label.items())),
        "tu_khoa_hay_gap": dict(by_keyword.most_common(40)),
    }
    return out, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=MERGED_PATH)
    parser.add_argument("--voz", action="store_true", help="quét thêm câu thô VOZ-HSD (không dùng nhãn)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sources: list[Iterable[dict]] = [read_jsonl(args.input)]
    if args.voz:
        sources.append(iter_voz_raw())
    candidates, stats = mine(r for source in sources for r in source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in candidates:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.out.with_name(args.out.stem + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Đã ghi {len(candidates)} ứng viên vào {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
