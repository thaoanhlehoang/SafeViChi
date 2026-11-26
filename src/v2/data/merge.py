"""Gộp ViHSD, ViHOS, ViTHSD về định dạng chung của v2 (xem CLAUDE.md).

Mỗi bản ghi đầu ra để trống `nhan` (chờ LLM và người gán). Nhãn của bộ gốc giữ ở `nhan_goc`,
split của bộ gốc giữ ở `split_goc`. Với ViHOS, các từ liên tiếp có span_mask=1 được ghép thành
`bang_chung_goi_y` (gợi ý cho người duyệt, KHÔNG phải bằng chứng đã chốt).

KHÔNG dùng nhãn của VOZ-HSD: chỉ lấy các dòng có source_dataset là uitnlp/vihsd.

Chạy:
    python -m src.v2.data.merge
    python -m src.v2.data.merge --vithsd-dir data/raw/vithsd --out data/v2/interim/merged.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[3]

VIHSD_DIR = ROOT / "data" / "processed" / "vietnamese_nonstandard_v1_2"
VIHOS_DIR = ROOT / "data" / "data_explain"
VITHSD_DIR = ROOT / "data" / "raw" / "vithsd"
OUT_PATH = ROOT / "data" / "v2" / "interim" / "merged.jsonl"

SPLITS = ("train", "validation", "test")
VIHSD_SOURCE = "uitnlp/vihsd"
LFS_POINTER_PREFIX = "version https://git-lfs"

# Nhãn gốc được coi là "sạch" hoặc "nguy hại" để so với nhãn LLM (dùng trong prelabel).
NHAN_GOC_SACH = {"CLEAN"}
NHAN_GOC_NGUY_HAI = {"OFFENSIVE", "HATE"}


def normalize_text(text: str) -> str:
    """NFC và gộp khoảng trắng (kể cả xuống dòng) thành một dòng. Không đổi chữ nào khác."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_lines(text: str) -> str:
    """Chuẩn hóa một đoạn hội thoại: mỗi dòng một lượt nói, chuẩn hóa từng dòng, bỏ dòng trống, giữ xuống dòng."""
    lines = [normalize_text(line) for line in unicodedata.normalize("NFC", text).splitlines()]
    return "\n".join(line for line in lines if line)


def make_id(nguon: str, nguon_id: str) -> str:
    """id ổn định, không phụ thuộc thứ tự đọc: thêm nguồn mới không làm đổi id cũ."""
    digest = hashlib.sha1(f"{nguon}|{nguon_id}".encode("utf-8")).hexdigest()[:12]
    return f"v2_{digest}"


def make_record(
    *,
    text: str,
    nguon: str,
    nguon_id: str,
    nhan_goc: str,
    split_goc: str,
    bang_chung_goi_y: list[str] | None = None,
    nhan_goc_chi_tiet: dict | None = None,
    nhieu_dong: bool = False,
) -> dict:
    """Bản ghi định dạng chung v2. Câu của bộ dữ liệu cũ là hội thoại 1 dòng (nhieu_dong=False gộp mọi
    xuống dòng). Hội thoại nhiều dòng: mỗi dòng là toàn bộ lượt nói của một người, các dòng luân phiên hai người."""
    text = normalize_lines(text) if nhieu_dong else normalize_text(text)
    so_dong = text.count("\n") + 1 if text else 0
    record = {
        "id": make_id(nguon, nguon_id),
        "text": text,
        "nhan": "",
        "bang_chung": [],
        "nguon": nguon,
        "nguon_id": nguon_id,
        "cach_gan": "",
        "the": [],
        "so_dong": so_dong,
        "bien_dang": "C0",
        "nhom_trung": "",
        "split": "",
        "nhan_goc": nhan_goc,
        "split_goc": split_goc,
        "bang_chung_goi_y": bang_chung_goi_y or [],
    }
    if nhan_goc_chi_tiet is not None:
        record["nhan_goc_chi_tiet"] = nhan_goc_chi_tiet
    return record


def is_lfs_pointer(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


# ---------------------------------------------------------------- ViHSD

def iter_vihsd(directory: Path = VIHSD_DIR) -> Iterator[dict]:
    """Câu gốc ViHSD (original_text, nhãn 3 mức) từ bộ v1_2; mỗi dòng nguồn lấy đúng một lần."""
    seen: set[tuple[str, str]] = set()
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}")
        if is_lfs_pointer(path):
            raise RuntimeError(f"{path} là con trỏ Git LFS. Chạy `git lfs pull` trước.")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if VIHSD_SOURCE not in line:  # lọc nhanh, bỏ VOZ-HSD mà không parse
                    continue
                row = json.loads(line)
                if row.get("source_dataset") != VIHSD_SOURCE:
                    continue
                key = (str(row["source_split"]), str(row["source_row_id"]))
                if key in seen:
                    continue
                seen.add(key)
                text = row.get("original_text") or ""
                if not text.strip():
                    continue
                yield make_record(
                    text=text,
                    nguon="vihsd",
                    nguon_id=f"{key[0]}:{key[1]}",
                    nhan_goc=str(row["label_original"]).upper(),
                    split_goc=key[0],
                )


VIHSD_RAW_DIR = ROOT / "data" / "raw" / "vihsd"
VIHSD_RAW_FILES = {"train": "train.csv", "validation": "dev.csv", "test": "test.csv"}
VIHSD_LABELS = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def vihsd_raw_available(directory: Path = VIHSD_RAW_DIR) -> bool:
    return all((directory / name).exists() for name in VIHSD_RAW_FILES.values())


def iter_vihsd_raw(directory: Path = VIHSD_RAW_DIR) -> Iterator[dict]:
    """ViHSD đủ 3 split từ CSV gốc (tải bằng src.v2.data.download). nguon_id = '<split>:<số dòng từ 0>',
    trùng cách đánh số của v1_2 với split train nên id cũ không đổi."""
    for split, name in VIHSD_RAW_FILES.items():
        path = directory / name
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["free_text", "label_id"]:
                raise ValueError(f"{path}: cột lạ {reader.fieldnames!r}, cần ['free_text', 'label_id']")
            for index, row in enumerate(reader):
                text = row["free_text"] or ""
                if not text.strip():
                    continue
                yield make_record(
                    text=text,
                    nguon="vihsd",
                    nguon_id=f"{split}:{index}",
                    nhan_goc=VIHSD_LABELS[int(row["label_id"])],
                    split_goc=split,
                )


# ---------------------------------------------------------------- ViHOS

def mask_to_phrases(words: list[str], span_mask: list[int]) -> list[str]:
    """Ghép các từ liên tiếp có mask=1 thành cụm, giữ thứ tự xuất hiện, bỏ cụm trùng."""
    if len(words) != len(span_mask):
        raise ValueError(f"words ({len(words)}) và span_mask ({len(span_mask)}) lệch độ dài")
    phrases: list[str] = []
    current: list[str] = []
    for word, flag in zip(words, span_mask):
        if flag:
            current.append(word)
        elif current:
            phrases.append(" ".join(current))
            current = []
    if current:
        phrases.append(" ".join(current))
    unique: list[str] = []
    for phrase in phrases:
        phrase = normalize_text(phrase)
        if phrase and phrase not in unique:
            unique.append(phrase)
    return unique


def iter_vihos(directory: Path = VIHOS_DIR) -> Iterator[dict]:
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}")
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                words = row["words"]
                text = " ".join(words)
                if not text.strip():
                    continue
                yield make_record(
                    text=text,
                    nguon="vihos",
                    nguon_id=f"{split}:{index}",
                    nhan_goc=str(row["label"]).upper(),
                    split_goc=split,
                    bang_chung_goi_y=mask_to_phrases(words, row["span_mask"]),
                )


# ---------------------------------------------------------------- ViTHSD

VITHSD_FILES = {"train": "train.xlsx.csv", "validation": "dev.xlsx.csv", "test": "test.xlsx.csv"}
TEXT_COLUMNS = ("content", "comment", "text", "free_text", "sentence")
# Mức độ theo thứ tự tăng dần; dùng để lấy mức nặng nhất trên mọi đối tượng (individual, groups, ...).
# Mã số của ViTHSD (sonlam1102/vithsd): 0 = không nhắm tới đối tượng này, 1 = clean, 2 = offensive, 3 = hate.
# Không nhắm tới đối tượng nào cũng tính là CLEAN.
SEVERITY = {"clean": 0, "": 0, "offensive": 1, "hate": 2, "0": 0, "1": 0, "2": 1, "3": 2}
SEVERITY_NAME = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def _vithsd_severity(value: str) -> int | None:
    """Nhận 'clean'/'offensive'/'hate', số 0/1/2, hoặc dạng 'individual#hate'. None nếu không hiểu."""
    value = (value or "").strip().lower()
    if "#" in value:
        value = value.rsplit("#", 1)[1]
    return SEVERITY.get(value)


def vithsd_row_to_record(row: dict, split: str, index: int) -> dict | None:
    """Đổi một dòng CSV ViTHSD về bản ghi chung.

    ViTHSD gán nhãn theo từng đối tượng bị nhắm tới. nhan_goc là mức nặng nhất trên mọi cột
    nhãn (CLEAN/OFFENSIVE/HATE); chi tiết từng cột giữ ở nhan_goc_chi_tiet. Cột nhãn dạng gộp
    'individual#hate|groups#offensive' cũng được hỗ trợ.
    """
    lowered = {str(k).strip().lower(): ("" if v is None else str(v)) for k, v in row.items() if k}
    text_col = next((c for c in TEXT_COLUMNS if c in lowered), None)
    if text_col is None:
        raise ValueError(f"Không tìm thấy cột văn bản trong ViTHSD; các cột: {list(lowered)}")
    text = lowered[text_col]
    if not text.strip():
        return None
    detail: dict[str, str] = {}
    levels: list[int] = []
    unknown: list[str] = []
    for column, value in lowered.items():
        if column == text_col or column in {"id", "index", "unnamed: 0", ""}:
            continue
        parts = [p for p in value.replace(";", "|").replace(",", "|").split("|") if p.strip()] or [value]
        for part in parts:
            level = _vithsd_severity(part)
            if level is None:
                unknown.append(f"{column}={part}")
            else:
                levels.append(level)
        detail[column] = value
    # Có giá trị lạ thì không tự suy ra CLEAN: để KHONG_RO cho người xem.
    nhan_goc = SEVERITY_NAME[max(levels)] if levels and not unknown else "KHONG_RO"
    row_id = lowered.get("id") or lowered.get("index") or lowered.get("unnamed: 0") or str(index)
    return make_record(
        text=text,
        nguon="vithsd",
        nguon_id=f"{split}:{row_id}",
        nhan_goc=nhan_goc,
        split_goc=split,
        nhan_goc_chi_tiet=detail,
    )


def iter_vithsd(directory: Path = VITHSD_DIR) -> Iterator[dict]:
    """Đọc ViTHSD nếu đã tải. Thiếu file thì không sinh gì (người gọi tự cảnh báo)."""
    for split, name in VITHSD_FILES.items():
        path = directory / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                record = vithsd_row_to_record(row, split, index)
                if record is not None:
                    yield record


def vithsd_available(directory: Path = VITHSD_DIR) -> bool:
    return any((directory / name).exists() for name in VITHSD_FILES.values())


# ---------------------------------------------------------------- gộp và thống kê

def compute_stats(records: Iterable[dict]) -> dict:
    by_source: Counter = Counter()
    by_source_label: Counter = Counter()
    by_source_split: Counter = Counter()
    with_hint: Counter = Counter()
    texts: Counter = Counter()
    for record in records:
        source = record["nguon"]
        by_source[source] += 1
        by_source_label[(source, record["nhan_goc"])] += 1
        by_source_split[(source, record["split_goc"])] += 1
        if record["bang_chung_goi_y"]:
            with_hint[source] += 1
        texts[record["text"]] += 1
    nested_label: dict[str, dict[str, int]] = {}
    for (source, label), count in sorted(by_source_label.items()):
        nested_label.setdefault(source, {})[label] = count
    nested_split: dict[str, dict[str, int]] = {}
    for (source, split), count in sorted(by_source_split.items()):
        nested_split.setdefault(source, {})[split] = count
    return {
        "tong": sum(by_source.values()),
        "theo_nguon": dict(sorted(by_source.items())),
        "theo_nguon_nhan_goc": nested_label,
        "theo_nguon_split_goc": nested_split,
        "co_bang_chung_goi_y": dict(sorted(with_hint.items())),
        "so_text_khac_nhau": len(texts),
        "so_ban_ghi_trung_text_chinh_xac": sum(c - 1 for c in texts.values() if c > 1),
    }


def merge(
    vihsd_dir: Path = VIHSD_DIR,
    vihos_dir: Path = VIHOS_DIR,
    vithsd_dir: Path = VITHSD_DIR,
    vihsd_raw_dir: Path = VIHSD_RAW_DIR,
) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    if vihsd_raw_available(vihsd_raw_dir):
        records = list(iter_vihsd_raw(vihsd_raw_dir))
    else:
        records = list(iter_vihsd(vihsd_dir))
        warnings.append(
            f"Chưa có ViHSD gốc trong {vihsd_raw_dir}: chỉ lấy split train từ bộ v1_2. "
            "Chạy `python -m src.v2.data.download` để có đủ train/dev/test."
        )
    records += list(iter_vihos(vihos_dir))
    if vithsd_available(vithsd_dir):
        records += list(iter_vithsd(vithsd_dir))
    else:
        warnings.append(
            f"Chưa có ViTHSD trong {vithsd_dir} (bộ trên Hugging Face bị gated: chấp nhận điều khoản, "
            "đặt HF_TOKEN rồi chạy `python -m src.v2.data.download`). Bỏ qua ViTHSD."
        )
    ids = Counter(r["id"] for r in records)
    clashes = [i for i, c in ids.items() if c > 1]
    if clashes:
        raise RuntimeError(f"Trùng id: {clashes[:5]}")
    return records, warnings


NHAN_GOC_MUC = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}


def collapse_by_text(records: list[dict]) -> list[dict]:
    """Mỗi text khác nhau (sau NFC + gộp khoảng trắng) giữ đúng một bản ghi.

    ViHOS và ViTHSD gần như toàn là câu của ViHSD, nên gộp theo text. Bản ghi giữ id và nguon của lần xuất
    hiện đầu tiên (thứ tự ViHSD, ViHOS, ViTHSD, nên id cũ của ViHSD không đổi). Mọi lần xuất hiện được lưu ở
    `nguon_goc` (nguon, nguon_id, nhan_goc, split_goc). `nhan_goc` là mức nặng nhất trong các nguồn,
    `nhan_goc_bat_dong` = True khi có nguồn nói CLEAN và nguồn khác nói OFFENSIVE/HATE.
    `bang_chung_goi_y` là hợp các cụm gợi ý (từ ViHOS).
    """
    by_text: dict[str, dict] = {}
    for record in records:
        origin = {k: record[k] for k in ("nguon", "nguon_id", "nhan_goc", "split_goc")}
        if "nhan_goc_chi_tiet" in record:
            origin["nhan_goc_chi_tiet"] = record["nhan_goc_chi_tiet"]
        kept = by_text.get(record["text"])
        if kept is None:
            kept = {k: v for k, v in record.items() if k != "nhan_goc_chi_tiet"}
            kept["bang_chung_goi_y"] = list(record["bang_chung_goi_y"])
            kept["nguon_goc"] = []
            by_text[record["text"]] = kept
        kept["nguon_goc"].append(origin)
        for phrase in record["bang_chung_goi_y"]:
            if phrase not in kept["bang_chung_goi_y"]:
                kept["bang_chung_goi_y"].append(phrase)
    for kept in by_text.values():
        levels = {o["nhan_goc"] for o in kept["nguon_goc"]}
        known = [lv for lv in levels if lv in NHAN_GOC_MUC]
        kept["nhan_goc"] = max(known, key=NHAN_GOC_MUC.__getitem__) if known else "KHONG_RO"
        kept["nhan_goc_bat_dong"] = "CLEAN" in levels and bool(levels & {"OFFENSIVE", "HATE"})
    return list(by_text.values())


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vihsd-dir", type=Path, default=VIHSD_DIR)
    parser.add_argument("--vihos-dir", type=Path, default=VIHOS_DIR)
    parser.add_argument("--vithsd-dir", type=Path, default=VITHSD_DIR)
    parser.add_argument("--vihsd-raw-dir", type=Path, default=VIHSD_RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    all_records, warnings = merge(args.vihsd_dir, args.vihos_dir, args.vithsd_dir, args.vihsd_raw_dir)
    records = collapse_by_text(all_records)
    write_jsonl(records, args.out)
    stats = {"truoc_khi_gop_text": compute_stats(all_records), **compute_stats(records)}
    stats["so_text_nhan_goc_bat_dong"] = sum(r["nhan_goc_bat_dong"] for r in records)
    stats["so_text_xuat_hien_o_nhieu_nguon"] = sum(len({o["nguon"] for o in r["nguon_goc"]}) > 1 for r in records)
    stats["canh_bao"] = warnings
    stats_path = args.out.with_name(args.out.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for warning in warnings:
        print(f"CẢNH BÁO: {warning}", file=sys.stderr)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Đã ghi {len(records)} bản ghi vào {args.out} và thống kê vào {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
