"""Đọc file CSV câu test do người trong nhóm tự viết (docs/v2/HUONG_DAN_VIET_TEST.md), kiểm lỗi, đổi về định dạng
v2 với nguon="nguoi_viet", cach_gan="nguoi", split="test". Nếu có cột nhan_nguoi_kia thì tính Cohen's kappa.

Cột: text, nhan, bang_chung (các cụm cách nhau bằng |), kich_ban, kenh, nguoi_viet, ghi_chu, nhan_nguoi_kia.

text có thể là đoạn hội thoại hai người, tối đa 8 dòng: mỗi dòng là toàn bộ lượt nói của một người (gộp các tin
liên tiếp), các dòng luân phiên hai người, không ghi người nói. Trong ô CSV xuống dòng trực tiếp, hoặc ngăn các
lượt bằng " || " (được đổi thành xuống dòng khi nạp). Mỗi cụm bằng chứng nằm gọn trong một dòng.

Chạy:
    python -m src.v2.data.load_test_nguoi_viet data/v2/raw/test_A.csv data/v2/raw/test_B.csv
Dòng lỗi được in ra kèm số dòng; chỉ ghi file đầu ra khi không còn lỗi (dùng --bo-qua-loi để ghi phần hợp lệ).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.v2.data.merge import make_record, normalize_lines, normalize_text

ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "data" / "v2" / "interim" / "test_nguoi_viet.jsonl"
LABELS = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
KENH = {"messenger_zalo", "nhom_lop", "chat_game", "binh_luan", "sms"}
REQUIRED = ("text", "nhan", "nguoi_viet")
MAX_LINES = 8
LINE_SEPARATOR = re.compile(r"\s*\|\|\s*")


def split_evidence(value: str) -> list[str]:
    return [normalize_text(p) for p in (value or "").split("|") if p.strip()]


def row_to_record(row: dict, source_file: str, line_no: int) -> tuple[dict | None, list[str]]:
    """Trả về (bản ghi, danh sách lỗi). Có lỗi thì bản ghi vẫn trả về để báo, nhưng không được ghi ra."""
    r = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
    errors: list[str] = []
    text = normalize_lines(LINE_SEPARATOR.sub("\n", r.get("text", "")))
    if not text:
        return None, []
    lines = text.split("\n")
    so_dong = len(lines)
    if so_dong > MAX_LINES:
        errors.append(f"qua {MAX_LINES} dong: {so_dong}")
    nhan = r.get("nhan", "").upper()
    if nhan not in LABELS:
        errors.append(f"nhan khong hop le: {r.get('nhan')!r}")
    evidence = split_evidence(r.get("bang_chung", ""))
    missing = [p for p in evidence if not any(p in line for line in lines)]  # mỗi cụm nằm gọn trong 1 dòng
    if missing:
        errors.append(f"bang_chung khong co nguyen van trong text: {missing}")
    if nhan == "AN_TOAN" and evidence:
        errors.append("cau AN_TOAN phai de trong bang_chung")
    kenh = r.get("kenh", "")
    if kenh and kenh not in KENH:
        errors.append(f"kenh la: {kenh!r}")
    nguoi_viet = r.get("nguoi_viet", "").upper()
    if not nguoi_viet:
        errors.append("thieu nguoi_viet")
    other = r.get("nhan_nguoi_kia", "").upper()
    if other and other not in LABELS:
        errors.append(f"nhan_nguoi_kia khong hop le: {other!r}")

    digest = hashlib.sha1(f"{nguoi_viet}|{text}".encode("utf-8")).hexdigest()[:12]
    record = make_record(text=text, nguon="nguoi_viet", nguon_id=digest, nhan_goc="", split_goc="",
                         nhieu_dong=True)
    record.pop("bang_chung_goi_y", None)
    record.update({
        "nhan": nhan, "bang_chung": evidence, "cach_gan": "nguoi", "split": "test",
        "kich_ban": r.get("kich_ban", ""), "kenh": kenh, "nguoi_viet": nguoi_viet,
        "nhan_nguoi_kia": other, "ghi_chu": r.get("ghi_chu", ""),
        "file_nguon": source_file, "dong": line_no,
    })
    return record, errors


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Kappa giữa hai người gán trên các cặp (nhãn người viết, nhãn người kia)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(left[k] * right[k] for k in set(left) | set(right)) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def load(paths: list[Path]) -> tuple[list[dict], list[str], dict]:
    records: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = [f.strip().lower() for f in reader.fieldnames or []]
            lacking = [c for c in REQUIRED if c not in fields]
            if lacking:
                raise ValueError(f"{path}: thiếu cột {lacking}")
            for index, row in enumerate(reader):
                line_no = index + 2
                record, row_errors = row_to_record(row, path.name, line_no)
                if record is None:
                    continue
                if row_errors:
                    errors.extend(f"{path.name}:{line_no}: {e}" for e in row_errors)
                    continue
                key = record["text"].lower()
                if key in seen:
                    errors.append(f"{path.name}:{line_no}: trung nguyen van voi cau truoc")
                    continue
                seen.add(key)
                records.append(record)

    pairs = [(r["nhan"], r["nhan_nguoi_kia"]) for r in records if r["nhan_nguoi_kia"]]
    disagreements = [f"{r['file_nguon']}:{r['dong']} {r['nhan']} vs {r['nhan_nguoi_kia']}: {r['text'][:80]}"
                     for r in records if r["nhan_nguoi_kia"] and r["nhan_nguoi_kia"] != r["nhan"]]
    kappa = cohen_kappa(pairs)
    stats = {
        "so_cau": len(records),
        "theo_nhan": dict(Counter(r["nhan"] for r in records).most_common()),
        "theo_nguoi_viet": dict(Counter(r["nguoi_viet"] for r in records).most_common()),
        "theo_kich_ban": dict(sorted(Counter(r["kich_ban"] or "(trong)" for r in records).items())),
        "so_cau_co_nhan_nguoi_kia": len(pairs),
        "cohen_kappa": None if kappa is None else round(kappa, 4),
        "cau_bat_dong": disagreements,
        "so_loi": len(errors),
    }
    return records, errors, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--bo-qua-loi", action="store_true", help="vẫn ghi các dòng hợp lệ khi có dòng lỗi")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    records, errors, stats = load(args.inputs)
    for error in errors:
        print(f"LỖI {error}", file=sys.stderr)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if errors and not args.bo_qua_loi:
        print(f"Có {len(errors)} lỗi, chưa ghi file. Sửa CSV rồi chạy lại.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.out.with_name(args.out.stem + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(records)} câu vào {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
