"""Đọc file CSV câu test do người trong nhóm tự viết (docs/v2/HUONG_DAN_VIET_TEST.md), kiểm lỗi, đổi về định dạng
v2 với nguon="nguoi_viet", cach_gan="nguoi", split="test". Nếu có cột nhan_nguoi_kia thì tính Cohen's kappa.

Cột: text, nhan, bang_chung (các cụm cách nhau bằng |), kich_ban, kenh, giong_vung (bac|trung|nam|chung), nguoi_viet, ghi_chu, nhan_nguoi_kia.
Cột tùy chọn: nguon (nguoi_viet | claude_viet: câu do Claude viết, người đã duyệt), cach_gan (nguoi | llm_nguoi_duyet),
do_lo, và các cột đối chiếu nhãn nhan_nguoi, nhan_llm, nhan_chon_theo, bang_chung_lay_tu (chỉ chép sang bản đầy đủ).

text có thể là đoạn hội thoại hai người, tối đa 8 dòng: mỗi dòng là toàn bộ lượt nói của một người (gộp các tin
liên tiếp), các dòng luân phiên hai người, không ghi người nói. Trong ô CSV xuống dòng trực tiếp, hoặc ngăn các
lượt bằng " || " (được đổi thành xuống dòng khi nạp). Mỗi cụm bằng chứng nằm gọn trong một dòng.

Chạy:
    python -m src.v2.data.load_test_nguoi_viet data/v2/raw/test_A.csv data/v2/raw/test_B.csv
Dòng lỗi được in ra kèm số dòng; chỉ ghi file đầu ra khi không còn lỗi (dùng --bo-qua-loi để ghi phần hợp lệ).

Tạo tập test cho mô hình (bàn giao):
    python -m src.v2.data.load_test_nguoi_viet data/v2/raw/test_A.csv data/v2/raw/test_B.csv data/v2/raw/test_C.csv \
        --xuat-test
ghi `data/v2/test.jsonl` (chỉ id, text, nhan, bang_chung như train), `data/v2/test_day_du.jsonl` (đủ trường, có
giong_vung, phan_test để tính điểm theo vùng miền và kịch bản chưa thấy) và `data/v2/test_stats.json`.
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
GIONG_VUNG = {"bac", "trung", "nam", "chung"}
REQUIRED = ("text", "nhan", "nguoi_viet")
NGUON = {"nguoi_viet", "claude_viet"}
CACH_GAN = {"nguoi", "llm_nguoi_duyet"}
COT_DOI_CHIEU = ("stt", "nhan_nguoi", "nhan_llm", "nhan_chon_theo", "bang_chung_lay_tu")
MODEL_FIELDS = ("id", "text", "nhan", "bang_chung")
TEST_DIR = ROOT / "data" / "v2"
MAX_LINES = 8
LINE_SEPARATOR = re.compile(r"\s*\|\|\s*")
KICH_BAN_PATH = ROOT / "src" / "v2" / "labeling" / "kich_ban.yaml"


def held_out_codes(path: Path = KICH_BAN_PATH) -> set[str]:
    """Mã kịch bản giữ ngoài train (giu_ngoai_train: true). Câu test thuộc các mã này hoặc `moi` là phần
    "kịch bản chưa thấy"; còn lại là phần "trong phân phối"."""
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {row["ma"] for row in data.get("kich_ban", []) if row.get("giu_ngoai_train")}


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
    giong_vung = r.get("giong_vung", "").lower()
    if not giong_vung:
        errors.append("thieu giong_vung (bac|trung|nam|chung)")
    elif giong_vung not in GIONG_VUNG:
        errors.append(f"giong_vung la: {giong_vung!r}")
    nguoi_viet = r.get("nguoi_viet", "").upper()
    if not nguoi_viet:
        errors.append("thieu nguoi_viet")
    nguon = r.get("nguon", "") or "nguoi_viet"
    if nguon not in NGUON:
        errors.append(f"nguon la: {nguon!r}")
    cach_gan = r.get("cach_gan", "") or "nguoi"
    if cach_gan not in CACH_GAN:
        errors.append(f"cach_gan la: {cach_gan!r}")
    other = r.get("nhan_nguoi_kia", "").upper()
    if other and other not in LABELS:
        errors.append(f"nhan_nguoi_kia khong hop le: {other!r}")

    digest = hashlib.sha1(f"{nguoi_viet}|{text}".encode("utf-8")).hexdigest()[:12]
    record = make_record(text=text, nguon=nguon, nguon_id=digest, nhan_goc="", split_goc="",
                         nhieu_dong=True)
    record.pop("bang_chung_goi_y", None)
    record.update({
        "nhan": nhan, "bang_chung": evidence, "cach_gan": cach_gan, "split": "test",
        "kich_ban": r.get("kich_ban", ""), "kenh": kenh, "do_lo": r.get("do_lo", ""), "giong_vung": giong_vung, "nguoi_viet": nguoi_viet,
        "nhan_nguoi_kia": other, "ghi_chu": r.get("ghi_chu", ""),
        "file_nguon": source_file, "dong": line_no,
    })
    record.update({k: r[k] for k in COT_DOI_CHIEU if r.get(k)})
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
    held = held_out_codes() | {"moi"}
    records: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = handle.readline()
            handle.seek(0)
            # Excel bản tiếng Việt lưu CSV bằng dấu ";"
            delimiter = ";" if first.count(";") > first.count(",") else ","
            reader = csv.DictReader(handle, delimiter=delimiter)
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
                record["phan_test"] = "chua_thay" if record["kich_ban"] in held else "trong_phan_phoi"
                records.append(record)

    pairs = [(r["nhan"], r["nhan_nguoi_kia"]) for r in records if r["nhan_nguoi_kia"]]
    disagreements = [f"{r['file_nguon']}:{r['dong']} {r['nhan']} vs {r['nhan_nguoi_kia']}: {r['text'][:80]}"
                     for r in records if r["nhan_nguoi_kia"] and r["nhan_nguoi_kia"] != r["nhan"]]
    kappa = cohen_kappa(pairs)
    stats = {
        "so_cau": len(records),
        "theo_nhan": dict(Counter(r["nhan"] for r in records).most_common()),
        "theo_nguoi_viet": dict(Counter(r["nguoi_viet"] for r in records).most_common()),
        "theo_nguon_va_cach_gan": dict(sorted(Counter(f"{r['nguon']}/{r['cach_gan']}" for r in records).items())),
        "theo_nhan_chon_theo": dict(sorted(Counter(r.get("nhan_chon_theo", "(trong)") for r in records).items())),
        "theo_kich_ban": dict(sorted(Counter(r["kich_ban"] or "(trong)" for r in records).items())),
        "theo_giong_vung_va_nhan": dict(sorted(Counter(f"{r['giong_vung']}/{r['nhan']}" for r in records).items())),
        "theo_phan_va_nhan": dict(sorted(Counter(f"{r['phan_test']}/{r['nhan']}" for r in records).items())),
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
    parser.add_argument("--xuat-test", action="store_true",
                        help="ghi data/v2/test.jsonl, test_day_du.jsonl, test_stats.json thay cho --out")
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
    if args.xuat_test:
        write_jsonl(TEST_DIR / "test.jsonl", [{k: r[k] for k in MODEL_FIELDS} for r in records])
        write_jsonl(TEST_DIR / "test_day_du.jsonl", records)
        (TEST_DIR / "test_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
        print(f"Đã ghi {len(records)} câu vào {TEST_DIR / 'test.jsonl'} và test_day_du.jsonl")
        return 0
    write_jsonl(args.out, records)
    args.out.with_name(args.out.stem + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(records)} câu vào {args.out}")
    return 0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
