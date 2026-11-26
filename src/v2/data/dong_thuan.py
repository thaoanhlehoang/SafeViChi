"""Lọc câu thật đã được LLM gán lại theo quy tắc đồng thuận (chốt 20/11), và xuất mẫu cho người kiểm.

Quy tắc: chỉ giữ câu mà nhãn LLM (quy về sạch/nguy hại) khớp với ít nhất một người gán gốc. Mỗi câu thật có nhãn
của 1–3 bộ (ViHSD, ViHOS, ViTHSD), tức 1–3 nhóm người gán khác nhau, lưu ở `nguon_goc`.
- `khop_nhan_goc`: LLM khớp nhãn gốc nặng nhất            -> giữ
- `khop_it_nhat_mot_nguoi_gan`: người gán gốc bất đồng với nhau, LLM đứng về một bên -> giữ
- `lech_moi_nguoi_gan`: mọi người gán gốc cùng một ý, LLM nói ngược lại -> bỏ khỏi train
- `loi_llm`: LLM không trả được nhãn hợp lệ                  -> bỏ
Không gán lại bằng tay; câu bị bỏ chỉ không vào train.

Mẫu kiểm tay (để đo tỉ lệ nhãn sai và Cohen's kappa, xem HUONG_DAN_GAN_NHAN.md mục 6): 100 câu giữ + 100 câu bỏ,
chọn ngẫu nhiên seed 42, trộn thứ tự, ẨN mọi nhãn. Hai người gán độc lập cùng 200 câu, mỗi người một file CSV.
Bảng khóa (nhóm giữ/bỏ, nhãn LLM, nhãn gốc) nằm riêng, người gán không mở.

Đầu ra:
    data/v2/interim/that_gan_lai.jsonl           mọi câu đã gán lại + dong_thuan, ly_do_dong_thuan
    data/v2/interim/that_gan_lai_stats.json      số câu giữ/bỏ theo nhãn
    data/v2/raw/kiem_nhan_A.csv, kiem_nhan_B.csv 200 câu cho người A và B gán (cột nhan để trống)
    data/v2/interim/kiem_nhan_khoa.jsonl         bảng khóa, KHÔNG mở khi đang gán

Chạy:
    python -m src.v2.data.dong_thuan
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "v2" / "interim"
RAW = ROOT / "data" / "v2" / "raw"
PRELABEL_DIR = INTERIM / "prelabel"
OUT_PATH = INTERIM / "that_gan_lai.jsonl"
KHOA_PATH = INTERIM / "kiem_nhan_khoa.jsonl"
SEED = 42
NHAN_HOP_LE = {"AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO"}
CSV_COLUMNS = ["stt", "id", "text", "nhan", "bang_chung", "ghi_chu"]


def nhi_phan(nhan: str) -> str | None:
    """Quy về sạch/nguy hại cho cả nhãn gốc (CLEAN/OFFENSIVE/HATE) và nhãn v2."""
    if nhan in ("CLEAN", "AN_TOAN"):
        return "SACH"
    if nhan in ("OFFENSIVE", "HATE", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO"):
        return "NGUY_HAI"
    return None


def xet_dong_thuan(record: dict) -> tuple[bool, str]:
    llm = nhi_phan(record.get("nhan") or "")
    if record.get("nhan") not in NHAN_HOP_LE or llm is None:
        return False, "loi_llm"
    if llm == nhi_phan(record["nhan_goc"]):
        return True, "khop_nhan_goc"
    nguoi_gan = {nhi_phan(o["nhan_goc"]) for o in record.get("nguon_goc", [])} - {None}
    if llm in nguoi_gan:
        return True, "khop_it_nhat_mot_nguoi_gan"
    return False, "lech_moi_nguoi_gan"


def gan_co(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        keep, reason = xet_dong_thuan(record)
        out.append({**record, "dong_thuan": keep, "ly_do_dong_thuan": reason})
    return out


def chon_mau(records: list[dict], n_moi_nhom: int = 100, seed: int = SEED) -> list[dict]:
    """n câu giữ + n câu bỏ (bỏ vì lệch, không lấy câu lỗi LLM), trộn thứ tự, đánh số thứ tự."""
    rng = random.Random(seed)
    giu = [r for r in records if r["dong_thuan"]]
    bo = [r for r in records if r["ly_do_dong_thuan"] == "lech_moi_nguoi_gan"]
    sample = ([{**r, "nhom_kiem": "giu"} for r in rng.sample(giu, min(n_moi_nhom, len(giu)))]
              + [{**r, "nhom_kiem": "bo"} for r in rng.sample(bo, min(n_moi_nhom, len(bo)))])
    rng.shuffle(sample)
    return [{**r, "stt": i} for i, r in enumerate(sample, 1)]


def write_csv(sample: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig để Excel mở đúng tiếng Việt
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in sample:
            writer.writerow({"stt": r["stt"], "id": r["id"], "text": r["text"].replace("\n", " || "),
                             "nhan": "", "bang_chung": "", "ghi_chu": ""})


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_prelabel(directory: Path = PRELABEL_DIR) -> Path:
    files = [p for p in directory.glob("prelabel_*.jsonl") if not p.name.endswith("_stats.jsonl")]
    if not files:
        raise SystemExit(f"Chưa có kết quả gán lại trong {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="mặc định: file prelabel_*.jsonl mới nhất")
    parser.add_argument("--so-cau-moi-nhom", type=int, default=100)
    parser.add_argument("--ghi-de-mau", action="store_true",
                        help="ghi đè file CSV kiểm tay đã có (mặc định không ghi đè để khỏi mất phần đã gán)")
    args = parser.parse_args(argv)

    source = args.input or latest_prelabel()
    records = gan_co([json.loads(line) for line in source.open(encoding="utf-8") if line.strip()])
    write_jsonl(records, OUT_PATH)

    by_label: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_label[r["nhan"] or "(loi)"]["giu" if r["dong_thuan"] else "bo"] += 1
    stats = {
        "dau_vao": source.name, "tong": len(records),
        "giu": sum(r["dong_thuan"] for r in records),
        "theo_ly_do": dict(Counter(r["ly_do_dong_thuan"] for r in records).most_common()),
        "theo_nhan_llm": {k: dict(v) for k, v in sorted(by_label.items())},
        "chuyen_nhan_bi_bo": dict(Counter(f"{r['nhan_goc']}->{r['nhan']}" for r in records
                                          if r["ly_do_dong_thuan"] == "lech_moi_nguoi_gan").most_common()),
    }
    OUT_PATH.with_name("that_gan_lai_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_a, csv_b = RAW / "kiem_nhan_A.csv", RAW / "kiem_nhan_B.csv"
    if (csv_a.exists() or csv_b.exists()) and not args.ghi_de_mau:
        print(f"Đã có {csv_a.name}/{csv_b.name}: không ghi đè (thêm --ghi-de-mau nếu thật sự muốn).")
    else:
        sample = chon_mau(records, args.so_cau_moi_nhom)
        write_csv(sample, csv_a)
        write_csv(sample, csv_b)
        write_jsonl([{"stt": r["stt"], "id": r["id"], "nhom_kiem": r["nhom_kiem"], "nhan_llm": r["nhan"],
                      "nhan_goc": r["nhan_goc"], "nguon_goc": r["nguon_goc"], "text": r["text"]} for r in sample],
                    KHOA_PATH)
        print(f"Đã ghi mẫu kiểm tay: {csv_a}, {csv_b} ({len(sample)} câu), bảng khóa {KHOA_PATH}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
