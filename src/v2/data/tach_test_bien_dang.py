"""Tách tập test biến dạng gọn (test_perturb_C1/C2, sinh bằng perturb.py --nhom ... --so-ban 2) thành các tập con
để đo riêng: theo giọng vùng (bac/trung/nam) và theo kịch bản đã thấy/chưa thấy (phan_test của câu gốc:
trong_phan_phoi = kịch bản đã có trong train -> da_thay; chua_thay = kịch bản giữ ngoài train như LD7, LD12, DD6, QR6,
moi -> chua_thay). Kiểu biến dạng đã thấy/chưa thấy là chuyện khác, nằm ở trường nhom_bien_dang của từng dòng.

Mỗi dòng biến dạng được thêm giong_vung, phan_test, kich_ban lấy từ câu gốc (ghép theo goc_id với test_day_du.jsonl);
file test_perturb_C1/C2 được ghi lại với các trường này. Nhãn, bằng chứng, text giữ nguyên.

Chạy:
    python -m src.v2.data.tach_test_bien_dang
Ghi vào data/v2/test_perturb/: C1_bac.jsonl, C1_trung.jsonl, C1_nam.jsonl, C1_da_thay.jsonl, C1_chua_thay.jsonl,
tương tự cho C2 và cho C0 (câu gốc chưa biến dạng, lấy thẳng từ test_day_du.jsonl, làm mốc để tính độ tụt
C1 − C0, C2 − C0 trên từng nhóm), và thong_ke.json. Tập con rỗng không ghi file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "v2"
THEM = ("giong_vung", "phan_test", "kich_ban")
GIONG = ("bac", "trung", "nam")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gan_thuoc_tinh(rows: list[dict], goc: dict[str, dict]) -> list[dict]:
    out = []
    for r in rows:
        src = goc.get(r["goc_id"])
        if src is None:
            raise ValueError(f"không thấy câu gốc {r['goc_id']} của {r['id']}")
        if src["nhan"] != r["nhan"]:
            raise ValueError(f"nhãn lệch với câu gốc: {r['id']}")
        out.append({**r, **{k: src.get(k, "") for k in THEM}})
    return out


def tach(rows: list[dict], nhom: str) -> dict[str, list[dict]]:
    parts: dict[str, list[dict]] = {}
    for g in GIONG:
        parts[f"{nhom}_{g}"] = [r for r in rows if r["giong_vung"] == g]
    parts[f"{nhom}_da_thay"] = [r for r in rows if r["phan_test"] == "trong_phan_phoi"]
    parts[f"{nhom}_chua_thay"] = [r for r in rows if r["phan_test"] == "chua_thay"]
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goc", type=Path, default=DATA / "test_day_du.jsonl")
    parser.add_argument("--vao", type=Path, nargs="+",
                        default=[DATA / "test_perturb_C1.jsonl", DATA / "test_perturb_C2.jsonl"])
    parser.add_argument("--ra", type=Path, default=DATA / "test_perturb")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    goc_rows = read_jsonl(args.goc)
    goc = {r["id"]: r for r in goc_rows}
    stats: dict[str, dict] = {}
    c0 = [{**r, "goc_id": r["id"], "bien_dang": "C0", "kieu_bien_dang": "goc", "nhom_bien_dang": "goc"}
          for r in goc_rows]
    bo = [("C0", c0, None)]
    for path in args.vao:
        rows = gan_thuoc_tinh(read_jsonl(path), goc)
        nhoms = {r["bien_dang"] for r in rows}
        if len(nhoms) != 1:
            raise ValueError(f"{path}: cần đúng một nhóm biến dạng, thấy {sorted(nhoms)}")
        bo.append((nhoms.pop(), rows, path))
    for nhom, rows, path in bo:
        if path is not None:
            write_jsonl(path, rows)
        for ten, part in tach(rows, nhom).items():
            stats[ten] = {"so_dong": len(part), "so_cau_goc": len({r["goc_id"] for r in part}),
                          "theo_nhan": dict(sorted(Counter(r["nhan"] for r in part).items()))}
            if part:
                write_jsonl(args.ra / f"{ten}.jsonl", part)
    (args.ra / "thong_ke.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["so_dong"] for k, v in stats.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
