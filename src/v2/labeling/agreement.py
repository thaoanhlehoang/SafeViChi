"""Đo đồng thuận sau khi hai người gán tay mẫu kiểm (tạo bằng python -m src.v2.data.dong_thuan).

Đọc kiem_nhan_A.csv, kiem_nhan_B.csv (cột nhan đã điền) và bảng khóa kiem_nhan_khoa.jsonl, tính:
- Cohen's kappa giữa A và B (5 nhãn và sạch/nguy hại). Mục tiêu từ 0,6 (HUONG_DAN_GAN_NHAN.md mục 6).
- Với câu A và B cùng nhãn ("nhãn người"):
  * nhóm GIỮ: tỉ lệ nhãn LLM đúng = ước tỉ lệ nhãn sai của câu thật trong train (kèm khoảng tin cậy Wilson 95%);
  * nhóm BỎ: tỉ lệ người đứng về phía LLM hay phía nhãn gốc (cho biết quy tắc bỏ có làm mất câu đúng không).
- Danh sách câu A và B bất đồng để ngồi thảo luận, sửa hướng dẫn.

Đầu ra: results/v2/dong_thuan_nhan/metrics.json và cau_bat_dong.csv.

Chạy:
    python -m src.v2.labeling.agreement
    python -m src.v2.labeling.agreement --a data/v2/raw/kiem_nhan_A.csv --b data/v2/raw/kiem_nhan_B.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from src.v2.data.dong_thuan import KHOA_PATH, NHAN_HOP_LE, RAW, nhi_phan
from src.v2.data.load_test_nguoi_viet import cohen_kappa

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "v2" / "dong_thuan_nhan"


def read_annotator(path: Path) -> str:
    """Giá trị cột nguoi_gan (nếu có): ai đã gán file này. Trống = không ghi."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        values = {(row.get("nguoi_gan") or "").strip() for row in csv.DictReader(handle)} - {""}
    return " / ".join(sorted(values))


def read_labels(path: Path) -> tuple[dict[str, str], list[str]]:
    """{id: nhan} và danh sách lỗi (nhãn không hợp lệ). Dòng để trống nhãn bị bỏ qua (chưa gán)."""
    labels, errors = {}, []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle), 2):
            nhan = (row.get("nhan") or "").strip().upper()
            if not nhan:
                continue
            if nhan not in NHAN_HOP_LE:
                errors.append(f"{path.name}:{i} nhãn lạ {nhan!r}")
                continue
            labels[row["id"]] = nhan
    return labels, errors


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def rate(k: int, n: int) -> dict:
    return {"dung": k, "tong": n, "ti_le": round(k / n, 3) if n else None, "khoang_tin_cay_95": wilson(k, n)}


def analyse(key: list[dict], a: dict[str, str], b: dict[str, str]) -> tuple[dict, list[dict]]:
    both = [k for k in key if k["id"] in a and k["id"] in b]
    pairs = [(a[k["id"]], b[k["id"]]) for k in both]
    kappa5 = cohen_kappa(pairs)
    kappa2 = cohen_kappa([(nhi_phan(x), nhi_phan(y)) for x, y in pairs])
    agreed = [k for k in both if a[k["id"]] == b[k["id"]]]
    giu = [k for k in agreed if k["nhom_kiem"] == "giu"]
    bo = [k for k in agreed if k["nhom_kiem"] == "bo"]
    metrics = {
        "so_cau_ca_hai_da_gan": len(both),
        "kappa_5_nhan": None if kappa5 is None else round(kappa5, 4),
        "kappa_sach_nguy_hai": None if kappa2 is None else round(kappa2, 4),
        "ti_le_A_B_cung_nhan": round(len(agreed) / len(both), 3) if both else None,
        "nhom_giu": {
            "llm_dung_5_nhan": rate(sum(a[k["id"]] == k["nhan_llm"] for k in giu), len(giu)),
            "llm_dung_sach_nguy_hai": rate(sum(nhi_phan(a[k["id"]]) == nhi_phan(k["nhan_llm"]) for k in giu),
                                           len(giu)),
        },
        "nhom_bo": {
            "nguoi_dong_y_llm_sach_nguy_hai": rate(
                sum(nhi_phan(a[k["id"]]) == nhi_phan(k["nhan_llm"]) for k in bo), len(bo)),
            "nguoi_dong_y_nhan_goc_sach_nguy_hai": rate(
                sum(nhi_phan(a[k["id"]]) == nhi_phan(k["nhan_goc"]) for k in bo), len(bo)),
        },
        "nham_lan_A_B": dict(Counter(f"{x}|{y}" for x, y in pairs if x != y).most_common()),
    }
    disagreements = [{"stt": k["stt"], "id": k["id"], "nhan_A": a[k["id"]], "nhan_B": b[k["id"]],
                      "text": k["text"].replace("\n", " || ")} for k in both if a[k["id"]] != b[k["id"]]]
    return metrics, disagreements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, default=RAW / "kiem_nhan_A.csv")
    parser.add_argument("--b", type=Path, default=RAW / "kiem_nhan_B.csv")
    parser.add_argument("--khoa", type=Path, default=KHOA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    a, err_a = read_labels(args.a)
    b, err_b = read_labels(args.b)
    for e in err_a + err_b:
        print(f"LỖI: {e}")
    key = [json.loads(line) for line in args.khoa.open(encoding="utf-8") if line.strip()]
    metrics, disagreements = analyse(key, a, b)
    nguoi_a, nguoi_b = read_annotator(args.a), read_annotator(args.b)
    metrics.update({"so_cau_mau": len(key), "A_da_gan": len(a), "B_da_gan": len(b), "loi_nhan": err_a + err_b,
                    "nguoi_gan_A": nguoi_a or "(không ghi)", "nguoi_gan_B": nguoi_b or "(không ghi)"})
    if nguoi_a and nguoi_a == nguoi_b:
        # Cùng một người (hoặc cùng một mô hình) gán cả hai file: kappa vô nghĩa, không báo cáo.
        metrics["kappa_5_nhan"] = metrics["kappa_sach_nguy_hai"] = None
        metrics["canh_bao"] = (f"A và B cùng do '{nguoi_a}' gán: không tính kappa. Các tỉ lệ nhóm giữ/bỏ là mức khớp "
                               "giữa người gán này và LLM gán sơ bộ, chỉ được báo cáo đúng với nguồn nhãn đó.")
        print("CẢNH BÁO: " + metrics["canh_bao"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    with (args.out_dir / "cau_bat_dong.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stt", "id", "nhan_A", "nhan_B", "text"])
        writer.writeheader()
        writer.writerows(disagreements)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Đã ghi {args.out_dir / 'metrics.json'} và {len(disagreements)} câu bất đồng vào cau_bat_dong.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
