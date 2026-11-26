"""Gom token, tiền và thời gian của các lượt chạy LLM tạo dữ liệu v2 thành file kết quả cho báo cáo.

Mọi số trong báo cáo phải sinh từ file (CLAUDE.md), nên script này đọc:
- log của từng bước (dòng "Token đã dùng lượt này: N lượt gọi, X token vào, Y token ra (trong đó Z token suy luận)"
  mà prelabel.py, generate.py, loc_sinh.py in ra khi xong);
- file thống kê của từng bước (số câu, phân bố nhãn, tỉ lệ giữ sau lọc).

Tiền tính theo giá endpoint đã khóa (wafer/us) lấy từ OpenRouter ngày 20/11/2026. Số tiền thật bị trừ xem ở trang
Activity của OpenRouter; chênh lệch nhỏ do làm tròn. Tỷ giá VND là giả định, ghi rõ trong file đầu ra.

Đầu ra: results/v2/chi_phi_du_lieu/chi_phi.json (+ bản sao log trong logs/).

Chạy:
    python -m src.v2.labeling.chi_phi
    python -m src.v2.labeling.chi_phi --log-dir logs_v2 --ty-gia 27000
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "logs_v2"
INTERIM = ROOT / "data" / "v2" / "interim"
OUT_DIR = ROOT / "results" / "v2" / "chi_phi_du_lieu"

# Giá USD cho 1 token, endpoint wafer/us của deepseek/deepseek-v4.1-flash-1017 (OpenRouter, 20/11/2026).
GIA = {"model": "deepseek/deepseek-v4.1-flash-1017", "endpoint": "wafer/us",
       "usd_moi_token_vao": 0.036e-6, "usd_moi_token_ra": 0.50e-6, "nguon_gia": "OpenRouter endpoints API, 20/11/2026"}

# (tên bước, mẫu tên log, file thống kê)
BUOC = [
    ("gan_lai_nhan", "1_prelabel*.log", "prelabel/prelabel_*_stats.json"),
    ("sinh_doan", "2_generate*.log", "sinh_llm_stats.json"),
    ("loc_lai_doan_sinh", "3_loc_sinh*.log", "sinh_llm_loc_stats.json"),
]

USAGE_RE = re.compile(
    r"Token đã dùng lượt này: ([\d,]+) lượt gọi, ([\d,]+) token vào, ([\d,]+) token ra "
    r"\(trong đó ([\d,]+) token suy luận\)")


def parse_usage(text: str) -> dict | None:
    """Cộng mọi dòng tổng token trong một log (log có thể gồm nhiều lần chạy nối tiếp)."""
    rows = USAGE_RE.findall(text)
    if not rows:
        return None
    total = [sum(int(r[i].replace(",", "")) for r in rows) for i in range(4)]
    return {"luot_goi": total[0], "token_vao": total[1], "token_ra": total[2], "token_suy_luan": total[3],
            "so_lan_chay": len(rows)}


def cost(usage: dict, ty_gia: float) -> dict:
    usd = usage["token_vao"] * GIA["usd_moi_token_vao"] + usage["token_ra"] * GIA["usd_moi_token_ra"]
    return {"usd": round(usd, 4), "vnd": round(usd * ty_gia)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--ty-gia", type=float, default=27000, help="VND/USD (giả định, ghi vào kết quả)")
    parser.add_argument("--bat-dau", default=None,
                        help="thời điểm bắt đầu bước đầu tiên, dạng 'YYYY-MM-DD HH:MM' (để tính thời gian chạy)")
    args = parser.parse_args(argv)

    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    steps = []
    prev_end = datetime.strptime(args.bat_dau, "%Y-%m-%d %H:%M") if args.bat_dau else None
    tong = {"luot_goi": 0, "token_vao": 0, "token_ra": 0, "token_suy_luan": 0}
    for name, log_glob, stats_glob in BUOC:
        logs = sorted(args.log_dir.glob(log_glob))
        text = "".join(path.read_text(encoding="utf-8", errors="replace") for path in logs)
        usage = parse_usage(text)
        step: dict = {"buoc": name, "log": [p.name for p in logs]}
        if usage is None:
            step["trang_thai"] = "chua_co_dong_tong_token (chưa chạy xong hoặc lỗi)"
            steps.append(step)
            continue
        for path in logs:
            shutil.copy2(path, args.out_dir / "logs" / path.name)
        end = datetime.fromtimestamp(max(p.stat().st_mtime for p in logs))
        step.update(usage)
        step.update(cost(usage, args.ty_gia))
        step["ket_thuc"] = end.strftime("%Y-%m-%d %H:%M:%S")
        if prev_end is not None:
            step["thoi_gian_phut"] = round((end - prev_end).total_seconds() / 60, 1)
        prev_end = end
        stats_files = sorted(INTERIM.glob(stats_glob))
        if stats_files:
            stats = json.loads(stats_files[-1].read_text(encoding="utf-8"))
            step["thong_ke"] = {k: v for k, v in stats.items()
                                if k in ("tong", "giu", "loai", "theo_nhan", "ly_do_loai", "so_doan",
                                         "can_duyet", "o_bi_tu_choi", "o_loi_api", "doan_bi_bo")}
        for key in tong:
            tong[key] += usage[key]
        steps.append(step)

    result = {
        "gia": GIA, "ty_gia_vnd_usd_gia_dinh": args.ty_gia,
        "cac_buoc": steps,
        "tong": {**tong, **cost(tong, args.ty_gia)},
        "ghi_chu": "Tiền tính từ số token × giá endpoint; số bị trừ thật xem trang Activity của OpenRouter.",
        "tao_luc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = args.out_dir / "chi_phi.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
