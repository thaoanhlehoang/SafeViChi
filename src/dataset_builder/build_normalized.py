"""Áp `normalize()` (Bước 2) lên cột `text` của bộ v1_2 -> data/processed/normalized/.

Đây là khâu TRUNG GIAN nằm giữa Bước 1 và Bước 3: `build_mixed.py` đọc thư mục
này để lấy 60% "perturbed_normalized" (câu đã bị cấy biến thể rồi cho chạy qua
normalizer, giữ lại phần nhiễu còn sót — chính phần nhiễu sót đó mới là thứ
model cần học chịu đựng).

Ánh xạ 1-1 theo đúng thứ tự dòng của file nguồn, chỉ giữ 2 cột `text` + `label`:

    {"text": normalize(row["text"]), "label": row["label"]}

Chạy:
    python -m src.dataset_builder.build_normalized

⚠️ CẢNH BÁO TÁI LẬP — ĐỌC TRƯỚC KHI CHẠY ĐÈ ⚠️
------------------------------------------------
File `data/processed/normalized/` hiện có trong repo được sinh ra ngày
2026-09-04 bằng một phiên bản normalizer RỘNG TAY HƠN bản đang có trong
`src/normalization/`. Chạy lại script này sẽ ra kết quả LỆCH ~14% số dòng
(339/2376 dòng ở split test), tập trung ở hai chỗ:

  1. Từ điển teencode lúc đó có thêm mục kiểu `vcl` -> `vãi lồn`; bản
     `teencode_dict.json` hiện tại (625 mục) không có.
  2. Bộ separator lúc đó xóa cả dấu chấm giữa `·` (U+00B7); bản hiện tại chỉ
     xóa `. - _`.

Hệ quả, nói thẳng: bộ train đã dùng (`data_final/train.jsonl`) có phần 60%
sạch hơn một chút so với thứ mà normalizer đang ship sẽ tạo ra lúc chạy thật.
Đây là lệch phân bố train/inference nhỏ, và nó lệch theo hướng LÀM XẤU con số
báo cáo chứ không phải thổi phồng (lúc eval dùng đúng normalizer hiện tại, cho
input khó hơn lúc train) — nên kết quả Bước 3 vẫn dùng được, chỉ là hơi bảo thủ.

Không có rò rỉ nhãn: normalize() không hề nhìn thấy `original_text` hay `label`.

Vì vậy script này MẶC ĐỊNH KHÔNG ghi đè. Muốn ghi đè phải truyền `--force`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.normalization.normalizer import normalize
from src.utils.seed import SEED_DEFAULT, set_global_seed

SPLITS = ("train", "validation", "test")


def build_split(source_path: Path, out_path: Path) -> dict:
    """Đọc jsonl nguồn, ghi jsonl {text, label}. Trả thống kê."""
    n = 0
    changed = 0
    labels: dict[str, int] = {}
    with open(source_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            normalized = normalize(row["text"])
            if normalized != row["text"]:
                changed += 1
            labels[row["label"]] = labels.get(row["label"], 0) + 1
            fout.write(json.dumps({"text": normalized, "label": row["label"]},
                                  ensure_ascii=False) + "\n")
            n += 1
    return {"total": n, "changed_by_normalizer": changed, "by_label": labels}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Áp normalize() lên v1_2 -> data/processed/normalized/")
    parser.add_argument("--source_dir", type=str,
                        default="data/processed/vietnamese_nonstandard_v1_2")
    parser.add_argument("--out_dir", type=str, default="data/processed/normalized")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT,
                        help="normalize() hoàn toàn tất định, không dùng ngẫu nhiên. "
                             "Seed ở đây chỉ để ghi nhật ký cho đồng bộ với các script khác.")
    parser.add_argument("--force", action="store_true",
                        help="cho phép ghi đè file đã có. Đọc cảnh báo ở đầu file trước.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    source_dir, out_dir = Path(args.source_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [s for s in args.splits if (out_dir / f"{s}.jsonl").exists()]
    if existing and not args.force:
        print(f"[dừng] Đã có sẵn: {', '.join(existing)} trong {out_dir}/")
        print("       Ghi đè sẽ tạo ra dữ liệu KHÁC với bộ đã dùng để train "
              "(xem cảnh báo ở đầu file). Thêm --force nếu thực sự muốn.")
        return

    manifest = {"source_dir": str(source_dir), "splits": {}}
    for split in args.splits:
        stats = build_split(source_dir / f"{split}.jsonl", out_dir / f"{split}.jsonl")
        manifest["splits"][split] = stats
        pct = stats["changed_by_normalizer"] / max(stats["total"], 1)
        print(f"{split:11s} {stats['total']:6d} dòng, normalizer sửa "
              f"{stats['changed_by_normalizer']:6d} ({pct:.1%}) {stats['by_label']}")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {out_dir}/")


if __name__ == "__main__":
    main()
