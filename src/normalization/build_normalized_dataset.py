"""
Tạo dataset đã chuẩn hóa từ dữ liệu biến thể.

Đọc train.jsonl (và validation/test), lấy cột text (biến thể) + label,
chạy qua bộ normalize, xuất ra file JSONL mới với 2 cột: text, label.

Chạy:
    python src/normalization/build_normalized_dataset.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.normalization.normalizer import normalize

INPUT_DIR = project_root / "data" / "processed" / "vietnamese_nonstandard_v1_2"
OUTPUT_DIR = project_root / "data" / "processed" / "normalized"


def process_split(split_name: str) -> None:
    """Đọc 1 split (train/validation/test), chuẩn hóa, ghi ra file mới."""
    input_path = INPUT_DIR / f"{split_name}.jsonl"
    if not input_path.exists():
        print(f"⚠ Không tìm thấy {input_path}, bỏ qua.")
        return

    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    print(f"\n{'='*60}")
    print(f"Đang xử lý: {split_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    start_time = time.time()
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            variant_text = data["text"]
            label = data["label"]

            normalized_text = normalize(variant_text)

            row = {
                "text": normalized_text,
                "label": label,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count % 10000 == 0:
                elapsed = time.time() - start_time
                speed = count / elapsed
                print(f"  ... {count:,} dòng ({speed:.0f} dòng/s)")

    elapsed = time.time() - start_time
    print(f"  ✓ Hoàn thành {split_name}: {count:,} dòng trong {elapsed:.1f}s")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TẠO DATASET CHUẨN HÓA TỪ DỮ LIỆU BIẾN THỂ")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    for split in ["train", "validation", "test"]:
        process_split(split)

    print(f"\n{'='*60}")
    print("✓ Hoàn thành tất cả!")
    print(f"Dữ liệu chuẩn hóa nằm ở: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
