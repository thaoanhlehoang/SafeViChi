"""
Kiểm tra hiệu năng của bộ chuẩn hóa (Normalizer) trên 100 mẫu biến thể.

Quy trình:
1. Đọc ngẫu nhiên 100 mẫu từ file data/processed/vietnamese_nonstandard_v1/train.jsonl
2. Lấy original_text và text (đã bị làm nhiễu).
3. Chạy text nhiễu qua bộ chuẩn hóa (normalizer.py).
4. In kết quả ra màn hình (Original -> Variant -> Normalized).

Chạy:
    python src/normalization/evaluate_variants.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH để import được src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.normalization.normalizer import normalize

JSONL_PATH = project_root / "data" / "processed" / "vietnamese_nonstandard_v1" / "train.jsonl"

def main():
    print("=" * 100)
    print("KIỂM TRA HIỆU NĂNG NORMALIZER TRÊN 100 MẪU (RULE-BASED)")
    print("=" * 100)
    print("Cấu trúc:")
    print("  [V] Biến thể (text)")
    print("  [N] Chuẩn hóa (sau khi qua normalizer)")
    print("-" * 100)

    if not JSONL_PATH.exists():
        print(f"Lỗi: Không tìm thấy file {JSONL_PATH}")
        return

    # Đọc tất cả các dòng vào list
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Lấy ngẫu nhiên 100 mẫu
    random.seed(42)
    sampled_lines = random.sample(lines, min(100, len(lines)))

    count = 0

    for line in sampled_lines:
        data = json.loads(line)
        original_text = data.get("original_text", "")
        variant_text = data.get("text", "")
        applied_types = data.get("perturbation_types", [])

        normalized_text = normalize(variant_text)
        count += 1
        label = data.get("label", "UNKNOWN")

        print(f"Mẫu #{count:03d} | Nhãn: {label} | Nhiễu: {', '.join(applied_types)}")
        print(f"  [V] {variant_text}")
        print(f"  [N] {normalized_text}")
        print("-" * 100)

if __name__ == "__main__":
    main()
