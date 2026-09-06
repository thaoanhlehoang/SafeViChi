#!/usr/bin/env bash
# Dựng lại TOÀN BỘ dữ liệu Bước 3 từ đầu ra Bước 1 (v1_2).
#
# Chạy từ gốc repo:
#     bash scripts/build_step3_data.sh
#
# Đầu vào bắt buộc:
#     data/processed/vietnamese_nonstandard_v1_2/{train,validation,test}.jsonl
#
# Đầu ra:
#     data/processed/normalized/          (trung gian Bước 2)
#     data/data_final/                    (bộ huấn luyện Bước 3)
#     data/eval_baseline/{test,val}/      (ma trận 3 điều kiện)
#
# ⚠️ Script này KHÔNG ghi đè data/processed/normalized/ nếu đã có sẵn. Lý do
#    xem cảnh báo đầu file src/dataset_builder/build_normalized.py — bản
#    normalizer hiện tại không tái tạo đúng file cũ, và file cũ mới là thứ đã
#    dùng để train ra checkpoint đang có.

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=42
SRC=data/processed/vietnamese_nonstandard_v1_2

echo "=============================================================="
echo " [0/5] Kiểm tra đầu vào"
echo "=============================================================="
for split in train validation test; do
    test -f "$SRC/$split.jsonl" || { echo "THIẾU $SRC/$split.jsonl"; exit 1; }
    printf '  %-12s %s dòng\n' "$split" "$(wc -l < "$SRC/$split.jsonl")"
done

echo
echo "=============================================================="
echo " [1/5] Bước 2: áp normalize() lên câu đã cấy biến thể"
echo "=============================================================="
python -m src.dataset_builder.build_normalized \
    --source_dir "$SRC" \
    --out_dir data/processed/normalized \
    --seed "$SEED"

echo
echo "=============================================================="
echo " [2/5] Bộ trộn 60/30/10 -> data/data_final/"
echo "=============================================================="
# --splits train validation: bỏ split test, vì test của Bước 3 là ma trận
#   3 điều kiện chứ không phải bản trộn.
# Không truyền --balance_eval: giữ phân bố nhãn tự nhiên (~18% HATE).
python -m src.dataset_builder.build_mixed \
    --normalized_dir data/processed/normalized \
    --source_dir "$SRC" \
    --out_dir data/data_final \
    --splits train validation \
    --clean_ratio 0.30 --blind_ratio 0.10 \
    --seed "$SEED"

echo
echo "=============================================================="
echo " [3/5] Ma trận 3 điều kiện — TEST"
echo "=============================================================="
python -m src.dataset_builder.build_eval_matrix \
    --source_path "$SRC/test.jsonl" \
    --out_dir data/eval_baseline/test \
    --seed "$SEED"

echo
echo "=============================================================="
echo " [4/5] Ma trận 3 điều kiện — VALIDATION (để dò ngưỡng)"
echo "=============================================================="
python -m src.dataset_builder.build_eval_matrix \
    --source_path "$SRC/validation.jsonl" \
    --out_dir data/eval_baseline/val \
    --seed "$SEED"

echo
echo "=============================================================="
echo " [5/5] Chốt C0 câu sạch làm validation/test của data_final"
echo "=============================================================="
# data_final/{validation,test}.jsonl CHÍNH LÀ điều kiện C0 — cùng một file,
# nhân đôi ra đây cho tiện dùng, checksum trùng nhau là đúng chứ không phải lỗi.
cp data/eval_baseline/val/C0_clean.jsonl  data/data_final/validation.jsonl
cp data/eval_baseline/test/C0_clean.jsonl data/data_final/test.jsonl
echo "  data_final/validation.jsonl <- eval_baseline/val/C0_clean.jsonl"
echo "  data_final/test.jsonl       <- eval_baseline/test/C0_clean.jsonl"

echo
echo "=============================================================="
echo " Đối chiếu checksum với bộ đã dùng để train"
echo "=============================================================="
python -m scripts.checksums || {
    echo
    echo "  [!] Có sai lệch. Nếu bạn CỐ Ý dựng bộ mới thì chạy:"
    echo "        python -m scripts.checksums --write"
    echo "      và nhớ train lại — checkpoint hiện tại thuộc về bộ dữ liệu cũ."
    exit 1
}
