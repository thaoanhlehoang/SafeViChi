#!/usr/bin/env bash
# Chạy lại toàn bộ phần ĐÁNH GIÁ của Bước 3: ma trận 6 hệ thống × 3 điều kiện,
# trên cả validation lẫn test, rồi dò ngưỡng.
#
#     bash scripts/run_step3_eval.sh [đường_dẫn_checkpoint]
#
# CẦN GPU. Trên CPU vẫn chạy được nhưng rất lâu (ước tính >6 giờ cho đủ 2 lần
# ma trận); để smoke-test trên CPU thì thêm --limit 32 vào hai lệnh run_matrix.
#
# Bước dò ngưỡng và vẽ biểu đồ thì KHÔNG cần GPU: chúng chỉ đọc scores.jsonl.

set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-models/vihatet5-v2-best}"
SEED=42

echo "Checkpoint B3: $CKPT"
test -d "$CKPT" || { echo "Không thấy thư mục checkpoint: $CKPT"; exit 1; }

echo
echo "=============================================================="
echo " [1/3] Ma trận trên VALIDATION (để dò ngưỡng)"
echo "=============================================================="
python -m src.baselines.run_matrix \
    --data_dir data/eval_baseline/val \
    --checkpoint "$CKPT" \
    --out_dir results/step3/val \
    --seed "$SEED" \
    --dump_scores

echo
echo "=============================================================="
echo " [2/3] Ma trận trên TEST (số liệu báo cáo)"
echo "=============================================================="
python -m src.baselines.run_matrix \
    --data_dir data/eval_baseline/test \
    --checkpoint "$CKPT" \
    --out_dir results/step3/test \
    --seed "$SEED" \
    --dump_scores

echo
echo "=============================================================="
echo " [3/3] Dò ngưỡng trên val, đóng băng, áp lên test (CPU)"
echo "=============================================================="
python -m src.baselines.tune_threshold \
    --val_scores  results/step3/val/scores.jsonl \
    --test_scores results/step3/test/scores.jsonl \
    --out_dir     results/step3/threshold \
    --seed "$SEED"

echo
echo "Xong. Kết quả ở results/step3/{val,test,threshold}/"
echo "Vẽ biểu đồ (CPU): python -m src.reporting.make_figures"
