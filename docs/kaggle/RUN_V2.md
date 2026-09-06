# SafeViChi Step 3 v2 — train lại + ma trận baseline

## 3 dataset upload lên Kaggle

| zip | đặt tên dataset | cỡ |
|---|---|---|
| `safevichi_v2_code.zip` | `safevichi-v2-code` | 41 KB |
| `safevichi_v2_traindata.zip` | `safevichi-v2-traindata` | 6.6 MB |
| `safevichi_v2_evaldata.zip` | `safevichi-v2-evaldata` | 708 KB |

**Notebook Settings: GPU T4 x2 + Internet ON** (cần tải `tarudesu/ViHateT5-base-HSD`
làm baseline B2).

## Đổi gì so với lần trước

| | v1 (lần trước) | v2 (lần này) |
|---|---|---|
| learning rate | 3e-4 | **1e-5** |
| FGM epsilon | 1.0 | **0.3** |
| nhãn val/test | ép cân bằng 50/50 | **tự nhiên (17.7% / 18.3%)** |
| model khởi đầu | — | `tarudesu/ViHateT5-base-HSD` (base, KHÔNG dùng checkpoint cũ) |
| ngưỡng quyết định | cố định 0.5 | **quét trên val, đóng băng, áp lên test** |

Tỉ lệ nguồn 60/30/10 (perturbed_normalized / clean_original / blindspot_raw)
**giữ nguyên** ở cả train lẫn val/test. Tập train vẫn 49% HATE — cố ý, để có đủ
tín hiệu HATE mà học; lệch prior so với test 18% được xử lý ở bước quét ngưỡng.

## Cell 1 — setup

```python
!cp -r /kaggle/input/datasets/truongdinhv/safevichi-v2-code/* /kaggle/working/
%cd /kaggle/working
!ls src/
```

Nếu báo không thấy đường dẫn, chạy `!find /kaggle/input -maxdepth 3 -iname "*safevichi*"`
rồi thay đúng path in ra vào mọi cell bên dưới.

## Cell 2 — train (~3-5 tiếng)

```python
!python -m src.classifier.train \
    --data_path /kaggle/input/datasets/truongdinhv/safevichi-v2-traindata/mixed_natural/train.jsonl \
    --val_path  /kaggle/input/datasets/truongdinhv/safevichi-v2-traindata/mixed_natural/validation.jsonl \
    --test_path /kaggle/input/datasets/truongdinhv/safevichi-v2-traindata/mixed_natural/test.jsonl \
    --output_dir /kaggle/working/vihatet5-v2 \
    --per_device_train_batch_size 16 --fp16 \
    2>&1 | tee /kaggle/working/train_v2.log
```

LR, epsilon, model gốc đã là mặc định trong code — không cần truyền tay.

**Xem log trước khi chạy tiếp**: val F1 phải tăng dần đều qua các epoch. Nếu lại
dao động kiểu 0.69→0.79→0.67 thì dừng, báo lại, đừng chạy tiếp cell sau.

## Cell 3 — ma trận trên VALIDATION (để quét ngưỡng)

```python
!python -m src.baselines.run_matrix \
    --data_dir /kaggle/input/datasets/truongdinhv/safevichi-v2-evaldata/eval_matrix_val \
    --checkpoint /kaggle/working/vihatet5-v2 \
    --base_model tarudesu/ViHateT5-base-HSD \
    --dump_scores --out_dir /kaggle/working/results/val
```

## Cell 4 — ma trận trên TEST

```python
!python -m src.baselines.run_matrix \
    --data_dir /kaggle/input/datasets/truongdinhv/safevichi-v2-evaldata/eval_matrix \
    --checkpoint /kaggle/working/vihatet5-v2 \
    --base_model tarudesu/ViHateT5-base-HSD \
    --dump_scores --out_dir /kaggle/working/results/test
```

## Cell 5 — quét ngưỡng trên val, áp lên test

```python
!python -m src.baselines.tune_threshold \
    --val_scores  /kaggle/working/results/val/scores.jsonl \
    --test_scores /kaggle/working/results/test/scores.jsonl \
    --out_dir /kaggle/working/results/threshold
!cat /kaggle/working/results/threshold/threshold.md
```

Cell này chạy CPU, cũng chạy được ở máy local nếu tải `scores.jsonl` về.

## Cell 6 — đóng gói kết quả tải về

```python
!cd /kaggle/working && zip -qr results_v2.zip results/ train_v2.log && ls -lh results_v2.zip
```

Tải `results_v2.zip` về. Bên trong có:
- `results/test/matrix.md` — bảng ở ngưỡng 0.5
- `results/threshold/threshold.md` — **bảng chính**, ở ngưỡng đã hiệu chỉnh
- `results/{val,test}/scores.jsonl` — điểm số từng dòng, phân tích lại offline được
- `train_v2.log` — log train

Muốn giữ checkpoint mới thì zip riêng, chỉ 5 file gốc, bỏ các thư mục `checkpoint-*`:

```python
!cd /kaggle/working/vihatet5-v2 && zip -0 -qj /kaggle/working/ckpt_v2.zip \
    config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json
```

## Đọc kết quả

Số chính: **F1 lớp HATE** trong `threshold.md`, kèm khoảng tin cậy bootstrap.
Không dùng accuracy (đoán bừa CLEAN hết đã được 81.8%).

Cần B3 (`b3_t5ft_norm`) thắng `b2_t5base_norm` ở **cả C0 và C1**, chứ không chỉ
C2 như lần trước. Nếu vẫn thua ở C0/C1 thì train lại chưa giải quyết được vấn
đề, phải xem tiếp — báo lại kèm `train_v2.log`.
