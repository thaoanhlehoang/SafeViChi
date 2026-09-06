# Hướng dẫn chạy lại pipeline Bước 3

> Dành cho người muốn dựng lại toàn bộ Bước 3 từ đầu, hoặc chỉ chạy lại một khâu.
> Số liệu và giải thích kết quả: [`buoc3_bao_cao.md`](buoc3_bao_cao.md).
> Vai trò từng file: [`buoc3_ban_do_file.md`](buoc3_ban_do_file.md).

---

## 0. Đọc trước khi chạy bất cứ thứ gì

**Bạn không cần chạy lại toàn bộ.** Repo đã có sẵn dữ liệu, checkpoint và kết
quả. Chọn đúng phần mình cần:

| Bạn muốn | Đọc mục | Cần GPU? | Mất bao lâu |
|---|---|:---:|---|
| Chỉ vẽ lại hình/bảng | [§5](#5-vẽ-lại-hình-và-bảng) | Không | ~5 giây |
| Dò lại ngưỡng, tính lại thống kê | [§4](#4-dò-ngưỡng) | Không | ~30 giây |
| Chấm lại ma trận 6×3 với checkpoint có sẵn | [§3](#3-chạy-ma-trận-đánh-giá) | **Có** | ~40 phút |
| Train lại model từ đầu | [§2](#2-huấn-luyện) | **Có** | ~5 giờ |
| Dựng lại toàn bộ dữ liệu | [§1](#1-dựng-lại-dữ-liệu) | Không | ~20 phút |

⚠️ **Cảnh báo lớn nhất:** dựng lại dữ liệu ở §1 sẽ tạo ra bộ **khác** với bộ đã
train ra checkpoint hiện tại. Lý do ở [§1.2](#12-vì-sao-dựng-lại-sẽ-ra-dữ-liệu-khác).
Nếu bạn dựng lại dữ liệu thì phải train lại model, không được trộn checkpoint cũ
với dữ liệu mới.

---

## Chuẩn bị môi trường

```bash
cd /mnt/winD/SafeViChi
python -m venv .venv
source .venv/bin/activate

# Bản khóa chính xác của môi trường đã tạo ra kết quả (khuyến nghị):
pip install -r requirements-lock.txt

# Hoặc bản khoảng phiên bản, nếu chấp nhận sai khác nhỏ:
pip install -r requirements.txt
```

Kiểm tra nhanh:

```bash
python -m pytest -q                # phải PASS toàn bộ
python -m scripts.checksums        # phải khớp toàn bộ 15 mục
```

`scripts.checksums` là chốt an toàn: nó băm SHA-256 mọi file dữ liệu và
checkpoint rồi so với bảng đã chốt. Seed chỉ đảm bảo "cùng đầu vào + cùng code →
cùng đầu ra"; nó **không** phát hiện được file bị sửa tay hay copy nhầm. Checksum
thì phát hiện được.

---

## 1. Dựng lại dữ liệu

### 1.1 Chạy

Một lệnh làm hết:

```bash
bash scripts/build_step3_data.sh
```

Script chạy 5 khâu, tất cả với `seed = 42`:

```
[0/5] Kiểm tra đầu vào — data/processed/vietnamese_nonstandard_v1_2/
[1/5] Bước 2: áp normalize() lên câu đã cấy biến thể
        -> data/processed/normalized/
[2/5] Bộ trộn 60/30/10
        -> data/data_final/train.jsonl (116.943 dòng)
        -> data/data_final/validation_mixed.jsonl (4.176 dòng)
[3/5] Ma trận 3 điều kiện — TEST
        -> data/eval_baseline/test/{C0_clean,C1_perturbed,C2_blindspot}.jsonl
[4/5] Ma trận 3 điều kiện — VALIDATION (để dò ngưỡng)
        -> data/eval_baseline/val/...
[5/5] Chốt C0 câu sạch làm validation/test của data_final
Đối chiếu checksum
```

Khâu 5 chỉ là copy: `data_final/{validation,test}.jsonl` **chính là** điều kiện
C0. Checksum của chúng trùng với `eval_baseline/*/C0_clean.jsonl` — đó là đúng
thiết kế, không phải lỗi.

Muốn chạy từng khâu riêng:

```bash
python -m src.dataset_builder.build_normalized --seed 42
python -m src.dataset_builder.build_mixed --out_dir data/data_final \
    --splits train validation --clean_ratio 0.30 --blind_ratio 0.10 --seed 42
python -m src.dataset_builder.build_eval_matrix \
    --source_path data/processed/vietnamese_nonstandard_v1_2/test.jsonl \
    --out_dir data/eval_baseline/test --seed 42
python -m src.dataset_builder.build_eval_matrix \
    --source_path data/processed/vietnamese_nonstandard_v1_2/validation.jsonl \
    --out_dir data/eval_baseline/val --seed 42
```

### 1.2 Vì sao dựng lại sẽ ra dữ liệu KHÁC

`build_normalized.py` **mặc định từ chối ghi đè** `data/processed/normalized/`
nếu file đã tồn tại. Đó là cố ý.

Bộ chuẩn hóa hiện tại trong `src/normalization/` **không tái tạo được** file
`normalized/` đã dùng để train — lệch khoảng 14% số dòng (339/2.376 ở split
test). Bản sinh ra file cũ rộng tay hơn ở hai chỗ: từ điển teencode có thêm mục
kiểu `vcl → vãi lồn`, và bộ separator xóa cả dấu `·` (U+00B7).

Vì `data_final/train.jsonl` dựng từ `normalized/`, dựng lại sẽ cho bộ train khác
→ checkpoint hiện tại không còn ứng với dữ liệu đó nữa.

Phân tích đầy đủ ở [`buoc3_bao_cao.md` §6.3](buoc3_bao_cao.md#63-lỗ-hổng-tái-lập-đã-phát-hiện).

**Nếu bạn thực sự muốn dựng bộ mới:**

```bash
python -m src.dataset_builder.build_normalized --force --seed 42
bash scripts/build_step3_data.sh          # sẽ báo checksum sai — đúng như dự kiến
python -m scripts.checksums --write       # chốt lại bảng băm cho bộ MỚI
```

rồi **bắt buộc train lại** ở §2.

### 1.3 Kiểm tra sau khi dựng

```bash
python -m scripts.checksums
```

Nếu bạn không chủ ý đổi gì mà checksum vẫn báo sai → có thứ đã lệch, dừng lại
điều tra chứ đừng train tiếp.

Kiểm tra phân bố nhãn (bẫy cũ: một phiên bản `build_mixed.py` từng sinh ra 100%
dòng CLEAN cho nhóm blindspot):

```bash
python -c "
import json
m = json.load(open('data/data_final/manifest_mix.json'))
for split, s in m['splits'].items():
    print(split, s['total'], s['by_source_label'])
"
```

Nhóm `blindspot_raw` phải có **cả HATE lẫn CLEAN**.

---

## 2. Huấn luyện

**Cần GPU.** Máy phát triển không có GPU nên khâu này chạy trên Kaggle (2×T4).
Hướng dẫn đóng gói và chạy trên Kaggle: [`kaggle/RUN_V2.md`](kaggle/RUN_V2.md).

Lệnh đúng như lần chạy đã tạo ra checkpoint hiện tại:

```bash
python -m src.classifier.train \
    --data_path   data/data_final/train.jsonl \
    --val_path    data/data_final/validation_mixed.jsonl \
    --test_path   data/data_final/test.jsonl \
    --output_dir  models/vihatet5-v2 \
    --seed 42 \
    --learning_rate 1e-5 \
    --fgm_epsilon 0.3 \
    --epochs 8 \
    --early_stopping_patience 2 \
    --per_device_train_batch_size 16 \
    --fp16
```

Ba tham số **không được đổi tùy tiện**:

- `--learning_rate 1e-5` — đây là fine-tune **tiếp** một checkpoint đã fine-tune
  sẵn, không phải train từ đầu. Mức 3e-4 làm F1 dao động ±11,5 điểm.
- `--fgm_epsilon 0.3` — epsilon 1.0 cộng với lr cao gây nhiễu kép.
- `--val_path` phải là bản **trộn**, không phải C0 câu sạch. Early stopping cần
  nhìn cùng loại phân bố với dữ liệu train.

Trước khi train, script ghi `models/vihatet5-v2/run_config.json` chứa toàn bộ
tham số và báo cáo seed — phòng khi job bị Kaggle cắt giữa chừng vẫn còn bằng
chứng đã chạy với cấu hình nào.

Kỳ vọng: ~5 giờ 11 phút, dừng ở epoch 6, checkpoint tốt nhất là epoch 4 với
F1 val ≈ 0,788.

### Lấy checkpoint tốt nhất về

Sau khi train, `output_dir` chứa cả các checkpoint trung gian (mỗi cái ~3,4 GB).
Chỉ giữ **model tốt nhất** cho suy luận:

```bash
mkdir -p models/vihatet5-v2-best
cp models/vihatet5-v2/{config.json,generation_config.json,model.safetensors,\
tokenizer.json,tokenizer_config.json} models/vihatet5-v2-best/
```

`model.safetensors` ở thư mục gốc **chính là** checkpoint tốt nhất, vì
`load_best_model_at_end=True` nạp lại nó trước khi lưu. Muốn chắc thì so hash:

```bash
python -c "
import json
s = json.load(open('models/vihatet5-v2/checkpoint-14620/trainer_state.json'))
print(s['best_model_checkpoint'], s['best_metric'])"
sha256sum models/vihatet5-v2/model.safetensors \
          models/vihatet5-v2/checkpoint-14620/model.safetensors
```

Hai hash phải trùng nhau.

---

## 3. Chạy ma trận đánh giá

**Cần GPU** cho phần chấm bằng model (trên CPU vẫn chạy nhưng rất lâu, ước tính
>6 giờ cho đủ hai lần ma trận).

```bash
bash scripts/run_step3_eval.sh models/vihatet5-v2-best
```

Script chạy 3 khâu:

1. Ma trận trên **validation** → `results/step3/val/` (dùng để dò ngưỡng)
2. Ma trận trên **test** → `results/step3/test/` (số liệu báo cáo)
3. Dò ngưỡng → `results/step3/threshold/` (khâu này chạy CPU)

Mỗi lần ma trận nạp lần lượt 2 model (T5 gốc từ Hugging Face + checkpoint của
bạn) và chấm 6 hệ thống × 3 điều kiện, mất ~20 phút trên T4.

Cờ `--dump_scores` (script đã bật sẵn) ghi P(hate) từng dòng ra `scores.jsonl`.
**Đừng bỏ cờ này** — không có nó thì không dò được ngưỡng và không vẽ được đường
cong ROC/PR mà không phải chạy lại GPU.

### Smoke test trên CPU

Kiểm tra đường dây còn nguyên mà không cần GPU:

```bash
python -m src.baselines.run_matrix --limit 8 --out_dir /tmp/smoke --dump_scores
```

Chạy 8 dòng mỗi điều kiện, vài phút trên CPU. Số liệu vô nghĩa (mẫu quá nhỏ),
mục đích chỉ là xác nhận nạp được model, đọc được dữ liệu, ghi được kết quả.

---

## 4. Dò ngưỡng

Chạy được **trên CPU**, chỉ đọc `scores.jsonl`:

```bash
python -m src.baselines.tune_threshold \
    --val_scores  results/step3/val/scores.jsonl \
    --test_scores results/step3/test/scores.jsonl \
    --out_dir     results/step3/threshold \
    --seed 42
```

Quy tắc **không được vi phạm**: ngưỡng chọn trên validation → đóng băng → mới áp
lên test. Chọn ngưỡng trên chính tập mình sắp báo cáo là rò rỉ, con số mất giá
trị.

Cờ `--pool_conditions` bật mặc định: mỗi hệ thống chọn **một ngưỡng duy nhất**
cho cả ba điều kiện. Lúc triển khai thật, hệ thống không biết trước câu đang tới
thuộc kiểu tấn công nào để mà đổi ngưỡng.

---

## 5. Vẽ lại hình và bảng

Chạy **trên CPU trong vài giây**, không nạp model:

```bash
python -m src.reporting.make_figures
```

Sinh 8 file PDF trong `results/step3/figures/`. Muốn đổi nơi lưu:

```bash
python -m src.reporting.make_figures --out_dir /duong/dan/khac
```

Chỉ xuất PDF. `style.save_pdf()` có `assert` chặn cứng mọi đuôi khác — đây là ràng
buộc của đề tài, không phải mặc định có thể đổi.

### Sửa hình

- Màu, font, kiểu nét, chú thích → `src/reporting/style.py`
- Bố cục từng hình → `src/reporting/make_figures.py`
- Ghép nguồn số liệu → `src/reporting/load.py`

Sau khi sửa, **hãy nhìn tận mắt** chứ đừng tin là xong:

```bash
python -m src.reporting.make_figures
cd results/step3/figures
for f in *.pdf; do pdftoppm -png -r 100 "$f" "/tmp/xem_${f%.pdf}"; done
# rồi mở các file /tmp/xem_*.png
```

Nhiều lỗi chỉ lộ ra khi nhìn: chữ chồng nhau, nhãn bị cắt cụt, chú giải đè lên
đường cong, số tràn sang ô bên cạnh. Trình kiểm thử không bắt được những lỗi đó.

---

## 6. Trình tự đầy đủ, từ số không

```bash
# 0. Môi trường
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m pytest -q

# 1. Dữ liệu  (CPU, ~20 phút)
bash scripts/build_step3_data.sh

# 2. Huấn luyện  (GPU, ~5 giờ — chạy trên Kaggle, xem docs/kaggle/RUN_V2.md)
python -m src.classifier.train \
    --data_path data/data_final/train.jsonl \
    --val_path  data/data_final/validation_mixed.jsonl \
    --test_path data/data_final/test.jsonl \
    --output_dir models/vihatet5-v2 \
    --seed 42 --learning_rate 1e-5 --fgm_epsilon 0.3 \
    --epochs 8 --early_stopping_patience 2 \
    --per_device_train_batch_size 16 --fp16

# 3. Ma trận đánh giá + dò ngưỡng  (GPU, ~40 phút)
bash scripts/run_step3_eval.sh models/vihatet5-v2-best

# 4. Hình và bảng  (CPU, ~5 giây)
python -m src.reporting.make_figures

# 5. Chốt lại bảng băm cho bộ dữ liệu mới
python -m scripts.checksums --write
```

---

## 7. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `python -m scripts.checksums` báo sai | File bị sửa/copy nhầm, hoặc bạn vừa dựng lại dữ liệu | Nếu cố ý dựng mới thì `--write` rồi train lại. Nếu không → điều tra, đừng train tiếp |
| `build_normalized.py` in "[dừng] Đã có sẵn" | Đúng như thiết kế, chống ghi đè | Đọc §1.2. Chỉ dùng `--force` khi hiểu hậu quả |
| F1 val dao động mạnh giữa các epoch | `learning_rate` quá cao | Về 1e-5, `fgm_epsilon` về 0,3 |
| Model sinh ra nhãn lạ, `invalid_generation_rate > 0` | Chuỗi đích viết hoa | Phải là `"hate"` / `"clean"` chữ thường |
| Nhóm `blindspot_raw` toàn CLEAN | Bẫy lấy mẫu cũ | Kiểm `by_source_label` trong manifest, dựng lại |
| `tune_threshold` báo thiếu file | Chưa bật `--dump_scores` lúc chạy ma trận | Chạy lại `run_matrix` với `--dump_scores` |
| Hình có chữ chồng nhau | Không nhìn lại sau khi sửa | Xuất PNG và xem, theo §5 |
| Log train có dòng `early stopping ... did not find eval_f1` | Xuất hiện ở lần evaluate cuối với tiền tố `test_` | Vô hại, bỏ qua |
| Hết bộ nhớ GPU | Batch quá lớn | Giảm `--per_device_train_batch_size`, tăng `--gradient_accumulation_steps` để giữ nguyên batch hiệu dụng |

### Muốn tái lập tới từng bit

```bash
python -m src.classifier.train ... --deterministic
```

Bật chế độ tất định của cuDNN/cuBLAS. **Chậm hơn đáng kể**, chỉ dùng khi cần
chứng minh tái lập chứ không dùng khi train thật.

Nói cho rõ: kể cả bật cờ này, train trên GPU với fp16 và `nn.DataParallel` vẫn có
thể lệch nhỏ giữa các lần chạy do thứ tự cộng dồn số thực khi gộp gradient từ
nhiều thiết bị. Seed đảm bảo cùng dữ liệu, cùng thứ tự batch, cùng khởi tạo —
không đảm bảo cùng bit cuối cùng.
