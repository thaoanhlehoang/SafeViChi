# Hướng dẫn tái lập Pipeline SafeViChi

Tài liệu này hướng dẫn chi tiết cách dựng lại toàn bộ quy trình từ dữ liệu thô, huấn luyện mô hình, đánh giá benchmark cho tới chạy demo on-device.

> Trong repo đã có sẵn dữ liệu đã tiền xử lý (`data/`), checkpoint mô hình tốt nhất (`models/vihatet5-v2-best/`) và kết quả thực nghiệm (`results/`). Bạn có thể chạy ngay từng phần mình cần mà không nhất thiết phải làm lại từ đầu.

---

## 1. Chuẩn bị môi trường

Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
python -m venv .venv
source .venv/bin/activate       # Trên Windows: .venv\Scripts\activate

# Cài đặt phiên bản chính xác khớp với môi trường thực nghiệm
pip install -r requirements-lock.txt

# Hoặc cài bản tổng quát nếu muốn linh hoạt phiên bản
pip install -r requirements.txt
```

Kiểm tra nhanh tính toàn vẹn của mã nguồn và dữ liệu có sẵn:

```bash
pytest -q                       # Chạy toàn bộ unit test (cần PASS toàn bộ)
python -m scripts.checksums     # Kiểm tra 15 mục dữ liệu và checkpoint qua mã băm SHA-256
```

---

## 2. Dựng dữ liệu (Bước 1 & Bước 2)

### 2.1 Chuẩn bị dữ liệu gốc

Nếu bạn muốn tạo lại dữ liệu từ số 0, cần tải các tập dữ liệu gốc từ Hugging Face:

```powershell
# Yêu cầu đăng nhập Hugging Face trước: hf auth login
./scripts/download_dataset_sources.ps1
```

Sau đó sinh bộ dữ liệu 75K câu phi chuẩn kèm nhãn và lịch sử biến dạng:

```bash
python -m src.dataset_builder.build
```

Dữ liệu đầu ra nằm tại `data/processed/vietnamese_nonstandard_v1_2/`.

### 2.2 Tạo dữ liệu phục vụ huấn luyện và đánh giá (Bước 3)

Bạn có thể chạy toàn bộ 5 khâu chuẩn bị dữ liệu bằng một script duy nhất:

```bash
bash scripts/build_step3_data.sh
```

Hoặc chạy từng lệnh độc lập:

1. **Áp lớp chuẩn hóa lên câu biến thể:**
   ```bash
   python -m src.dataset_builder.build_normalized --seed 42
   ```
   *Lưu ý:* File `data/processed/normalized/` có sẵn trong repo được sinh từ phiên bản trước. Script mặc định sẽ không ghi đè để bảo vệ checkpoint hiện tại. Chỉ thêm cờ `--force` khi bạn xác định sẽ huấn luyện lại mô hình mới.

2. **Tạo tập huấn luyện trộn 60/30/10:**
   ```bash
   python -m src.dataset_builder.build_mixed \
       --out_dir data/data_final \
       --splits train validation \
       --clean_ratio 0.30 \
       --blind_ratio 0.10 \
       --seed 42
   ```
   Tập `train.jsonl` thu được gồm 60% câu biến thể đã chuẩn hóa, 30% câu gốc sạch và 10% biến thể điểm mù.

3. **Dựng ma trận 3 điều kiện C0, C1, C2 phục vụ đánh giá:**
   ```bash
   # Tập test (để báo cáo số liệu)
   python -m src.dataset_builder.build_eval_matrix \
       --source_path data/processed/vietnamese_nonstandard_v1_2/test.jsonl \
       --out_dir data/eval_baseline/test \
       --seed 42

   # Tập validation (dùng riêng cho khâu dò ngưỡng)
   python -m src.dataset_builder.build_eval_matrix \
       --source_path data/processed/vietnamese_nonstandard_v1_2/validation.jsonl \
       --out_dir data/eval_baseline/val \
       --seed 42
   ```

---

## 3. Huấn luyện mô hình (Bước 3)

Khâu này cần GPU (khoảng 5-6 giờ trên GPU 2x Nvidia T4). Bạn có thể chạy trên máy nội bộ hoặc đẩy lên Kaggle.

### 3.1 Lệnh huấn luyện

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

**Các tham số quan trọng cần lưu ý:**
- `--learning_rate 1e-5`: Vì đây là fine-tune tiếp trên checkpoint ViHateT5 đã được huấn luyện trước, learning rate cao (như 3e-4) sẽ phá vỡ trọng số và làm F1 dao động thất thường.
- `--fgm_epsilon 0.3`: Cường độ nhiễu gradient đối kháng vừa phải trên tầng embedding.
- `--val_path data/data_final/validation_mixed.jsonl`: Bắt buộc dùng tập validation có tỷ lệ trộn tương tự tập train để early-stopping dừng đúng thời điểm tối ưu.

### 3.2 Huấn luyện trên Kaggle

1. Nén thư mục mã nguồn và dữ liệu cần thiết:
   ```bash
   zip -r safevichi_train.zip src/ scripts/ data/data_final/ data/eval_baseline/ requirements.txt
   ```
2. Tạo Notebook trên Kaggle, bật GPU **2x T4**, tải file zip lên và giải nén.
3. Chạy lệnh cài đặt và thực thi huấn luyện tương tự như trên.
4. Sau khi huấn luyện hoàn tất, tải checkpoint tốt nhất về máy cục bộ và đặt vào `models/vihatet5-v2-best/`.

---

## 4. Đánh giá Benchmark 6x3 & Dò ngưỡng

Đánh giá 6 hệ thống trên ma trận 3 điều kiện (C0 câu sạch, C1 câu né lọc, C2 câu điểm mù):

```bash
# Chạy đánh giá trên cả validation và test (cần GPU, mất khoảng 40 phút)
bash scripts/run_step3_eval.sh models/vihatet5-v2-best
```

Quy trình tự động gồm:
1. Tính điểm xác suất và ghi ra `scores.jsonl` cho cả 2 tập `val` và `test`.
2. Dò ngưỡng phân loại tối ưu trên tập validation rồi đóng băng áp lên tập test:
   ```bash
   python -m src.baselines.tune_threshold \
       --val_scores  results/step3/val/scores.jsonl \
       --test_scores results/step3/test/scores.jsonl \
       --out_dir     results/step3/threshold \
       --seed 42
   ```
3. Xuất toàn bộ 8 biểu đồ và bảng kết quả dạng PDF:
   ```bash
   python -m src.reporting.make_figures
   ```
   Các file PDF được lưu tại `results/step3/figures/`.

*Lưu ý kiểm tra nhanh (Smoke test):* Muốn kiểm tra luồng đánh giá trên CPU mà không cần GPU, chạy với giới hạn vài dòng:
```bash
python -m src.baselines.run_matrix --limit 8 --out_dir results/tmp_smoke --dump_scores
```

---

## 5. Module giải thích bằng Occlusion (Bước 4)

Khâu này chạy trên CPU, không đòi hỏi GPU.

1. **Chuẩn bị dữ liệu ViHOS:**
   ```bash
   python -m src.dataset_builder.build_vihos
   ```
   Dữ liệu được lưu tại `data/data_explain/`.

2. **Đánh giá độ chính xác khoanh vùng vi phạm:**
   ```bash
   python -m src.explainer.evaluate_localization --out_dir results/step4
   ```
   Quá trình mất khoảng 8-10 phút trên CPU, tính toán F1 micro/macro và độ phủ từ vi phạm so với nhãn chuẩn (gold span).

3. **Chạy thử nghiệm giải thích trên một câu tùy ý:**
   ```bash
   python -m src.explainer.demo "mày ngu như bò vậy"
   ```
   Hệ thống sẽ in ra câu văn kèm các từ ngữ bị tô đậm theo mức độ rủi ro tương ứng.

---

## 6. Chạy Web Demo on-device

Ứng dụng demo chạy hoàn toàn bằng ONNX Runtime Web trong trình duyệt, không cần backend Python:

1. Di chuyển vào thư mục demo:
   ```bash
   cd demo
   ```
2. Khởi động một web server nội bộ đơn giản:
   ```bash
   python -m http.server 8000
   ```
3. Mở trình duyệt tại địa chỉ `http://localhost:8000`.
4. Nhập câu thử nghiệm để kiểm tra quá trình chuẩn hóa, phân loại và tô sáng từ ngữ vi phạm. Mở tab Network trong DevTools để xác minh không có request nào được gửi ra ngoài.

---

## 7. Các vấn đề thường gặp & Khắc phục

| Hiện tượng | Nguyên nhân | Hướng xử lý |
|---|---|---|
| `scripts.checksums` báo sai lệch | File dữ liệu hoặc checkpoint đã bị sửa tay hoặc tạo lại | Nếu chủ ý tạo bộ dữ liệu mới, chạy `python -m scripts.checksums --write` rồi huấn luyện lại mô hình. |
| `build_normalized` từ chối ghi đè | Cơ chế bảo vệ dữ liệu cũ đã dùng huấn luyện | Chỉ dùng cờ `--force` khi bạn thực sự muốn làm mới dữ liệu và chấp nhận train lại. |
| F1 validation dao động mạnh giữa các epoch | Learning rate quá cao | Giảm về `1e-5` và đặt `fgm_epsilon` ở `0.3`. |
| Tokenizer sinh nhãn `<unk>` | Truyền chuỗi nhãn viết HOA | Tokenizer ViHateT5 chỉ có chữ thường trong từ điển; luôn chuẩn hóa về chữ thường (`hate` / `clean`). |
| Lỗi thiếu bộ nhớ GPU khi huấn luyện | Kích thước batch quá lớn | Giảm `--per_device_train_batch_size` xuống 8 và tăng `--gradient_accumulation_steps` lên 2 để giữ nguyên batch hiệu dụng. |
