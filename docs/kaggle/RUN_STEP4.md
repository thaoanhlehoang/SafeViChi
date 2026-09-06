# SafeViChi Bước 4 — đo độ chính xác định vị (occlusion trên ViHOS)

**KHÔNG train gì cả.** Chỉ chạy inference bằng checkpoint đã có từ Bước 3.

## 3 dataset upload lên Kaggle

Các zip KHÔNG được commit (đã vào `.gitignore`) vì tái tạo được. Dựng lại bằng:

```bash
zip -qr safevichi_step4_code.zip src/ docs/kaggle/RUN_STEP4.md -x "*__pycache__*" "*.pyc"
cd data && zip -qr ../safevichi_step4_vihos.zip data_explain/ && cd ..
cd models && zip -0 -qr ../safevichi_step4_ckpt.zip vihatet5-v2-best/ && cd ..
```

| zip | đặt tên dataset | cỡ |
|---|---|---|
| `safevichi_step4_code.zip` | `safevichi-step4-code` | ~120 KB |
| `safevichi_step4_vihos.zip` | `safevichi-step4-vihos` | ~440 KB |
| `safevichi_step4_ckpt.zip` | `safevichi-step4-ckpt` | ~1.2 GB |

Nếu checkpoint `models/vihatet5-v2-best` đã có sẵn trên Kaggle từ Bước 3 thì
bỏ qua zip thứ 3, chỉ cần sửa lại `--model_name` cho trỏ đúng chỗ.

**Lưu ý đường dẫn**: bộ ViHOS giờ nằm ở `data/data_explain/` (trước đây là
`data/processed/vihos/`), nên trong zip thư mục con tên là `data_explain/` —
các cell dưới đây đã cập nhật theo.

**Notebook Settings: GPU T4 (1 cái là đủ) + Internet OFF** — không cần tải gì
từ HF hub vì checkpoint đi kèm sẵn.

## Cell 1 — setup

```python
!cp -r /kaggle/input/datasets/truongdinhv/safevichi-step4-code/* /kaggle/working/
%cd /kaggle/working
!ls src/explainability/
```

Nếu báo không thấy đường dẫn, chạy `!find /kaggle/input -maxdepth 3 -iname "*safevichi*"`
rồi thay đúng path in ra vào mọi cell bên dưới.

## Cell 2 — smoke test 20 câu (~1 phút, chạy trước cho chắc)

```python
!python -m src.explainability.evaluate_localization \
    --model_name /kaggle/input/datasets/truongdinhv/safevichi-step4-ckpt/vihatet5-v2-best \
    --val_path  /kaggle/input/datasets/truongdinhv/safevichi-step4-vihos/data_explain/validation.jsonl \
    --test_path /kaggle/input/datasets/truongdinhv/safevichi-step4-vihos/data_explain/test.jsonl \
    --out_dir /kaggle/working/results/step4_smoke \
    --limit 20
```

Phải thấy dòng `device=cuda`. Nếu in ra `device=cpu` thì notebook chưa bật GPU,
dừng lại bật đã rồi chạy tiếp.

## Cell 3 — chạy thật (toàn bộ 537 câu val + 531 câu test)

```python
!python -m src.explainability.evaluate_localization \
    --model_name /kaggle/input/datasets/truongdinhv/safevichi-step4-ckpt/vihatet5-v2-best \
    --val_path  /kaggle/input/datasets/truongdinhv/safevichi-step4-vihos/data_explain/validation.jsonl \
    --test_path /kaggle/input/datasets/truongdinhv/safevichi-step4-vihos/data_explain/test.jsonl \
    --out_dir /kaggle/working/results/step4 \
    2>&1 | tee /kaggle/working/step4.log
```

Ước lượng (đã đo thật, không phải đoán): val 537 câu + test 531 câu = **19.3k
forward pass**. Đo trên CPU máy dev được 37.6 pass/s → **~8.5 phút trên CPU**,
trên T4 khoảng **1-3 phút**.

**Lưu ý**: với quy mô này, chạy thẳng ở máy local (CPU, ~8.5 phút) là hoàn toàn
ổn — không bắt buộc phải lên Kaggle. Lệnh local:

```bash
python -m src.explainability.evaluate_localization --out_dir results/step4
```

## Cell 4 — tải kết quả về

```python
!cd /kaggle/working && zip -qr step4_results.zip results/ step4.log && ls -lh step4_results.zip
```

Bên trong:
- `results/step4/localization.json` — **số liệu chính**: bảng quét ngưỡng trên
  val, ngưỡng đã đóng băng, và P/R/F1 trên test.
- `results/step4/predictions_test.jsonl` — từng câu: điểm occlusion mỗi từ,
  cụm từ dự đoán vs cụm từ gold. Dùng để soi định tính / lấy ví dụ cho báo cáo.

## Đọc kết quả

Số chính: **`test.micro_f1`** trong `localization.json` — mức trùng khớp giữa
cụm từ occlusion chỉ ra và cụm từ con người khoanh (ViHOS), tính ở mức từ.

Ngưỡng được quét trên **val** rồi đóng băng áp lên **test** (giống kỷ luật quét
ngưỡng ở Bước 3) — không được chọn ngưỡng theo kết quả test.

Điểm occlusion là **logit margin**, không phải xác suất (lý do: model tự tin cực
đoan, softmax bão hòa về đúng 1.0/0.0 trong float32 và xoá sạch tín hiệu — xem
`src/explainability/occlusion.py`).

**Mặc định `--scoring relative --window 2` (CẤU HÌNH CHÍNH THỨC)**: điểm được
chuẩn hóa theo từ mạnh nhất trong TỪNG CÂU rồi mới áp ngưỡng (ngưỡng nằm trong
dải 0..1, "mạnh bằng ít nhất X% cụm mạnh nhất của chính câu này"), và mỗi lần
che 2 từ liền kề thay vì 1. Hai lựa chọn này đến từ việc soi số liệu, không
phải chọn cho đẹp — xem `plan.md` §4.3 để có đầy đủ bảng so sánh 4 cấu hình đã
thử (`absolute`/`relative` x `window` 1/2/3) và lý do loại từng cái.

Số chính thức trên ViHOS test: **micro F1 = 0.5455**, recall 0.6977 (ưu tiên
recall — với công cụ hỗ trợ kiểm duyệt, bỏ sót nguy hiểm hơn flag nhầm),
chỉ 1.9% số câu trượt hoàn toàn.

`--scoring absolute` và `--window 1/3` vẫn giữ trong code để tái lập/so sánh,
nhưng KHÔNG dùng để báo cáo số liệu chính.
