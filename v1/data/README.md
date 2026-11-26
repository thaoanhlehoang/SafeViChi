# Dữ liệu SafeViChi

Thư mục chứa toàn bộ dữ liệu phục vụ huấn luyện, thẩm định và đánh giá thực nghiệm của dự án SafeViChi.

---

## 1. Cấu trúc thư mục

```
data/
├── processed/
│   ├── vietnamese_nonstandard_v1_2/   # Bộ ngữ liệu 75K câu gốc + biến thể (đầu ra Bước 1)
│   └── normalized/                    # Dữ liệu v1_2 đã qua normalize() (đầu vào của build_mixed.py)
│
├── data_final/                        # Dữ liệu phục vụ huấn luyện (Bước 3)
│   ├── train.jsonl                    # 116.943 mẫu, tỷ lệ trộn 60/30/10 (49.19% HATE)
│   ├── validation.jsonl               # 2.506 mẫu C0 câu sạch (17.28% HATE)
│   ├── test.jsonl                     # 2.376 mẫu C0 câu sạch (18.18% HATE)
│   ├── validation_mixed.jsonl         # 4.176 mẫu trộn 60/30/10 dùng cho early-stopping
│   └── manifest_mix.json              # Thông số và thống kê của bộ trộn
│
├── eval_baseline/                     # Bộ ma trận đánh giá 3 điều kiện (cho test và val)
│   ├── test/                          # C0_clean (2.376), C1_perturbed (2.376), C2_blindspot (1.920)
│   └── val/                           # Dùng riêng cho khâu dò ngưỡng (tune threshold)
│
├── data_explain/                      # Dữ liệu ViHOS phục vụ đánh giá module giải thích (Bước 4)
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── manifest.json
│
└── CHECKSUMS.json                     # Bảng mã băm SHA-256 xác thực toàn vẹn 15 file dữ liệu & model
```

---

## 2. Ý nghĩa 3 điều kiện đánh giá

Cả ba điều kiện đều được sinh ra từ cùng một tập câu gốc và ghép cặp 1-1 theo `row_id` để phục vụ kiểm định McNemar:
- `C0_clean`: Câu gốc nguyên bản chưa biến dạng (trần hiệu năng).
- `C1_perturbed`: Câu đã cấy biến thể né lọc, chưa qua chuẩn hóa.
- `C2_blindspot`: Biến thể điểm mù được thiết kế để normalizer không can thiệp (`normalize(x) == x`).

---

## 3. Quy cách dữ liệu (JSONL Schema)

- `train.jsonl`, `validation_mixed.jsonl`:
  ```json
  {"text": "nội dung văn bản", "label": "HATE", "source": "perturbed_normalized"}
  ```
  Trong đó `source` nhận một trong ba giá trị: `perturbed_normalized`, `clean_original`, `blindspot_raw`.
  
- `eval_baseline/*/*.jsonl`:
  ```json
  {"row_id": 0, "text": "nội dung văn bản", "label": "HATE", "source": "perturbed"}
  ```
  Nhãn `label` chuẩn hóa thành: `"HATE"` hoặc `"CLEAN"`.
  
- `data_explain/*/*.jsonl`:
  ```json
  {"words": ["con", "này", "ngu", "thế"], "span_mask": [0, 0, 1, 0], "label": "HATE"}
  ```

---

## 4. Tái tạo dữ liệu

Xem hướng dẫn chi tiết tại [docs/pipeline.md](../docs/pipeline.md).
Lệnh chạy nhanh:
```bash
bash scripts/build_step3_data.sh
```
