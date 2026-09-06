# Bố cục dữ liệu SafeViChi (sau khi dọn — Bước 3)

```
data/
├── processed/
│   ├── vietnamese_nonstandard_v1_2/   ← ĐẦU RA BƯỚC 1 (nguồn gốc, KHÔNG xóa)
│   │                                    75K câu + biến thể; sinh ra mọi thứ bên dưới
│   └── normalized/                    ← trung gian: v1_2 đã qua normalize() (Bước 2)
│                                        là đầu vào của build_mixed.py
├── data_final/                        ← BỘ HUẤN LUYỆN BƯỚC 3
│   ├── train.jsonl              116,943   mix 60/30/10, 49.19% HATE
│   ├── validation.jsonl           2,506   C0 câu sạch, 17.28% HATE
│   ├── test.jsonl                 2,376   C0 câu sạch, 18.18% HATE
│   ├── validation_mixed.jsonl     4,176   ← file THẬT dùng early-stop khi train
│   │                                        (mix 60/30/10, 17.67% HATE)
│   └── manifest_mix.json                  thống kê build_mixed.py
└── eval_baseline/                     ← BỘ SO SÁNH 3 BASELINE (ma trận 3 điều kiện)
    ├── test/   C0_clean 2,376 | C1_perturbed 2,376 | C2_blindspot 1,920
    └── val/    C0_clean 2,506 | C1_perturbed 2,506 | C2_blindspot 2,060
                (val dùng để dò ngưỡng, đóng băng rồi áp lên test)
```

## Ba điều kiện đánh giá

Cả ba đều sinh ra từ **cùng một tập câu gốc sạch**, ghép cặp theo `row_id`
để so sánh thống kê McNemar theo cặp:

| Điều kiện | Nội dung `text` | Mục đích |
|---|---|---|
| `C0_clean` | `original_text` — chưa cấy biến thể | Trần hiệu năng, không né lọc |
| `C1_perturbed` | `text` — đã cấy biến thể, **chưa** normalize | Kịch bản né lọc chính |
| `C2_blindspot` | biến thể điểm mù (`normalize(x) == x` được assert) | Chỗ normalizer bó tay |

`C2` ít dòng hơn vì một số câu quá ngắn / không có ký tự phù hợp nên bộ sinh
điểm mù bỏ qua (456 câu ở test, 446 ở val).

## Schema JSONL

- `data_final/train.jsonl`, `validation_mixed.jsonl`: `{text, label, source}`
  với `source ∈ {perturbed_normalized, clean_original, blindspot_raw}`
- Còn lại: `{row_id, text, label, source}` với `source ∈ {clean, perturbed, blindspot}`
- `label ∈ {"HATE", "CLEAN"}`

**KHÔNG cân bằng 50/50 ở val/test** — cố ý giữ phân bố tự nhiên (~17–18% HATE)
để đo được precision, chỗ blacklist yếu nhất. Riêng `train` giữ ~49% HATE.

## Seed

Toàn bộ khâu dựng dữ liệu dùng `seed = 42` (xem `manifest_mix.json` và
`eval_baseline/*/manifest.json`).

## Tái tạo

```bash
python -m src.dataset_builder.build_mixed --no_balance_eval    # -> data_final/
python -m src.dataset_builder.build_eval_matrix                # -> eval_baseline/
```
