# Bố cục mã nguồn `src/`

Thư mục `src/` ở root chỉ chứa **mã nguồn vòng chung kết (`src/v2/`)** và **các mô-đun dùng chung (`src/shared/`)**. Toàn bộ mã nguồn đặc thù của vòng loại đã được gom gọn vào [v1/src/](../v1/README.md).

## Mã nguồn v2 (`src/v2/`)

| Thư mục | Vai trò |
| --- | --- |
| `src/v2/data/` | Tải nguồn, gộp dữ liệu, tách kho test, kiểm tra đồng thuận, kiểm tra rò rỉ, chia tập, nạp test người viết, sinh biến dạng C1/C2 |
| `src/v2/labeling/` | Gọi LLM gán sơ bộ (`prelabel`), sinh câu theo kịch bản (`generate`), lọc lại (`loc_sinh`), đo đồng thuận (`agreement`), thống kê chi phí (`chi_phi`) |
| `src/v2/train/` | Khung huấn luyện mô hình v2 (encoder đa nhiệm / QLoRA) |
| `src/v2/eval/` | Đánh giá chuẩn v2: Macro-F1 5 nhãn, Binary F1 (so với v1), F1 theo ý & F1 mức từ cho bằng chứng, cắt lát C0/C1/C2 × vùng miền × kịch bản, Bootstrap 95% CI, McNemar, độ tin cậy đầu ra LLM |
| `src/v2/bench/` | Đo độ trễ p50/p95, dung lượng, RAM đỉnh và năng lượng trên laptop CPU / trình duyệt |

## Mô-đun dùng chung (`src/shared/`)

- `src/shared/normalization/`: Bộ chuẩn hóa 5 tầng chống né lọc và `teencode_dict.json`.
- `src/shared/variant_generator/`: Bộ sinh biến dạng C1 và C2 (điểm mù).
- `src/shared/reporting/`: Vẽ biểu đồ và tổng hợp bảng báo cáo.
- `src/shared/utils/`: Cố định seed tái hiện (`seed.py`, mặc định `42`).
