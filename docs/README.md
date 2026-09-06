# Tài liệu SafeViChi

SafeViChi — phát hiện nội dung độc hại tiếng Việt chạy trên thiết bị, bền trước
các cách né lọc (teencode, bỏ dấu, homoglyph, chèn ký tự phân cách, lỗi gõ).

## Kiến trúc 4 bước

| Bước | Nội dung | Trạng thái | Tài liệu |
|:---:|---|---|---|
| 1 | Bộ sinh biến thể & bộ dữ liệu 75K | ✅ Xong | [`dataset_pipeline.md`](dataset_pipeline.md) |
| 2 | Bộ chuẩn hóa 5 tầng theo luật | ✅ Xong | [`dataset_pipeline.md`](dataset_pipeline.md) |
| 3 | Huấn luyện đối kháng & so sánh 3 baseline | ✅ Xong | [`buoc3_bao_cao.md`](buoc3_bao_cao.md) |
| 4 | Module giải thích bằng occlusion | ✅ Xong | [`buoc4_bao_cao.md`](buoc4_bao_cao.md) |

## Tài liệu Bước 3

Đọc theo thứ tự này tùy mục đích:

| Bạn muốn | Đọc |
|---|---|
| Xem kết quả, số liệu, đánh giá | [`buoc3_bao_cao.md`](buoc3_bao_cao.md) |
| Chạy lại pipeline | [`buoc3_tai_lap_pipeline.md`](buoc3_tai_lap_pipeline.md) |
| Biết file nào làm gì | [`buoc3_ban_do_file.md`](buoc3_ban_do_file.md) |
| Chạy phần cần GPU trên Kaggle | [`kaggle/RUN_V2.md`](kaggle/RUN_V2.md) |

## Kết quả Bước 3 tóm gọn

F1 của lớp HATE trên tập test, ngưỡng đã dò trên validation rồi đóng băng:

| Hệ thống | C0 câu sạch | C1 né lọc | C2 điểm mù |
|---|---:|---:|---:|
| B1 · Blacklist | 0,6551 | 0,5982 | 0,4564 |
| B2 · ViHateT5 gốc | 0,7279 | 0,5973 | 0,4625 |
| B2 · ViHateT5 gốc + chuẩn hóa | 0,7217 | 0,7198 | 0,4763 |
| **B3 · Pipeline SafeViChi đầy đủ** | **0,8220** | **0,7963** | **0,6370** |

Bốn kết luận, tất cả có ý nghĩa thống kê (McNemar ghép cặp, `p < 0,001`):

1. **Né lọc phá được model gốc** — ViHateT5 tụt 0,1395 F1 từ C0 xuống C1.
2. **Bước 2 (chuẩn hóa) góp +0,1225** trên C1, và **không làm hỏng gì** trên câu
   vốn đã sạch.
3. **Bước 3 (fine-tune đối kháng) góp thêm +0,0765** trên C1, và **+0,1607** trên
   điểm mù — nơi chuẩn hóa hoàn toàn bó tay.
4. **Hai bước bổ trợ nhau, không thay thế nhau**: model đã fine-tune vẫn cần
   chuẩn hóa (+0,0919 trên C1, `p = 4,69e-10`).

Chi tiết, khoảng tin cậy, điểm yếu và các sai lầm thiết kế đã sửa: xem
[`buoc3_bao_cao.md`](buoc3_bao_cao.md).

## Tài liệu Bước 4

| Bạn muốn | Đọc |
|---|---|
| Xem kết quả, số liệu, đánh giá | [`buoc4_bao_cao.md`](buoc4_bao_cao.md) |
| Chạy phần này trên Kaggle | [`kaggle/RUN_STEP4.md`](kaggle/RUN_STEP4.md) |

## Kết quả Bước 4 tóm gọn

Khi hệ thống kết luận một câu là HATE, cụm từ nó chỉ ra có trùng với cụm con
người khoanh không? Đo trên ViHOS test (531 câu HATE), ngưỡng dò trên validation
rồi đóng băng:

| Chỉ số | Giá trị |
|---|---:|
| **micro F1 (mức từ)** | **0,5455** |
| Precision / Recall | 0,4478 / 0,6977 |
| macro F1 (trung bình theo câu) | 0,5766 |

Ba con số đáng nhớ hơn cả F1 tổng:

1. **Chỉ 1,9% số câu trượt hoàn toàn** (không trúng lấy một từ gold nào) — hệ
   thống hầu như luôn nhìn đúng vùng, sai chủ yếu ở biên.
2. **67,2% số câu đạt F1 ≥ 0,5**.
3. **Vùng khoanh rộng gấp 1,56 lần vùng gold** — đó là lý do precision (0,4478)
   thấp hơn recall (0,6977): khoanh quá tay, không phải tìm sai chỗ.

Lưu ý khi đọc: model **chưa bao giờ được học nhãn span** (Bước 3 chỉ train nhãn
nhị phân cả câu). Đây là giải thích hậu kỳ không giám sát, không so sánh trực
tiếp được với hệ thống token classification train thẳng trên span.

Chi tiết, phân tích theo độ dài câu/span, ví dụ định tính, điểm yếu và hai lỗi
phương pháp đã sửa: xem [`buoc4_bao_cao.md`](buoc4_bao_cao.md).

## Lệnh hay dùng

```bash
source .venv/bin/activate

python -m pytest -q                      # 107 test, ~13 giây
python -m scripts.checksums              # kiểm dữ liệu chưa bị sửa nhầm
python -m src.reporting.make_figures     # vẽ lại 8 file PDF, ~5 giây, không cần GPU

bash scripts/build_step3_data.sh         # dựng lại dữ liệu — ĐỌC CẢNH BÁO TRƯỚC
bash scripts/run_step3_eval.sh models/vihatet5-v2-best   # cần GPU

# Bước 4 — không cần GPU
python -m src.dataset_builder.build_vihos                        # → data/data_explain/
python -m src.explainability.evaluate_localization --out_dir results/step4   # ~9 phút CPU
python -m src.explainability.demo "Con này óc chó thế"           # giải thích 1 câu
```

## Quy ước seed

| Khâu | Seed |
|---|---:|
| Bước 1 — sinh biến thể | 20260902 |
| Mọi khâu còn lại | 42 |

Tất cả đi qua [`src/utils/seed.py`](../src/utils/seed.py).
