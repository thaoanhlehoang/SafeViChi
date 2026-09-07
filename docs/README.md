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
| B2 · ViHateT5 gốc | 0,7683 | 0,6046 | 0,4627 |
| B2 · ViHateT5 gốc + chuẩn hóa | 0,7497 | 0,7218 | 0,4720 |
| **B3 · Pipeline SafeViChi đầy đủ** | **0,8552** | **0,8266** | **0,6657** |

Bốn kết luận, tất cả có ý nghĩa thống kê (McNemar ghép cặp, `p < 1e-6`):

1. **Né lọc phá được model gốc** — ViHateT5 tụt 0,1638 F1 từ C0 xuống C1.
2. **Bước 2 (chuẩn hóa) góp +0,1172** trên C1 (`p = 7,47e-30`), và **không làm
   hỏng gì** trên câu vốn đã sạch (C0: Δ = −0,0052, `p = 0,550`).
3. **Bước 3 (fine-tune đối kháng) góp thêm +0,1048** trên C1, và **+0,1937** trên
   điểm mù (`p = 1,21e-26`) — nơi chuẩn hóa hoàn toàn bó tay.
4. **Hai bước bổ trợ nhau, không thay thế nhau**: model đã fine-tune vẫn cần
   chuẩn hóa (+0,0825 trên C1, `p = 2,85e-08`).

> **Số liệu cập nhật sau khi sửa lỗi chữ hoa** (xem [`buoc4_bao_cao.md`](buoc4_bao_cao.md) §3.2):
> từ điển của ViHateT5 không có ký tự hoa nên mọi chữ hoa bị hủy thành `<unk>`,
> làm mất ~10% token. Sau khi hạ chữ thường ở đầu vào, **mọi hệ T5 đều tăng**
> (B3 đầy đủ: C0 +0,0332, C1 +0,0303, C2 +0,0287). Blacklist **không đổi** vì
> vốn đã hạ chữ thường sẵn — nghĩa là bảng cũ đang thiệt cho T5 và lợi cho
> blacklist.

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
