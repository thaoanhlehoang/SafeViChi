# Thẻ dữ liệu SafeViChi v2 – tập train (bàn giao đợt 1, 20/11/2026)

Số liệu trong thẻ này lấy từ `data/v2/train_stats.json` (sinh bằng `python -m src.v2.data.split`); nếu chạy lại,
xem file đó thay vì số ở đây. Checksum: `data/v2/CHECKSUMS.json`.

## Có gì

| File | Nội dung |
| --- | --- |
| `data/v2/train.jsonl` | Tập train, 13.392 dòng, đã trộn thứ tự (seed 42). Chỉ có `id`, `text`, `nhan`, `bang_chung` |
| `data/v2/so_sanh.jsonl` | Cùng các dòng, cùng thứ tự, đủ mọi trường + `meta`. Dùng để phân tích theo nhóm, ghép theo `id` |
| `data/v2/train_stats.json` | Phân bố nhãn, nguồn, số dòng; checksum file nguồn |
| `data/v2/validation.jsonl`, `data/v2/test.jsonl` | **Chưa có.** Bàn giao đợt 2 (21/11) sau khi hai người gán tay kho test |

## Định dạng mỗi dòng

`train.jsonl` chỉ có `id`, `text`, `nhan`, `bang_chung`. `so_sanh.jsonl` có đủ định dạng trong `CLAUDE.md`: thêm
`nguon`, `nguon_id`, `cach_gan`, `the`, `so_dong`, `bien_dang` (hiện chỉ `C0`), `nhom_trung`, `split` (= `train`), cộng
trường phụ `meta` (nguồn thật/sinh, nhãn gốc, kịch bản, kênh, giọng... dùng để phân tích theo nhóm, **không đưa vào
mô hình**).

- `text` là đoạn hội thoại 1–8 dòng ngăn bằng `\n`, mỗi dòng là toàn bộ lượt nói của một người, không ghi người nói.
- Đầu ra mô hình cần học: `{"nhan": ..., "bang_chung": [...]}`. `bang_chung` chép nguyên văn, mỗi cụm nằm trong một
  dòng; `AN_TOAN` luôn rỗng.

## Phân bố

| Nhãn | Câu thật | Đoạn LLM sinh | Tổng |
| --- | --- | --- | --- |
| `AN_TOAN` | 3.967 | 2.000 | 5.967 |
| `XUC_PHAM` | 2.527 | 470 | 2.997 |
| `LUA_DAO` | 0 | 1.751 | 1.751 |
| `DE_DOA` | 156 | 1.361 | 1.517 |
| `QUAY_ROI` | 62 | 1.098 | 1.160 |

`AN_TOAN` gấp khoảng 5 lần `QUAY_ROI`: nên dùng trọng số theo nhãn hoặc lấy mẫu cân bằng khi train.

## Nguồn và cách gán nhãn

- **Câu thật** (ViHSD, ViHOS, ViTHSD, 1 dòng): DeepSeek V4.1 Flash gán lại theo `docs/v2/HUONG_DAN_GAN_NHAN.md`, chỉ
  giữ câu mà nhãn LLM khớp với ít nhất một người gán gốc. Câu `AN_TOAN` ưu tiên ca khó (có từ tục nhưng vô hại, người
  gán gốc bất đồng). Đối chiếu 100 câu với một mô hình thứ hai (Claude): khớp 86%
  (`results/v2/dong_thuan_nhan/metrics.json`; **chưa có nhãn người**).
- **Đoạn sinh** (hội thoại 1–8 dòng): DeepSeek sinh theo `docs/v2/KICH_BAN.md`, rồi được gán lại mà không biết nhãn
  yêu cầu; chỉ giữ đoạn trùng nhãn. Đoạn `AN_TOAN` sinh là "sinh đôi" dễ nhầm của kịch bản nguy hại.
- Kịch bản `LD7` (vay tiền online) để ngoài train, dùng làm cột "kịch bản chưa thấy".
- Đã khử trùng gần đúng với kho test (`data/v2/interim/kho_test.jsonl`): không câu train nào cùng nhóm với câu kho test.

## Giới hạn cần biết khi train

- **Nhãn hiếm gần như toàn câu sinh.** Điểm trên câu thật của `DE_DOA`, `QUAY_ROI`, `LUA_DAO` phải đo riêng ở test.
- **Lệch số dòng theo nhãn:** câu thật đều 1 dòng, còn đoạn sinh phần lớn nhiều dòng. Trong train, 1.591 `AN_TOAN` và
  470 `XUC_PHAM` là nhiều dòng, nhưng phần lớn câu nhiều dòng vẫn thuộc nhãn hiếm. Kiểm tra mô hình có đoán theo số
  dòng không (so điểm theo `so_dong` trên validation).
- `QUAY_ROI` ít hơn mục tiêu (~1.450) vì kịch bản `QR3` (bình phẩm ngoại hình) bị bộ lọc gán thành `XUC_PHAM`;
  hướng dẫn gán nhãn đang chưa thống nhất chỗ này.
- Chưa có biến dạng C1/C2 (bước `perturb.py`); hiện chỉ đủ cho cấu hình #5 (fine-tune câu sạch).
- Chưa có validation thật: nếu cần gỡ lỗi, tạm tách một phần train theo `nhom_trung`, **không dùng để dò ngưỡng**.
