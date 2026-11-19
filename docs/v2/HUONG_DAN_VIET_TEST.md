# Hướng dẫn tự viết câu test (v2)

Nhóm không thu thập tin nhắn thật, nên test của `DE_DOA`, `QUAY_ROI`, `LUA_DAO` do hai bạn trong nhóm tự viết, cộng vài
chục câu thật lọc từ dữ liệu cũ. Câu người viết có `nguon="nguoi_viet"`, `cach_gan="nguoi"` và chỉ dùng cho test.

## Quy tắc quan trọng nhất

**Không đọc câu LLM sinh (`data/v2/interim/sinh_llm.jsonl`) trước khi viết xong.** Đọc rồi sẽ vô tình bắt chước,
và test không còn đo được khả năng khái quát. Câu LLM sinh gần trùng câu người viết sẽ tự bị loại khỏi train.

## Mỗi người viết gì

| Nhãn | Số câu mỗi người |
| --- | --- |
| `DE_DOA` | 25 |
| `QUAY_ROI` | 25 |
| `LUA_DAO` | 25 |
| `AN_TOAN` dễ nhầm | 25 |

Tổng 100 câu mỗi người, 200 câu cả nhóm: mỗi nhãn nguy hại có 50 câu người viết, cộng vài chục câu thật lọc từ
dữ liệu cũ.

- Viết như tin nhắn thật mà học sinh, sinh viên nhận được: Messenger/Zalo, nhóm lớp, chat game, bình luận TikTok/Facebook.
- Trộn cách viết: có dấu chuẩn, teencode, không dấu, viết tắt, giọng Bắc, Trung, Nam.
- Có câu **lộ liễu** (ý đồ rõ) và câu **ẩn ý** (nói tránh, lịch sự giả tạo).
- Rải đều các kịch bản trong `docs/v2/KICH_BAN.md`. **Bắt buộc có vài câu `LD7`** (vay online cho học sinh, sinh viên):
  kịch bản này không được sinh vào train, dùng để đo "kịch bản chưa thấy".
- Khuyến khích viết tình huống không có trong bảng, ghi `kich_ban` là `moi`.
- `AN_TOAN` dễ nhầm: tin trông giống nguy hại nhưng an toàn (bạn đùa "tao giết mày =))", nhắc trả tiền trà sữa lịch sự,
  tuyển part-time thật không thu cọc, giveaway chỉ cần follow...).

## Mỗi câu ghi gì

Điền vào file CSV theo mẫu `data/v2/raw/test_nguoi_viet_mau.csv`, mỗi dòng một câu:

| Cột | Nội dung |
| --- | --- |
| `text` | Tin nhắn hoặc đoạn hội thoại hai người (tối đa 8 dòng): mỗi lượt nói một dòng (gộp các tin liên tiếp của cùng người), luân phiên hai người, không ghi ai nói. Trong CSV ngăn các lượt bằng ` \|\| ` hoặc xuống dòng trong ô |
| `nhan` | `AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA` hoặc `LUA_DAO` |
| `bang_chung` | Các cụm chép **nguyên văn** từ `text`, cách nhau bằng dấu `|`. `AN_TOAN` để trống |
| `kich_ban` | Mã trong bảng kịch bản (`LD1`, `DD2`...), hoặc `moi` |
| `kenh` | `messenger_zalo`, `nhom_lop`, `chat_game`, `binh_luan`, `sms` |
| `nguoi_viet` | `A` hoặc `B` |
| `ghi_chu` | Tùy chọn |
| `nhan_nguoi_kia` | Để trống khi viết; người kia điền ở bước đổi bài |

Bằng chứng theo mục 5 của `HUONG_DAN_GAN_NHAN.md`: cụm ngắn nhất đủ nghĩa, đúng chính tả như trong câu.

## Quy tắc an toàn

- Không có nội dung tình dục tường minh, không bao giờ mô tả trẻ vị thành niên theo hướng tình dục. Với gạ gẫm,
  tống tiền bằng ảnh: chỉ thể hiện kiểu tiếp cận ("gửi anh tấm ảnh riêng tư", "đừng kể với ai"), không mô tả chi tiết.
- Không dùng tên, số điện thoại, số tài khoản, trường học có thật. Số điện thoại ghi `[SDT]`, số tài khoản ghi `[STK]`,
  link dùng tên miền bịa dạng `xxx[.]com`.

## Sau khi viết xong

1. Hai người đổi file. Mỗi người **che cột `nhan`** của người kia, tự gán nhãn từng câu vào cột `nhan_nguoi_kia`.
2. Chạy `python -m src.v2.data.load_test_nguoi_viet data/v2/raw/<file>.csv` để kiểm lỗi và tính Cohen's kappa.
   Mục tiêu kappa từ 0,6 trở lên.
3. Ngồi xem các câu bất đồng, chốt nhãn cuối vào cột `nhan`, ghi lý do vào `ghi_chu`. Câu không chốt được thì bỏ.
4. Chạy lại script để ghi `data/v2/interim/test_nguoi_viet.jsonl`. Sau đó mới được đọc câu LLM sinh.
