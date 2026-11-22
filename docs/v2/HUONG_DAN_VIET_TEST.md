# Hướng dẫn tự viết câu test (v2)

Nhóm không thu thập tin nhắn thật, nên test của `DE_DOA`, `QUAY_ROI`, `LUA_DAO` do hai bạn trong nhóm tự viết, cộng vài
chục câu thật lọc từ dữ liệu cũ. Câu người viết có `nguon="nguoi_viet"`, `cach_gan="nguoi"` và chỉ dùng cho test.

## Quy tắc quan trọng nhất

**Không đọc câu LLM sinh (`data/v2/interim/sinh_llm.jsonl`) trước khi viết xong.** Đọc rồi sẽ vô tình bắt chước,
và test không còn đo được khả năng khái quát. Câu LLM sinh gần trùng câu người viết sẽ tự bị loại khỏi train.

## Mỗi người viết gì

Bộ test có hai phần, chấm điểm riêng:

- **Trong phân phối:** kịch bản đã được sinh vào train (`LD1`–`LD6`, `LD8`–`LD11`, `DD1`–`DD5`, `QR1`–`QR5`).
- **Kịch bản chưa thấy:** `LD7`, `LD12`, `DD6`, `QR6` (đánh dấu "Ngoài train" trong `docs/v2/KICH_BAN.md`) và mọi
  câu `kich_ban = moi`. Train không có câu nào của các kịch bản này, nên phần này đo khả năng khái quát.

| Phần | Nhãn / kịch bản | Mỗi người | Cả nhóm |
| --- | --- | --- | --- |
| Trong phân phối | `DE_DOA` | 15 | 30 (+ khoảng 20–25 câu thật lọc từ dữ liệu cũ ≈ 50) |
| Trong phân phối | `QUAY_ROI` | 15 | 30 (+ câu thật ≈ 50) |
| Trong phân phối | `LUA_DAO` | 15 | 30 (+ câu thật ≈ 50) |
| Trong phân phối | `AN_TOAN` dễ nhầm | 25 | 50 |
| Chưa thấy | `LD7` vay online | 10 | 20 |
| Chưa thấy | `LD12` giả shop/shipper | 10 | 20 |
| Chưa thấy | `DD6` dọa trả thù vì mách thầy cô | 15 | 30 |
| Chưa thấy | `QR6` kiểm soát, theo dõi | 15 | 30 |
| Chưa thấy | `AN_TOAN` sinh đôi của 4 kịch bản trên (5 câu mỗi kịch bản) | 20 | 40 |

Tổng **140 câu mỗi người, 280 câu cả nhóm**. Cộng `XUC_PHAM` (~50 câu lấy từ split test của ViHSD/ViHOS, người
duyệt) và câu thật lọc từ `ung_vien.jsonl`, bộ test khoảng **390 câu**: trong phân phối ~250, chưa thấy ~140.

**Giọng vùng miền** (để đo mô hình có hiểu nhiều giọng không): trong 280 câu, mỗi người chia đều câu nguy hại cho
ba giọng, khoảng 1/3 giọng `trung` (từ địa phương như tau, mi, nỏ, mô, rứa, răng, chi), 1/3 `nam` (tui, hổng,
nha, dzậy, mấy bà), phần còn lại `bac` hoặc `chung`. Mục tiêu cả nhóm: **mỗi giọng Bắc, Trung, Nam ít nhất ~50 câu
nguy hại và ~20 câu `AN_TOAN`.** Viết đúng cách người miền đó nhắn, không chỉ thay một hai từ.

Vì sao những con số này: với 30 câu một ô, khoảng tin cậy 95% của recall rộng khoảng ±15 điểm, nên đây là mức tối
thiểu để so "trong phân phối" với "chưa thấy". Báo cáo ghi kèm khoảng tin cậy (Wilson hoặc bootstrap), không chỉ một
con số. Nếu thiếu thời gian: **giữ nguyên phần chưa thấy**, cắt phần trong phân phối trước (đã có câu thật bù vào).
Câu thật lọc từ dữ liệu cũ thiếu bao nhiêu thì người viết bù thêm cho đủ ~50 mỗi nhãn.

- Viết như tin nhắn thật mà học sinh, sinh viên nhận được: Messenger/Zalo, nhóm lớp, chat game, bình luận TikTok/Facebook.
- Trộn cách viết: có dấu chuẩn, teencode, không dấu, viết tắt, giọng Bắc, Trung, Nam.
- Có câu **lộ liễu** (ý đồ rõ) và câu **ẩn ý** (nói tránh, lịch sự giả tạo).
- Phần trong phân phối: rải đều các kịch bản trong `docs/v2/KICH_BAN.md`. Phần chưa thấy: viết đúng số câu ở bảng trên.
- Khuyến khích viết tình huống không có trong bảng, ghi `kich_ban` là `moi`.
- `AN_TOAN` dễ nhầm: tin trông giống nguy hại nhưng an toàn (bạn đùa "tao giết mày =))", nhắc trả tiền trà sữa lịch sự,
  tuyển part-time thật không thu cọc, giveaway chỉ cần follow...).

## Phiếu viết có sẵn

`data/v2/raw/test_A.csv` và `test_B.csv`, mỗi file 140 dòng đã điền sẵn `nhan`, `kich_ban`, `kenh`, `giong_vung`,
`do_lo` (`lo_lieu` hoặc `an_y`, để trống với `AN_TOAN`), `phan_test` và `goi_y` (mô tả kịch bản hoặc tình huống sinh
đôi an toàn). Người viết chỉ điền `text` và `bang_chung`, thêm `ghi_chu` nếu cần. Không đổi các cột đã điền sẵn; nếu
một dòng không hợp, ghi lý do vào `ghi_chu` và đổi `kich_ban` thành `moi`. Dòng để trống `text` sẽ bị bỏ qua khi nạp.

- Kênh `binh_luan` và `sms`: viết 1 dòng. Kênh `messenger_zalo`, `nhom_lop`, `chat_game`: 2–6 lượt nói, ngăn bằng ` || `.
- Đổi bài: trước khi đưa file cho người kia, ẩn các cột `nhan`, `bang_chung`, `kich_ban`, `do_lo`, `phan_test`, `goi_y`
  (chọn cột → Hide trong Excel). Người kia chỉ đọc `text` và điền `nhan_nguoi_kia`.
- Mở bằng Excel: file có BOM UTF-8 nên tiếng Việt hiển thị đúng; lưu lại dạng "CSV UTF-8".

## Mỗi câu ghi gì

Điền vào file CSV theo mẫu `data/v2/raw/test_nguoi_viet_mau.csv`, mỗi dòng một câu:

| Cột | Nội dung |
| --- | --- |
| `text` | Tin nhắn hoặc đoạn hội thoại hai người (tối đa 8 dòng): mỗi lượt nói một dòng (gộp các tin liên tiếp của cùng người), luân phiên hai người, không ghi ai nói. Trong CSV ngăn các lượt bằng ` \|\| ` hoặc xuống dòng trong ô |
| `nhan` | `AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA` hoặc `LUA_DAO` |
| `bang_chung` | Các cụm chép **nguyên văn** từ `text`, cách nhau bằng dấu `|`. `AN_TOAN` để trống |
| `kich_ban` | Mã trong bảng kịch bản (`LD1`, `DD2`...), hoặc `moi` |
| `kenh` | `messenger_zalo`, `nhom_lop`, `chat_game`, `binh_luan`, `sms` |
| `giong_vung` | Bắt buộc: `bac` ("tao kh cần đâu"), `trung` ("tau nỏ cần mô"), `nam` ("tui hổng cần đâu"), `chung` (không rõ miền) |
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
4. Chạy lại script với `--xuat-test` để ghi `data/v2/test.jsonl` (id, text, nhan, bang_chung như train) và
   `data/v2/test_day_du.jsonl` (mỗi câu có thêm `giong_vung`, `phan_test` là `trong_phan_phoi` hoặc `chua_thay`):
   ```
   python -m src.v2.data.load_test_nguoi_viet data/v2/raw/test_A.csv data/v2/raw/test_B.csv data/v2/raw/test_C.csv --xuat-test
   ```
   Câu lấy nhãn theo LLM khi người và LLM bất đồng ghi `cach_gan=llm_nguoi_duyet`. Sau đó mới được đọc câu LLM sinh.
5. Kiểm tra rò rỉ với train:
   ```
   python -m src.v2.data.kiem_tra_ro_ri --test data/v2/test_day_du.jsonl \
       --train data/v2/so_sanh.jsonl data/v2/interim/kho_test.jsonl
   ```
   Script báo lỗi nếu id test trùng id train, nếu train có câu thuộc kịch bản giữ ngoài train, hoặc nếu có câu trùng
   nguyên văn. Câu train gần trùng câu test (điểm 3-gram >= 0,8, so cả đoạn và từng dòng) được liệt kê trong
   `results/v2/kiem_tra_ro_ri/id_train_can_bo.txt` (`--ghi-train-sach <file>` ghi bản train đã bỏ các câu đó). Kết quả
   ngày 22/11: 0 câu gần trùng, điểm cao nhất 0,46, nên giữ nguyên train. Cặp điểm 0,5–0,8 nằm trong
   `results/v2/kiem_tra_ro_ri/cap_can_xem.csv`: hai bạn đọc, cặp nào cùng một tình huống thì thêm id train vào danh sách bỏ.
   Muốn bắt câu diễn đạt lại, thêm `--embedding <tên model sentence-transformers>`.

## Về id

id được tính từ nguồn (`v2_` + hash của `nguon|nguon_id`). Câu người viết có `nguon="nguoi_viet"`, câu sinh có
`nguon="sinh_llm"`, nên id hai bên không thể trùng; script ở bước 5 vẫn kiểm. Không cần tự đặt id. Điều quan trọng là
**nội dung** không trùng, và đó là việc của bước 5.
