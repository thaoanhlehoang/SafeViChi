# Kịch bản sinh câu bằng LLM (v2)

Nguồn sự thật là `src/v2/labeling/kich_ban.yaml`, còn file này là bản đọc cho người. Script: `src/v2/labeling/generate.py`.

SafeViChi hướng tới trẻ em và thanh thiếu niên, nên kịch bản xoay quanh học đường, game, việc làm thêm, nợ tiền và
làm quen. Người nhận tin luôn là học sinh THCS, THPT hoặc sinh viên. Câu do LLM sinh **chỉ vào train**
(`nguon=sinh_llm`).

## Dạng dữ liệu: đoạn hội thoại

Mỗi mẫu là một đoạn hội thoại hai người, đúng dạng ứng dụng sẽ gửi vào mô hình:

```
Mày có vở không cho t mượn
t có
thế chiều hẹn ở cổng trường t mượn nha
```

- Mỗi dòng là **toàn bộ lượt nói của một người**: tin nhắn liên tiếp của cùng người gộp vào một dòng (nối bằng dấu
  cách), rồi mới xuống dòng sang người kia. Các dòng luôn luân phiên hai người. Không ghi người nói.
- Kênh hội thoại (`messenger_zalo`, `nhom_lop`, `chat_game`): 2–6 dòng. Kênh `binh_luan`, `sms`: 1 dòng.
- Nhãn gán cho cả đoạn theo tình huống nặng nhất (`DE_DOA` > `LUA_DAO` > `QUAY_ROI` > `XUC_PHAM` > `AN_TOAN`).
  Mỗi cụm bằng chứng nằm gọn trong một dòng, được lấy từ nhiều dòng.
- Nhắn dồn dập (QR2) là một dòng dài gồm nhiều tin của cùng một người.

## Cách sinh

- Một lần gọi LLM cho một **ô** = kịch bản × kênh × giọng × độ lộ (`lo_lieu` | `an_y`) × kiểu lộ, xin 5 đoạn.
- Kiểu lộ: `ngay_dau` (lộ ngay dòng đầu), `lo_dan` (vài dòng đầu bình thường, lộ ở cuối), `mot_dong` (kênh bình luận
  và SMS). Kênh hội thoại luân phiên `ngay_dau` / `lo_dan` theo ô, nên tổng số lượt gọi không tăng.
- Mỗi ô nguy hại có một ô **sinh đôi** `AN_TOAN` cùng kênh, giọng, kiểu lộ: đoạn an toàn nhưng dễ nhầm (có chữ
  trông nguy hiểm nhưng ngữ cảnh vô hại, như bạn thân đùa, nhắc nợ trà sữa).
- Mỗi kịch bản có tối đa 3 kênh × 4 giọng (chọn trong `kich_ban.yaml`) × 2 độ lộ = 24 ô. Số đoạn mục tiêu mỗi
  nhãn đặt ở `muc_tieu_doan`; nếu cần ít ô hơn, script chọn đều trong 24 ô (LUA_DAO: 16 ô mỗi kịch bản).
- Kênh: `messenger_zalo`, `nhom_lop`, `chat_game`, `binh_luan` (TikTok/Facebook), `sms`.
- Giọng: teencode Gen Z, thân mật, không dấu, viết tắt, giọng Bắc, Trung, Nam; kẻ giả danh thêm giọng trang trọng
  giả admin, shop hoặc nhà tuyển dụng.
- Dự kiến: 800 lượt gọi, khoảng 4.000 đoạn: `LUA_DAO` 160 ô ≈ 800 đoạn; `DE_DOA`, `QUAY_ROI` mỗi nhãn 120 ô ≈ 600
  đoạn; sinh đôi `AN_TOAN` 400 ô ≈ 2.000 đoạn.
- `LD7` để ngoài train (không sinh), dùng làm cột "kịch bản chưa thấy" khi đánh giá.

## Kiểm tra tự động

- Nhãn hợp lệ và khớp ô: nhãn của kịch bản, hoặc `AN_TOAN` với ô sinh đôi. Đoạn lệch nhãn bị bỏ.
- Số dòng đúng kiểu lộ (1 dòng với `mot_dong`, 2–8 dòng với kênh hội thoại). Tiền tố người nói ở đầu dòng
  ("A:", "B:", "Người 1:") tự bị bỏ.
- Cụm bằng chứng phải khớp nguyên văn trong một dòng. Cụm không khớp hoặc vắt qua hai dòng bị bỏ; đoạn nguy hại mất
  hết bằng chứng thì gắn `can_duyet=true`. Đoạn `AN_TOAN` không giữ bằng chứng.
- Bỏ đoạn dưới 3 từ và đoạn có giọng trợ lý ("Dưới đây là", "Lưu ý:", "Tôi không thể"...).
- Tên miền tự vô hiệu hóa thành dạng `xxx[.]com`.
- Khử trùng gần đúng trên cả đoạn bằng `src/dataset_builder/dedup.py`, và loại đoạn gần trùng câu test người viết.
- Ô bị LLM từ chối chỉ ghi log và bỏ qua, không cố lách. Ô lỗi mạng hoặc API được thử lại ở lần chạy sau.
- Câu ví dụ (`vi_du`, tối đa 5 mỗi kịch bản) lấy từ phần mẫu; script dừng nếu ví dụ trùng câu test.

## Quy tắc an toàn (đưa nguyên vào prompt)

- Không có nội dung tình dục tường minh, và không bao giờ mô tả trẻ vị thành niên theo hướng tình dục. Với QR1 và
  DD3, câu chỉ thể hiện kiểu tiếp cận ("gửi anh tấm ảnh riêng tư", "đừng kể với ai", "đổi thẻ game lấy ảnh"),
  không mô tả chi tiết hình ảnh hay hành vi.
- Không dùng tên, số điện thoại, tài khoản, trường học có thật; link chỉ là tên miền giả đã vô hiệu hóa dạng
  xxx[.]com.
- Câu sinh chỉ là nội dung tin nhắn, không sinh hướng dẫn cách lừa đảo hay cách thực hiện hành vi.

## Bảng kịch bản

| Mã | Nhãn | Kịch bản | Sinh đôi an toàn | Ngoài train |
| --- | --- | --- | --- | --- |
| LD1 | `LUA_DAO` | Game: nạp thẻ giá rẻ, kim cương/quân huy/Robux miễn phí, gửi mã thẻ cào hoặc đăng nhập link để nhận | Bạn rủ chơi game, khoe vừa nạp thẻ ở kênh chính thức | |
| LD2 | `LUA_DAO` | Game: mua bán, trao đổi acc; nhận tiền hoặc acc rồi biến mất, đòi "phí xác minh" | Bạn bè bàn chuyện leo rank, mượn acc chơi thử có thỏa thuận rõ | |
| LD3 | `LUA_DAO` | Giveaway giả của idol, streamer, shop: trúng quà, nộp phí ship hoặc đăng nhập để nhận | Giveaway thật, chỉ yêu cầu follow và bình luận | |
| LD4 | `LUA_DAO` | Việc làm thêm cho học sinh, sinh viên: CTV online, like/đánh giá, gõ chữ tại nhà, phải nộp cọc hoặc nạp tiền làm nhiệm vụ | Tin tuyển part-time thật (quán cà phê, gia sư), không thu cọc | |
| LD5 | `LUA_DAO` | Nick bạn bè bị hack: hỏi mượn tiền gấp, nhờ nhận mã OTP, nhờ "bình chọn giúp" qua link đăng nhập | Bạn bè hỏi mượn tiền thật, có chuyện chung rõ ràng | |
| LD6 | `LUA_DAO` | Bán vé concert, đồ cũ, đồ học tập trên nhóm: nhận cọc rồi chặn | Bạn cùng lớp pass lại sách, hẹn gặp ở trường | |
| LD7 | `LUA_DAO` | Vay tiền online nhanh cho học sinh sinh viên, chỉ cần ảnh căn cước | Bạn nhắc nhau cẩn thận với app vay | Có |
| LD8 | `LUA_DAO` | Làm quen online rồi nhờ nhận quà, đóng phí, chuyển tiền giúp | Làm quen lịch sự, không đòi gì | |
| LD9 | `LUA_DAO` | Rủ rê "đầu tư", kiếm tiền nhanh, rủ thêm bạn để nhận hoa hồng | Bạn kể chuyện tiết kiệm tiền tiêu vặt | |
| LD10 | `LUA_DAO` | Học bổng, cuộc thi giả: trúng tuyển, đóng phí giữ suất | Thông báo học bổng thật của trường, không thu phí | |
| LD11 | `LUA_DAO` | Bán khóa học, tài liệu giả: khóa luyện thi, IELTS, "khóa học kiếm tiền online" giá sốc, bán đề thi "lộ"; chuyển khoản trước rồi biến mất | Trung tâm thật giới thiệu khóa học, có địa chỉ, học thử, không ép chuyển gấp | |
| DD1 | `DE_DOA` | Bạo lực học đường: hẹn đánh, chặn đường cổng trường, "ra sau trường gặp tao" | Bạn đùa kiểu thân, có dấu hiệu đùa rõ trong câu; hẹn gặp ở cổng trường | |
| DD2 | `DE_DOA` | Đòi nợ giữa bạn bè hoặc vay nóng: dọa đến nhà, báo nhà trường, đăng lên nhóm lớp | Nhắc trả tiền trà sữa, tiền quỹ lịch sự | |
| DD3 | `DE_DOA` | Tống tiền bằng ảnh hoặc clip: "không chuyển tiền tao gửi cho cả lớp" | Bạn trêu dọa đăng ảnh dìm hàng kiểu đùa | |
| DD4 | `DE_DOA` | Dọa tung tin, bêu xấu lên confession, nhóm lớp nếu không làm theo | Nhắc nhau đừng đăng ảnh người khác khi chưa hỏi | |
| DD5 | `DE_DOA` | Trong game: dọa hack acc, dọa tìm ra ngoài đời "tao biết trường mày" | Trash talk trong game không nhắm ra đời thực | |
| QR1 | `QUAY_ROI` | Người lạ làm quen online rồi xin ảnh riêng tư, rủ gặp riêng, dặn giữ bí mật, đổi quà/thẻ game lấy ảnh | Làm quen lịch sự, rủ đi chơi nhóm | |
| QR2 | `QUAY_ROI` | Nhắn dồn dập sau khi bị từ chối, lập nick khác khi bị chặn | Hỏi lại một lần vì chưa thấy trả lời | |
| QR3 | `QUAY_ROI` | Bình phẩm ngoại hình, body shaming nhắm vào một bạn trong nhóm lớp | Khen nhau lịch sự | |
| QR4 | `QUAY_ROI` | Bêu riếu, đăng ảnh chế, rủ cả lớp tẩy chay một bạn | Kể chuyện vui về chính mình | |
| QR5 | `QUAY_ROI` | Trong game: bám đuôi, spam chat quấy rối một người chơi | Rủ nhau chơi tiếp trận sau | |

## Lệnh

```bash
python -m src.v2.labeling.generate --dry-run                      # in prompt mẫu, không gọi API
python -m src.v2.labeling.generate --kich-ban LD1 DD1 --limit-o 2  # thử nhỏ
python -m src.v2.labeling.generate --workers 4                     # chạy thật; chạy lại để tiếp tục
```

Đầu ra: `data/v2/interim/sinh_llm.jsonl`, thống kê `data/v2/interim/sinh_llm_stats.json`
(theo nhãn, kịch bản, kênh, giọng, độ lộ, câu bị bỏ, ô bị từ chối), cache thô trong `data/v2/interim/sinh_llm_cache/`.
