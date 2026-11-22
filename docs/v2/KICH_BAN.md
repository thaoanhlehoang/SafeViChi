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
  và SMS). Kênh hội thoại có **cả hai** kiểu `ngay_dau` và `lo_dan`, kênh một dòng chỉ có `mot_dong`.
- Mỗi ô nguy hại có một ô **sinh đôi** `AN_TOAN` cùng kênh, giọng, kiểu lộ: đoạn an toàn nhưng dễ nhầm (có chữ
  trông nguy hiểm nhưng ngữ cảnh vô hại, như bạn thân đùa, nhắc nợ trà sữa).
- Mỗi kịch bản có 3 kênh và 4 giọng (`LUA_DAO`, `XUC_PHAM`) hoặc 6 giọng (`DE_DOA`, `QUAY_ROI`); số ô tối đa là
  (kênh hội thoại × 2 + kênh một dòng) × số giọng × 2 độ lộ, ví dụ DD1: (2 × 2 + 1) × 6 × 2 = 60 ô. Số đoạn mục tiêu
  mỗi nhãn đặt ở `muc_tieu_doan`; nếu cần ít ô hơn, script chọn đều các ô. Script cảnh báo khi một nhãn thiếu quá
  10% so với mục tiêu.
- Kênh: `messenger_zalo`, `nhom_lop`, `chat_game`, `binh_luan` (TikTok/Facebook), `sms`.
- Giọng: teencode Gen Z, thân mật, không dấu, viết tắt, giọng Bắc, Trung, Nam; kẻ giả danh thêm giọng trang trọng
  giả admin, shop hoặc nhà tuyển dụng.
- Phương án train 15.000 đoạn (chốt 20/11): **2.144 lượt gọi, khoảng 10.720 đoạn**: `LUA_DAO` 376 ô ≈ 1.880 đoạn;
  `DE_DOA`, `QUAY_ROI` mỗi nhãn 300 ô ≈ 1.500 đoạn; `XUC_PHAM` 96 ô ≈ 480 đoạn (chỉ kênh hội thoại, để mô hình không
  học mẹo "nhiều dòng = nhãn hiếm" vì câu `XUC_PHAM` thật đều một dòng); sinh đôi `AN_TOAN` 1.072 ô ≈ 5.360 đoạn.
- Prompt yêu cầu các đoạn trong một ô **khác nhau thật sự** (tình huống, câu mở đầu, độ dài, không lặp cụm đặc trưng)
  và **giọng tự nhiên**: câu ngắn, viết tắt, gõ sai như tin nhắn thật; không từ sách vở, không giảng đạo lý; người
  nhận phản ứng như người thật; lời dọa `DE_DOA` phải là dọa thật; đoạn `AN_TOAN` không cố chứng minh mình an toàn.
  `[SDT]`/`[STK]` chỉ xuất hiện khi người thật sẽ gửi số, kể cả trong đoạn `AN_TOAN`, để mô hình không coi chúng là
  dấu hiệu lừa đảo.
- `LD7`, `LD12`, `DD6`, `QR6` để ngoài train (không sinh), dùng làm phần "kịch bản chưa thấy" khi đánh giá: mỗi
  nhãn `LUA_DAO`, `DE_DOA`, `QUAY_ROI` có ít nhất một kịch bản chưa thấy. `LD12`, `DD6`, `QR6` được thêm ngày 20/11,
  sau khi đã sinh train, nên chắc chắn không có trong prompt sinh. `XUC_PHAM` chưa có kịch bản chưa thấy (`XP1`, `XP2`
  đều đã sinh).

## Kiểm tra tự động

- Nhãn hợp lệ và khớp ô: nhãn của kịch bản, hoặc `AN_TOAN` với ô sinh đôi. Đoạn lệch nhãn bị bỏ.
- Số dòng đúng kiểu lộ (1 dòng với `mot_dong`, 2–8 dòng với kênh hội thoại). Tiền tố người nói ở đầu dòng
  ("A:", "B:", "Người 1:") tự bị bỏ.
- Cụm bằng chứng phải khớp nguyên văn trong một dòng. Cụm không khớp hoặc vắt qua hai dòng bị bỏ; đoạn nguy hại mất
  hết bằng chứng thì gắn `can_duyet=true`. Đoạn `AN_TOAN` không giữ bằng chứng.
- Bỏ đoạn dưới 3 từ và đoạn có giọng trợ lý ("Dưới đây là", "Lưu ý:", "Tôi không thể"...).
- Tên miền tự vô hiệu hóa thành dạng `xxx[.]com`.
- Khử trùng gần đúng trên cả đoạn bằng `src/dataset_builder/dedup.py`, và loại đoạn gần trùng câu test người viết.
- Ô bị LLM từ chối chỉ ghi log và bỏ qua, không cố lách. Ô lỗi mạng hoặc API được thử lại ở lần chạy sau. Phản hồi
  thiếu ngoặc `[ ]` hoặc các object ngăn bằng dấu phẩy vẫn được đọc.
- **Lọc lại** (`src/v2/labeling/loc_sinh.py`): LLM gán nhãn lại từng đoạn theo hướng dẫn gán nhãn mà không biết nhãn
  yêu cầu; chỉ giữ đoạn trùng nhãn. Đoạn bị loại ghi ở `sinh_llm_bi_loai.jsonl` kèm lý do; tỉ lệ giữ theo nhãn và kịch
  bản ở `sinh_llm_loc_stats.json`. Thử ngày 20/11: giữ 18/20; 2 đoạn bị loại là đoạn sinh đôi "chửi đùa" dùng từ nặng,
  khớp quy tắc "không chắc là đùa thì gán theo nghĩa đen" của hướng dẫn.
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
| LD2 | `LUA_DAO` | Game: kẻ lừa là người bán hoặc người nhận acc; mỗi đoạn một chiêu: đòi cọc rồi biến mất, đòi phí mở khóa hoặc phí chuyển acc, giả trung gian uy tín, bán acc rồi lấy lại bằng email gốc, dụ đăng nhập trang giả "kiểm tra acc", nạp kim cương giá rẻ | Bạn bè bàn chuyện leo rank, mượn acc chơi thử có thỏa thuận rõ | |
| LD3 | `LUA_DAO` | Giveaway giả của idol, streamer, shop: trúng quà, nộp phí ship hoặc đăng nhập để nhận | Giveaway thật, chỉ yêu cầu follow và bình luận | |
| LD4 | `LUA_DAO` | Việc làm thêm cho học sinh, sinh viên: CTV online, like/đánh giá, gõ chữ tại nhà, phải nộp cọc hoặc nạp tiền làm nhiệm vụ | Tin tuyển part-time thật (quán cà phê, gia sư), không thu cọc | |
| LD5 | `LUA_DAO` | Nick bạn bè bị hack: hỏi mượn tiền gấp, nhờ nhận mã OTP, nhờ "bình chọn giúp" qua link đăng nhập | Bạn bè hỏi mượn tiền thật, có chuyện chung rõ ràng | |
| LD6 | `LUA_DAO` | Bán vé concert, đồ cũ, đồ học tập trên nhóm: nhận cọc rồi chặn | Bạn cùng lớp pass lại sách, hẹn gặp ở trường | |
| LD7 | `LUA_DAO` | Vay tiền online nhanh cho học sinh sinh viên, chỉ cần ảnh căn cước | Bạn nhắc nhau cẩn thận với app vay | Có |
| LD8 | `LUA_DAO` | Làm quen online rồi nhờ nhận quà, đóng phí, chuyển tiền giúp | Làm quen lịch sự, không đòi gì | |
| LD9 | `LUA_DAO` | Rủ rê "đầu tư", kiếm tiền nhanh, rủ thêm bạn để nhận hoa hồng | Bạn kể chuyện tiết kiệm tiền tiêu vặt | |
| LD10 | `LUA_DAO` | Học bổng, cuộc thi giả: trúng tuyển, đóng phí giữ suất | Thông báo học bổng thật của trường, không thu phí | |
| LD11 | `LUA_DAO` | Bán khóa học, tài liệu giả: khóa luyện thi, IELTS, "khóa học kiếm tiền online" giá sốc, bán đề thi "lộ"; chuyển khoản trước rồi biến mất | Trung tâm thật giới thiệu khóa học, có địa chỉ, học thử, không ép chuyển gấp | |
| LD12 | `LUA_DAO` | Giả shop online hoặc shipper: giao hàng thất bại, đơn lỗi cần hoàn tiền, bấm link nhập thông tin hoặc chuyển phí giao lại | Shipper thật hẹn giờ giao, shop báo đơn đang giao, không đòi tiền hay link | Có |
| DD1 | `DE_DOA` | Bạo lực học đường: hẹn đánh, chặn đường cổng trường, "ra sau trường gặp tao" | Bạn đùa kiểu thân, có dấu hiệu đùa rõ trong câu; hẹn gặp ở cổng trường | |
| DD2 | `DE_DOA` | Đòi nợ giữa bạn bè hoặc vay nóng: dọa đến nhà, báo nhà trường, đăng lên nhóm lớp | Nhắc trả tiền trà sữa, tiền quỹ lịch sự | |
| DD3 | `DE_DOA` | Tống tiền bằng ảnh hoặc clip: "không chuyển tiền tao gửi cho cả lớp" | Bạn trêu dọa đăng ảnh dìm hàng kiểu đùa | |
| DD4 | `DE_DOA` | Dọa tung tin, bêu xấu lên confession, nhóm lớp nếu không làm theo | Nhắc nhau đừng đăng ảnh người khác khi chưa hỏi | |
| DD5 | `DE_DOA` | Trong game: dọa hack acc, dọa tìm ra ngoài đời "tao biết trường mày" | Trash talk trong game không nhắm ra đời thực | |
| DD6 | `DE_DOA` | Dọa trả thù vì bị mách thầy cô, phụ huynh hoặc bị report nick: "mày mách cô thì liệu hồn" | Bạn khuyên nên báo thầy cô khi bị bắt nạt, hứa đi cùng | Có |
| QR1 | `QUAY_ROI` | Người lạ làm quen online rồi xin ảnh riêng tư, rủ gặp riêng, dặn giữ bí mật, đổi quà/thẻ game lấy ảnh | Làm quen lịch sự, rủ đi chơi nhóm | |
| QR2 | `QUAY_ROI` | Nhắn dồn dập sau khi bị từ chối, lập nick khác khi bị chặn | Hỏi lại một lần vì chưa thấy trả lời | |
| QR3 | `QUAY_ROI` | Bình phẩm ngoại hình, body shaming nhắm vào một bạn trong nhóm lớp | Khen nhau lịch sự | |
| QR4 | `QUAY_ROI` | Bêu riếu, đăng ảnh chế, rủ cả lớp tẩy chay một bạn | Kể chuyện vui về chính mình | |
| QR5 | `QUAY_ROI` | Trong game: bám đuôi, spam chat quấy rối một người chơi | Rủ nhau chơi tiếp trận sau | |
| QR6 | `QUAY_ROI` | Kiểm soát, theo dõi bạn hoặc người yêu: đòi xem tin nhắn, bắt bật định vị, cấm chơi với bạn khác, ghen tuông tra hỏi liên tục | Bạn bè hỏi han quan tâm, hai bên đồng ý chia sẻ vị trí khi đi chơi tối | Có |
| XP1 | `XUC_PHAM` | Cãi nhau rồi lăng mạ, chửi thẳng, hạ nhục người đang nói chuyện (đầu óc, học lực, gia cảnh); không dọa, không gạ gẫm | Bạn thân chửi đùa, văng tục thân mật, trêu nhau mà cả hai đều vui | |
| XP2 | `XUC_PHAM` | Miệt thị vùng miền, dân tộc, tôn giáo hoặc một nhóm người ngay trong cuộc trò chuyện, kể cả nói ẩn ý hay dùng biệt danh miệt thị | Nói chuyện vui về giọng nói, món ăn, phong tục các vùng miền, không miệt thị | |

## Lệnh

```bash
python -m src.v2.labeling.generate --dry-run                      # in prompt mẫu, không gọi API
python -m src.v2.labeling.generate --kich-ban LD1 DD1 --limit-o 2  # thử nhỏ
python -m src.v2.labeling.generate --workers 8                     # chạy thật; chạy lại để tiếp tục
python -m src.v2.labeling.loc_sinh --batch-size 20 --workers 8     # lọc lại đoạn sinh
```

Đầu ra: `data/v2/interim/sinh_llm.jsonl`, thống kê `data/v2/interim/sinh_llm_stats.json`
(theo nhãn, kịch bản, kênh, giọng, độ lộ, câu bị bỏ, ô bị từ chối), cache thô trong `data/v2/interim/sinh_llm_cache/`.
