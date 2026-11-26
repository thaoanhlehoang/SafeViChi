# Hướng dẫn gán nhãn SafeViChi v2

Phiên bản 0.1, ngày 19/11/2026. Hai người gán thử 100 câu theo bản này, rồi sửa những chỗ hay bất đồng.

## 1. Gán cái gì

Mỗi tin nhắn được gán:

1. **Một nhãn** trong 5 nhãn dưới đây.
2. **Bằng chứng:** các cụm từ chép nguyên văn từ tin nhắn cho thấy vì sao nó nguy hại. Câu `AN_TOAN` để trống.
   Câu nguy hại cũng được để trống nếu cái hại nằm ở cả câu chứ không ở chữ nào cụ thể (ví dụ mỉa mai).
3. **Thẻ (nếu có):** `tieng_long`, `vung_mien`, `an_y`, `trich_dan`. Thẻ chỉ để phân tích lỗi, không phải nhãn.

Một mẫu có thể là một tin nhắn, hoặc một **đoạn hội thoại** hai người tối đa 8 dòng: mỗi dòng là toàn bộ lượt nói
của một người (các tin liên tiếp của cùng người gộp vào một dòng), các dòng luân phiên hai người, không ghi ai nói
(ứng dụng gửi vào mô hình đúng dạng này). Chỉ nhìn những gì có trong mẫu, không đoán bối cảnh không có trong đó.

## 2. Năm nhãn

| Nhãn | Định nghĩa | Dấu hiệu |
| --- | --- | --- |
| `AN_TOAN` | Không thuộc 4 nhãn dưới | |
| `XUC_PHAM` | Lăng mạ, chửi, miệt thị một người hay một nhóm (vùng miền, giới, tôn giáo, dân tộc, ngoại hình, nghề nghiệp…) | Từ tục nhắm vào người, gọi người bằng tên con vật, miệt thị vùng miền |
| `QUAY_ROI` | Làm phiền có chủ đích nhắm vào một người: gạ gẫm tình dục không mong muốn, bình phẩm cơ thể, bám riết, bêu riếu đời tư | Đòi ảnh nhạy cảm, mời "đi khách sạn", "sao không trả lời anh", tung chuyện riêng |
| `DE_DOA` | Nói sẽ gây hại thân thể, tài sản, danh dự, hoặc dọa để ép người khác làm gì | "tao đánh", "biết nhà mày ở đâu", "không trả tiền thì tung ảnh" |
| `LUA_DAO` | Dùng thông tin giả để dụ người nhận chuyển tiền, đưa OTP/mật khẩu, bấm link, cài ứng dụng, nộp phí | Giả danh ngân hàng, công an, bưu điện; việc nhẹ lương cao; trúng thưởng; hối thúc "trong 24h" |

## 3. Cách quyết định

Đi lần lượt, dừng ở câu trả lời "có" đầu tiên:

1. Tin nhắn có **dọa gây hại hoặc dọa để ép** ai làm gì không? → `DE_DOA`
2. Có **dụ người nhận đưa tiền, OTP, mật khẩu, bấm link hay cài ứng dụng bằng thông tin giả hoặc đáng ngờ** không? → `LUA_DAO`
3. Có **gạ gẫm, bình phẩm cơ thể, bám riết hoặc bêu riếu một người cụ thể** không? → `QUAY_ROI`
4. Có **chửi, lăng mạ, miệt thị người hoặc nhóm người** không? → `XUC_PHAM`
5. Còn lại → `AN_TOAN`

Thứ tự này cũng là quy tắc khi một câu dính nhiều loại: "Chuyển 5 triệu không tao tung ảnh con đĩ mày" là `DE_DOA`.

Với đoạn hội thoại nhiều dòng (mỗi dòng một lượt nói, luân phiên hai người): gán **một nhãn cho cả đoạn** là tình huống nặng nhất xuất hiện
trong đoạn, theo cùng thứ tự `DE_DOA` > `LUA_DAO` > `QUAY_ROI` > `XUC_PHAM` > `AN_TOAN`. Ý đồ có thể chỉ lộ ở các
dòng cuối; các dòng trước giúp hiểu ngữ cảnh (ví dụ lời "tao giết mày" giữa hai bạn đang đùa nhau về trận game).

## 4. Ví dụ và ca khó

### AN_TOAN, dù trông giống nguy hại

| Tin nhắn | Vì sao an toàn | Thẻ |
| --- | --- | --- |
| "Vãi cả chưởng, kèo này căng đét" | Chửi thề cảm thán, không nhắm ai | `tieng_long` |
| "Thằng chó, đi nhậu không mậy" | Bạn bè gọi nhau, không có ý lăng mạ (chỉ khi câu tự nói rõ là rủ rê thân mật) | `tieng_long` |
| "Ai bảo người miền Tây lười là phân biệt vùng miền đấy" | Trích lại để phản bác | `trich_dan`, `vung_mien` |
| "Mã OTP của bạn là 123456. Không chia sẻ mã này cho bất kỳ ai" | Tin OTP thật: không đòi gì, còn dặn không chia sẻ | |
| "Đơn hàng của bạn đang được giao, vui lòng giữ máy" | Thông báo giao hàng bình thường, không link lạ, không đòi phí | |
| "Tao giết mày luôn á, cười đau bụng" | Cách nói quá kiểu đùa, có dấu hiệu đùa ngay trong câu | `tieng_long` |

Nếu không chắc một câu là đùa hay thật, gán theo nghĩa đen và ghi chú lại để thảo luận.

### XUC_PHAM

| Tin nhắn | Bằng chứng | Thẻ |
| --- | --- | --- |
| "Mặt như cái l*n mà đòi làm idol" | `mặt như cái l*n` | |
| "Dân bắc kỳ toàn lũ lừa lọc" | `bắc kỳ`, `lũ lừa lọc` | `vung_mien` |
| "Đúng là não cá vàng, nói mãi không hiểu" | `não cá vàng` | |
| "Ôi giỏi quá cơ, chắc học trường làng ra" | (để trống: mỉa mai cả câu) | `an_y` |

### QUAY_ROI

| Tin nhắn | Bằng chứng |
| --- | --- |
| "Em ơi gửi anh tấm ảnh không mặc gì đi" | `gửi anh tấm ảnh không mặc gì` |
| "Sao seen mà không rep, anh nhắn 50 tin rồi đó" | `anh nhắn 50 tin rồi` |
| "Mọi người ơi con này nó ngủ với sếp mới được lên chức" | `ngủ với sếp mới được lên chức` |

Lịch sự làm quen ("Chào bạn, mình làm quen được không?") là `AN_TOAN`.

### DE_DOA

| Tin nhắn | Bằng chứng |
| --- | --- |
| "Mai tao chặn đầu mày ở cổng trường" | `chặn đầu mày ở cổng trường` |
| "Không trả nợ thì đừng trách tao đến nhà" | `đừng trách tao đến nhà` |
| "Chuyển 10 triệu trước 8h không tao gửi clip cho cả nhà mày" | `không tao gửi clip cho cả nhà mày` |

Tin nhắn dọa và đòi tiền (tống tiền) là `DE_DOA`, không phải `LUA_DAO`.

### LUA_DAO

| Tin nhắn | Bằng chứng |
| --- | --- |
| "VCB thông báo tài khoản của quý khách sẽ bị khóa, xác minh tại vcb-xacminh.com" | `tài khoản của quý khách sẽ bị khóa`, `xác minh tại vcb-xacminh.com` |
| "Tuyển CTV làm tại nhà, mỗi ngày 500-800k, chỉ cần like video, ib nhận việc" | `mỗi ngày 500-800k, chỉ cần like video` |
| "Chúc mừng bạn trúng iPhone 15, nộp phí ship 300k để nhận quà" | `nộp phí ship 300k để nhận quà` |
| "Mẹ ơi con bị tai nạn, chuyển gấp cho bạn con 20 triệu số TK này" | `chuyển gấp cho bạn con 20 triệu` |
| "Cán bộ công an quận đây, anh liên quan đường dây rửa tiền, cài app này để xác minh" | `liên quan đường dây rửa tiền`, `cài app này để xác minh` |

Quảng cáo thật (giá rõ, không đòi trả trước bất thường, không link giả) và bạn bè mượn tiền bình thường là `AN_TOAN`.

## 5. Quy tắc bằng chứng

- Chép nguyên văn, đúng chính tả như trong tin nhắn, kể cả teencode và lỗi gõ.
- Lấy cụm ngắn nhất vẫn đủ nghĩa. "mày ngu như bò" thì lấy `ngu như bò`, không lấy cả câu.
- Nhiều chỗ thì ghi nhiều cụm. Không lấy từ đệm ("thì", "là", "ạ") ở hai đầu cụm.
- Với `LUA_DAO`: lấy lời yêu cầu (chuyển tiền, đưa mã, bấm link, cài app), lời hứa hẹn bất thường và câu hối thúc.
- Với đoạn hội thoại: mỗi cụm nằm gọn trong **một dòng**; được lấy nhiều cụm từ nhiều dòng khác nhau.

## 6. Quy trình

1. Hai người gán độc lập 100 câu đầu, không bàn trước.
2. Tính Cohen's kappa (`src/v2/labeling/agreement.py`). Mục tiêu từ 0,6 trở lên.
3. Ngồi xem các câu bất đồng, thêm ví dụ vào mục 4 cho đến khi rõ.
4. Tập test: cả hai cùng gán mọi câu; câu bất đồng thì thảo luận chốt và ghi lại lý do.
5. Tập train: không duyệt tay toàn bộ. Câu thật chỉ được giữ nếu nhãn LLM khớp với ít nhất một người gán gốc
   (`python -m src.v2.data.dong_thuan`). Để biết nhãn train sai bao nhiêu, hai người gán độc lập mẫu 200 câu trong
   `data/v2/raw/kiem_nhan_A.csv` và `kiem_nhan_B.csv` (chỉ điền cột `nhan`, có thể thêm `bang_chung`, `ghi_chu`;
   không mở `data/v2/interim/kiem_nhan_khoa.jsonl` và không bàn với nhau khi đang gán), rồi chạy
   `python -m src.v2.labeling.agreement`. Mẫu này cũng dùng cho bước 1–2 ở trên.

## 7. Định dạng lưu

Mỗi dòng JSONL:

```json
{"id": "v2_000123", "text": "...", "nhan": "LUA_DAO", "bang_chung": ["..."], "the": [],
 "nguoi_gan": "A", "ghi_chu": ""}
```

## Phụ lục: prompt cho LLM gán sơ bộ

Dùng nguyên mục 2, 3, 4, 5 của file này làm phần hướng dẫn trong prompt, rồi thêm:

```
Bạn là người gán nhãn dữ liệu an toàn tin nhắn tiếng Việt. Làm đúng theo hướng dẫn ở trên.
Với mỗi tin nhắn, trả về đúng một dòng JSON:
{"nhan": "<một trong AN_TOAN|XUC_PHAM|QUAY_ROI|DE_DOA|LUA_DAO>",
 "bang_chung": [<các cụm chép nguyên văn từ tin nhắn, có thể rỗng>],
 "the": [<tieng_long|vung_mien|an_y|trich_dan, có thể rỗng>],
 "do_chac": <số từ 0 đến 1>,
 "ly_do": "<một câu ngắn>"}
Không thêm gì ngoài JSON. Nếu một cụm bằng chứng không xuất hiện nguyên văn trong tin nhắn, bỏ cụm đó.
```

Sau khi LLM trả về, script tự kiểm: nhãn hợp lệ, mọi cụm bằng chứng có trong tin nhắn. Câu có `do_chac` dưới 0,7
hoặc khác nhãn gốc của bộ dữ liệu nguồn được đưa vào hàng chờ người duyệt.
