<!-- Tệp sinh tự động bằng `python -m src.v2.eval.xuat_tai_lieu` từ docs/v2/bao_cao/so_lieu.tex. Không sửa tay. -->

# SafeViChi v2: đánh giá và nhận xét

Tài liệu nhận xét kết quả theo ba yêu cầu cải tiến của ban tổ chức. Bảng số đầy đủ nằm ở [KET_QUA.md](KET_QUA.md); mọi số dưới đây cùng sinh từ `results/v2/` qua script. Điểm F1 tính trên thang 0–100.

## Tóm tắt

**Điểm mạnh**

- **Vượt LLM lớn trên cả ba tập.** Macro-F1 5 nhãn của v2 là 86,5 / 78,6 / 77,1 (C0 / C1 / C2), so với 77,2 / 58,5 / 62,7 của gpt-oss-120b qua API, trong khi v2 chạy trên máy người dùng và không tốn phí mỗi tin.
- **Bền trước biến dạng.** Từ câu sạch sang 6 kiểu điểm mù chưa thấy (C2), Binary-F1 chỉ đổi −2,0 điểm (92,9 → 90,9); Macro-F1 đổi −9,4, so với −14,5 của LLM API.
- **Bằng chứng đáng tin.** BC F1 theo ý 83,0 trên C0 (LLM API 70,7); 0,0% cụm bằng chứng không chép nguyên văn từ tin nhắn.
- **Đọc được ngữ cảnh nhiều lượt.** Đọc cả hội thoại cho Macro-F1 98,0 so với 70,9 khi chỉ đọc dòng cuối; với nguy hại ẩn ý trên C0, v2 đạt 84,3 so với 64,7 của LLM API.
- **Nhẹ và nhanh.** Gói INT8 301 MiB, p50 24,7 ms trên laptop CPU, 337 ms trong trình duyệt; giữ chất lượng gần như bản FP32 (McNemar p = 0,07).

**Điểm yếu**

- **Báo nhầm câu an toàn khoảng 10%** (FPR 10,4% trên C0, so với 3,1% của LLM API). Lỗi dồn vào kịch bản chưa thấy (FPR 19,6%) và kênh SMS (FPR 23,5%).
- **Kịch bản hoàn toàn mới.** Macro-F1 trên C0 rơi từ 93,0 (kịch bản đã có trong train) xuống 62,1 (chưa thấy), giảm 30,9 điểm. Đây là giới hạn lớn nhất của mô hình.
- **`QUAY_ROI` khó nhất.** F1 trên C0 chỉ 78,3 với recall 70,6; dữ liệu train thật của nhãn này rất ít (62 câu).
- **Một số kiểu biến dạng làm tăng báo nhầm:** Chèn dấu cách (FPR 21,9%), Ký tự vô hình (zero-width) (FPR 20,7%), Teencode (FPR 20,0%), Gõ sai (FPR 21,2%).

## Yêu cầu 1: giảm lỗi giải thích và kiểm duyệt sai

### Báo nhầm (false positive)

Trên C0 có 10/96 đoạn `AN_TOAN` bị báo nhầm: 6 thành `DE_DOA`, 3 thành `LUA_DAO`, 1 thành `QUAY_ROI`, 0 thành `XUC_PHAM`. Với kịch bản đã có trong train, FPR chỉ 2,0%; phần lớn lỗi nằm ở đoạn đối chứng thuộc kịch bản chưa thấy có chứa từ khóa hay gặp trong tình huống nguy hại (“OTP”, “mất phí”, “shipper”, “báo cô”, “chặn đường”). Mô hình bám từ khóa với độ tin cậy cao, nên nâng ngưỡng chung không giải quyết gốc vấn đề; cần thêm đoạn đối chứng an toàn cho từng kịch bản.

| Đoạn tin nhắn | Nhãn đúng | v2 (độ tin cậy) | LLM API | Bằng chứng v2 | Bằng chứng đúng | Kịch bản · vùng |
|---|---|---|---|---|---|---|
| Mã OTP xác nhận đăng nhập vừa gửi tới điện thoại của bạn. Nếu bạn đang đăng nhập trên thiết bị mới, vui lòng nhập mã trên màn hình ứng dụng để tiếp tục. | `AN_TOAN` | `LUA_DAO` (91,9%) | `AN_TOAN` | “Mã OTP xác nhận đăng nhập vừa gửi”, “nhập mã trên màn hình ứng” | (rỗng) | an_toan_chung · Bắc |
| Mọi người đừng lan truyền ảnh vụ cãi nhau nữa.<br> Vậy có cần báo cô không?<br> Có, nói cô đi. Để t gửi cho cô giải quyết, đừng tự ra gặp nhóm kia. | `AN_TOAN` | `DE_DOA` (93,7%) | `AN_TOAN` | “Để t gửi cho cô giải quyết, đừng tự ra gặp nhóm kia.” | (rỗng) | DD6 · Bắc |

**Tiếng lóng.** Không đoạn tiếng lóng vô hại nào trên C0 bị báo thành `XUC_PHAM`. Trên lát cắt tiếng lóng của dev (n = 178), mmBERT-base đạt Macro-F1 74,8, cao nhất trong 4 ứng viên (ViSoBERT 62,7, BamiBERT 55,5). Đây là lý do chính chọn mmBERT-base dù ViSoBERT nhỉnh hơn ở chỉ số tổng (hai mô hình không khác biệt có ý nghĩa thống kê, p = 0,29).

### Bỏ sót (false negative)

Recall nguy hại trên C0 là 90,3 (LLM API 81,5, v1 18,5). Chỉ 23 đoạn nguy hại bị dự đoán `AN_TOAN`: 9 `DE_DOA`, 8 `QUAY_ROI`, 4 `LUA_DAO`, 2 `XUC_PHAM`. Các ca này hoặc ẩn ý (đe dọa nói tránh), hoặc thuộc kịch bản chưa xuất hiện khi train; xem ví dụ ở Yêu cầu 2. Ngoài ra `QUAY_ROI` hay bị nhầm sang `DE_DOA` (7 đoạn) vì ranh giới bám riết và dọa dẫm khá mờ.

### Hiệu chỉnh ngưỡng và tách phân loại khỏi bằng chứng

- Hệ số nhiệt độ T = 1,22 và ngưỡng bằng chứng (0,30; cụm một từ 0,40) dò trên tập `cal`, đóng băng trước khi mở test.
- Nhãn do đầu phân loại hội thoại quyết định; bằng chứng do BIO-CRF quyết định với ngưỡng riêng. Hai nhiệm vụ được đo riêng: trên C0, Binary-F1 92,9 còn BC F1 theo ý 83,0 (P 85,7, R 84,6).
- Tô oan đoạn an toàn 9,4%; đoạn nguy hại có bằng chứng rỗng 3,3%.
- Còn thiếu: ngưỡng riêng cho từng nhãn nguy hại, dò trên một tập validation do người gán (tập `cal` hiện mang nhãn LLM).

## Yêu cầu 2: khái quát trước kiểu né lọc mới

- **Kiểu biến dạng chưa thấy.** Trên C2, Macro-F1 77,1 và Binary-F1 90,9, gần bằng C1 (78,6 / 90,3). Trong C1, nhóm kiểu không có trong tăng cường lúc train đạt Macro-F1 82,2, không thấp hơn nhóm đã thấy (76,1). Nhánh Byte-CNN và tăng cường trực tuyến giúp mô hình không phụ thuộc một danh sách biến dạng cố định.
- **Kiểu yếu nhất:** Chèn dấu cách (Macro-F1 67,1), Ký tự vô hình (zero-width) (Macro-F1 72,4) ở C2; Teencode (70,9), Gõ sai (72,8) ở C1. Đây cũng là các kiểu có FPR cao nhất (xem phần Tóm tắt): mô hình vừa phân loại kém hơn vừa báo nhầm nhiều hơn.
- **Vùng miền.** Chênh Macro-F1 giữa Bắc, Trung, Nam trên C0 là 3,5 điểm (thấp nhất 84,5). FPR dao động mạnh hơn (Trung 3,1%, Bắc 12,1%, Nam 16,1%) nhưng mỗi vùng chỉ có khoảng 32 đoạn an toàn nên cần đọc thận trọng.
- **Không rò rỉ:** 0 cặp gần trùng giữa train và test; 0 mẫu train thuộc các kịch bản giữ ngoài (DD6, LD12, LD7, QR6).

### Trường hợp mô hình vẫn thất bại

Các đoạn nguy hại dưới đây bị v2 dự đoán `AN_TOAN`:

| Đoạn tin nhắn | Nhãn đúng | v2 (độ tin cậy) | LLM API | Bằng chứng v2 | Bằng chứng đúng | Kịch bản · vùng |
|---|---|---|---|---|---|---|
| Đứa nào gửi ghi âm cho cô thì chắc đang đọc tin này đấy nhỉ<br> Ai gửi thì kệ người ta đi<br> Ừ kệ thôi, giỏi thế cơ mà. Mai đi học nhớ đi đường đông người nhé | `DE_DOA` | `AN_TOAN` (87,4%) | `AN_TOAN` | (rỗng) | “Mai đi học nhớ đi đường đông người nhé” | DD6 · Bắc |
| Thiếu tiền trả phòng trọ hả? Vay online 5 triệu chỉ cần CCCD, không hỏi phụ huynh. Nhắn tui, duyệt trong buổi. | `LUA_DAO` | `AN_TOAN` (92,8%) | `LUA_DAO` | (rỗng) | “Vay online 5 triệu chỉ cần CCCD”, “duyệt trong buổi” | LD7 · Nam |
| Chị lỡ chuyển nhầm 2 triệu vô tài khoản em rồi, em kiểm tra giùm chị nha. Em chuyển lại vô [STK] này giùm chị, chị cảm ơn em nhiều lắm | `LUA_DAO` | `AN_TOAN` (83,4%) | `LUA_DAO` | (rỗng) | “Chị lỡ chuyển nhầm 2 triệu vô tài khoản em rồi”, “Em chuyển lại vô [STK] này giùm chị” | moi · Nam |
| Tau thấy mi vừa theo dõi lại người yêu cũ. Bỏ theo dõi liền đi, rồi chụp màn hình gửi tau. Đừng bắt tau phải vào tài khoản mi tự kiểm tra. | `QUAY_ROI` | `AN_TOAN` (92,5%) | `QUAY_ROI` | (rỗng) | “Bỏ theo dõi liền đi, rồi chụp màn hình gửi tau”, “Đừng bắt tau phải vào tài khoản mi tự kiểm tra” | QR6 · Trung |

- Đe dọa ẩn ý (“nhớ đi đường đông người nhé”) không có từ thô tục; LLM API cũng bỏ sót.
- Vay online chỉ cần căn cước và chiêu “chuyển nhầm tiền” khác mẫu lừa đảo quen thuộc (không đòi OTP, không có link); LLM API vẫn nhận ra cả hai.
- Kiểm soát người yêu không chứa từ xúc phạm hay dọa thân thể, chỉ là ép gửi ảnh chụp màn hình và đòi vào tài khoản.

Điểm chung: mô hình học tốt các mẫu đã thấy nhưng chưa có khái niệm tổng quát về “ép buộc” hay “dụ chuyển tiền” khi cách diễn đạt mới hoàn toàn.

## Yêu cầu 3: so sánh phương pháp và chi phí

- **v1 (ViHateT5) không đủ cho bài toán mới.** Recall nguy hại trên C0 chỉ 18,5: v1 bắt được `XUC_PHAM` (73,6%) nhưng gần như bỏ qua `LUA_DAO` (0,0%) và `QUAY_ROI` (0,0%), vì được huấn luyện cho câu đơn xúc phạm.
- **LLM lớn qua API** có FPR thấp hơn trên C0 (3,1%) và mạnh hơn ở kênh SMS (Macro-F1 87,3 so với 67,1), nhưng recall thấp (81,5), tụt mạnh khi gặp biến dạng (Macro-F1 C1 58,5), có độ trễ p50 3,46 s mỗi đoạn và buộc gửi tin nhắn ra ngoài máy.
- **v2 on-device:** p50 24,7 ms trên laptop AMD, 69,1 ms trên laptop Intel chạy pin, 337 ms trong trình duyệt, 25,0 ms trên điện thoại; năng lượng tăng thêm 0,14 Wh (CPU) và 0,27 Wh (trình duyệt) cho 1.000 đoạn.
- **Lượng tử hóa INT8** nhanh hơn PyTorch FP32 khoảng 2,45× và dùng ít RAM hơn khoảng 4,85× trên cùng máy, Macro-F1 toàn bộ đổi −1,0 điểm (McNemar p = 0,07).
- **Bộ chuẩn hóa v1 bị tắt mặc định.** Nó giúp vài kiểu C1 (chèn ký tự +17,5) nhưng làm giảm Macro-F1 trên C0 (−1,2), C2 (−1,4) và nhóm biến dạng chưa thấy (−1,1), vì đôi khi sửa sai những từ mô hình vốn hiểu đúng.
- **Chi phí tạo dữ liệu thấp:** 2,28 USD tiền API cho toàn bộ dữ liệu sinh và gán lại.

## Hạn chế

- Bộ test gốc chỉ có 334 đoạn C0 do nhóm tự viết, nên lát cắt nhỏ (vùng, kênh, người viết) có biên độ dao động lớn. Ví dụ người viết D không có đoạn `AN_TOAN` nào nên điểm của lát này không so sánh được với các lát khác.
- Phần lớn dữ liệu `QUAY_ROI`, `DE_DOA`, `LUA_DAO` trong train do LLM sinh (`LUA_DAO`: 0 câu thật), nên mô hình có thể học theo phong cách viết của LLM.
- Tập `cal` dùng để hiệu chỉnh mang nhãn LLM, có thể chứa sai lệch của LLM.
- Độ trễ trong trình duyệt (WASM, vector 128 bit) còn cao hơn CPU gốc nhiều lần.

## Hướng phát triển

- Bổ sung các kịch bản chưa thấy (DD6, LD12, LD7, QR6, chuyển nhầm tiền) cùng tin nhắn thật đã ẩn danh và đoạn đối chứng an toàn tương ứng (OTP hợp lệ, thông báo giao hàng, lời khuyên báo thầy cô).
- Gán tay một tập validation độc lập để dò ngưỡng riêng cho từng nhãn nguy hại.
- Tăng cường riêng cho các kiểu chèn dấu cách, ký tự vô hình, teencode để giảm báo nhầm.
- Thử WebGPU cho trình duyệt và BamiBERT làm phương án dự phòng cho điện thoại cấu hình thấp (p50 18,6 ms, trọng số 397 MiB).
