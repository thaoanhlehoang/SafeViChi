# Báo cáo thực nghiệm & Phân tích số liệu SafeViChi

Tài liệu này tổng hợp kết quả đánh giá thực nghiệm của SafeViChi trên hai bài toán: phân loại nội dung độc hại chống né bộ lọc (Bước 3) và khoanh vùng từ ngữ vi phạm để giải thích (Bước 4).

---

## 1. Phân loại nội dung độc hại (Bước 3)

### 1.1 Thiết lập thực nghiệm

Đánh giá được thực hiện trên ma trận 3 điều kiện tạo từ cùng một tập câu test (ghép cặp theo từng dòng dữ liệu):
- **C0 (Câu sạch - Clean):** Văn bản gốc chưa bị biến dạng, đại diện cho trần hiệu năng trong điều kiện bình thường.
- **C1 (Né lọc - Perturbed):** Văn bản đã bị biến dạng bởi 12 hiện tượng phi chuẩn (teencode, bỏ dấu, ký tự phân tách, homoglyph, lỗi gõ...).
- **C2 (Điểm mù - Blindspot):** Các biến thể được sinh để vượt qua lớp chuẩn hóa (thỏa mãn `normalize(x) == x`), kiểm tra khả năng ứng phó khi bộ chuẩn hóa bất lực.

Để đảm bảo tính khách quan và tránh rò rỉ dữ liệu:
- Toàn bộ ngưỡng phân loại được dò quét trên tập validation, sau đó **đóng băng** rồi mới áp vào tính điểm trên tập test.
- Mỗi hệ thống sử dụng duy nhất một ngưỡng cố định cho cả 3 điều kiện.

### 1.2 Bảng kết quả tổng hợp

Chỉ số chính: **F1 của lớp HATE** trên tập test (2.376 mẫu C0/C1, 1.920 mẫu C2):

| STT | Hệ thống | C0 (câu sạch) | C1 (né lọc) | C2 (điểm mù) |
|:---:|---|:---:|:---:|:---:|
| 1 | Blacklist từ khóa | 0.6551 | 0.5982 | 0.4564 |
| 2 | ViHateT5 gốc | 0.7683 | 0.6046 | 0.4627 |
| 3 | ViHateT5 gốc + Chuẩn hóa | 0.7497 | 0.7218 | 0.4720 |
| 4 | ViHateT5 fine-tune đối kháng (không chuẩn hóa) | 0.8523 | 0.7441 | 0.6657 |
| **5** | **SafeViChi đầy đủ (Chuẩn hóa + Fine-tune đối kháng)** | **0.8552** | **0.8266** | **0.6657** |

*(Ghi chú: Điểm số đã áp dụng tiền xử lý hạ chữ thường đầu vào cho tokenizer ViHateT5).*

### 1.3 Phân tích & Kiểm định ý nghĩa thống kê

Bốn kết luận rút ra từ bảng số liệu, được xác nhận qua kiểm định McNemar ghép cặp với mức ý nghĩa `p < 1e-6`:

1. **Văn bản né lọc làm sụp giảm hiệu năng mô hình gốc:**
   Khi chuyển từ C0 sang C1, ViHateT5 gốc bị tụt mạnh `0.1638` F1 (từ 0.7683 xuống 0.6046). Kỹ thuật né lọc thực sự là điểm yếu chí mạng của mô hình ngôn ngữ nếu không có cơ chế bảo vệ.
2. **Lớp chuẩn hóa (Bước 2) phục hồi đáng kể hiệu năng:**
   Khi bổ sung normalizer vào ViHateT5 gốc, F1 trên C1 tăng thêm `+0.1172` (`p = 7.47e-30`), đồng thời không làm tổn hại hiệu năng trên câu vốn đã sạch (C0: mức giảm nhỏ `0.0052`, không có ý nghĩa thống kê với `p = 0.550`).
3. **Huấn luyện đối kháng (Bước 3) tạo ra bước nhảy vọt:**
   Mô hình được huấn luyện đối kháng (dữ liệu trộn 60/30/10 kết hợp FGM) tăng thêm `+0.1048` F1 trên câu né lọc C1, đặc biệt tăng `+0.1937` F1 trên tập điểm mù C2 (`p = 1.21e-26`) - nơi bộ chuẩn hóa hoàn toàn không thể can thiệp.
4. **Chuẩn hóa và Đối kháng mang tính bổ trợ, không thay thế nhau:**
   Mô hình đã huấn luyện đối kháng vẫn cần thêm lớp chuẩn hóa để đạt hiệu năng cao nhất trên C1 (tăng thêm `+0.0825` F1, `p = 2.85e-08`). Hai cơ chế này phối hợp chặt chẽ: Normalizer xử lý nhanh các mẫu biến thể theo luật rõ ràng, còn mô hình đối kháng học cách thích ứng với các biến dạng ngẫu nhiên và tinh vi hơn.

---

## 2. Module giải thích khoanh vùng bằng Occlusion (Bước 4)

### 2.1 Thiết lập đánh giá

Module giải thích được đánh giá độc lập trên tập test của **ViHOS** gồm 531 câu HATE có nhãn khoanh vùng từ ngữ độc hại do con người gán (gold span).

Thuật toán che từng cửa sổ từ (occlusion với window = 2) và đo mức giảm logit margin của lớp HATE. Từ ngữ có mức giảm vượt ngưỡng sẽ được đánh dấu là vùng gây cảnh báo. Ngưỡng margin cũng được quét trên validation của ViHOS rồi đóng băng trước khi tính điểm trên test.

### 2.2 Kết quả định lượng

| Chỉ số | Giá trị |
|---|:---:|
| **Micro F1 (tính trên từng từ)** | **0.5455** |
| Precision ở mức từ | 0.4478 |
| Recall ở mức từ | 0.6977 |
| **Macro F1 (trung bình theo từng câu)** | **0.5766** |
| **Tỷ lệ trượt hoàn toàn (không trúng từ nào)** | **1.9%** |
| Tỷ lệ câu đạt F1 ≥ 0.5 | 67.2% |

### 2.3 Nhận xét

1. **Định vị đúng trọng tâm:**
   Chỉ 1.9% số câu bị trượt hoàn toàn. Điều này chứng minh mô hình hầu như luôn tập trung đúng khu vực chứa từ ngữ vi phạm, sai lệch chủ yếu nằm ở ranh giới biên của cụm từ.
2. **Recall cao hơn Precision:**
   Vùng được thuật toán khoanh trung bình rộng gấp 1.56 lần so với vùng do người gán nhãn khoanh. Đây là đặc thù của phương pháp che cửa sổ n-gram: các từ đệm nằm liền kề từ độc hại cũng chịu ảnh hưởng sụt giảm điểm khi che cùng cửa sổ, dẫn đến xu hướng khoanh bao quát hơn một chút thay vì bỏ sót.

---

## 3. Các bài học kỹ thuật quan trọng

### 3.1 Tokenizer SentencePiece không có ký tự hoa
Từ điển của mô hình nền ViHateT5 chỉ chứa các token chữ thường. Khi gặp ký tự viết hoa, tokenizer tự động chuyển thành `<unk>`. Điều này khiến các câu viết hoa toàn bộ (thường gặp trong các phát ngôn quá khích) bị mất ngữ nghĩa đối với mô hình.
- *Giải pháp:* Hạ chữ thường toàn bộ văn bản đầu vào trước khi đưa vào tokenizer (`t.lower()`). Text hiển thị cho người dùng vẫn giữ nguyên định dạng ban đầu. Nhờ sửa lỗi này, F1 của tất cả các hệ thống T5 đều tăng từ 2.8 đến 3.3 điểm phần trăm.

### 3.2 Dùng Logit Margin thay cho xác suất Softmax trong Occlusion
Khi câu văn mang tính độc hại rõ ràng, mô hình dự đoán rất tự tin (logit chênh lệch từ 16 đến 30). Khi tính qua hàm sigmoid/softmax ở độ chính xác float32, xác suất sẽ bị bão hòa về đúng `1.000000`. Nếu so sánh sự thay đổi xác suất khi che từ, tín hiệu chênh lệch sẽ hoàn toàn biến mất (`1.0 - 1.0 = 0.0`).
- *Giải pháp:* Điểm dùng để tính mức suy giảm trong thuật toán occlusion là `margin = hate_logit - clean_logit`. Vì margin là số thực không bị chặn trong đoạn [0, 1] nên giữ nguyên được tín hiệu chênh lệch một cách rõ rệt.

### 3.3 Kỷ luật dò ngưỡng độc lập
Trong thực tế triển khai, hệ thống không thể biết trước một câu đang đến là câu bình thường, câu né lọc hay câu thuộc điểm mù. Do đó, việc dò một ngưỡng riêng cho từng loại tấn công là rò rỉ thông tin phi thực tế. SafeViChi chọn một ngưỡng chung duy nhất cho toàn bộ hệ thống bằng cách tối ưu trên tập validation tổng hợp, sau đó đóng băng tuyệt đối trước khi chấm điểm trên tập test.
