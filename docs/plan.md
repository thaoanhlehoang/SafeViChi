# SafeViChi — Kế hoạch dự án (bản demo cuộc thi sáng tạo)

> Đây là bản tóm tắt markdown để tiện tra cứu trong repo. Bản đầy đủ (định dạng
> Word, có bảng, tài liệu tham khảo) nên được lưu song song, ví dụ tại
> `docs/Plan_Demo_CuocThi.docx`.

## Luận điểm cốt lõi

Mô hình phân loại nội dung độc hại tiếng Việt sụp giảm hiệu năng khi văn bản
bị biến dạng có chủ đích để né bộ lọc. Thêm một lớp chuẩn hóa và huấn luyện
đối kháng phục hồi được phần lớn hiệu năng đã mất — toàn bộ chạy được ngay
trên thiết bị, không gửi nội dung của trẻ lên bất kỳ máy chủ nào.

## Phạm vi CÓ / KHÔNG CÓ trong demo

| Hạng mục            | Trong phạm vi demo                       | Ngoài phạm vi (hướng mở rộng)                         |
| ------------------- | ---------------------------------------- | ----------------------------------------------------- |
| Loại nội dung       | Chỉ văn bản tiếng Việt                   | Hình ảnh, video, âm thanh                             |
| Kênh đầu vào        | Ô nhập liệu trên web app                 | Accessibility Service, Screen Time, tích hợp app thật |
| Lớp chuẩn hóa       | Rule-based + từ điển                     | Mô hình neural khôi phục văn bản (xem ViSoLex)        |
| Mô hình phân loại   | Một mô hình duy nhất: ViSoBERT fine-tune | Kiến trúc đa chuyên gia, router đa phương thức        |
| Bộ sinh biến thể    | 6 nhóm tham số hóa, rule-based           | Vòng lặp LLM red-teaming                              |
| Cập nhật mô hình    | Huấn luyện một lần                       | Federated learning                                    |
| Giao diện phụ huynh | Mock tĩnh                                | Ứng dụng đồng bộ đa thiết bị hoàn chỉnh               |

## Pipeline

```
Văn bản đầu vào
    → [1] Lớp chuẩn hóa (rule-based)          src/normalization/
    → [2] Bộ phân loại (ViSoBERT fine-tune)    src/classifier/
    → [3] Module giải thích (occlusion)        src/explainer/
    → Đầu ra: nhãn rủi ro + cụm từ gây cảnh báo
```

## Dữ liệu

- **ViHSD** (huggingface.co/datasets/uitnlp/vihsd): 33.400 bình luận, nhãn sạch/xúc phạm/thù ghét — nguồn train/eval chính
- **ViHOS** (huggingface.co/datasets/htdung167/ViHOS): nhãn mức cụm từ — chỉ dùng để đánh giá module giải thích, KHÔNG dùng để train
- **ViLexNorm** (arxiv.org/abs/2401.16403): 10.000+ cặp câu nhiễu/chuẩn — nguồn bootstrap từ điển teencode cho lớp chuẩn hóa

## Bậc thang baseline

| STT | Baseline                                              | Mục đích                            |
| --- | ----------------------------------------------------- | ----------------------------------- |
| 1   | Bộ lọc từ khóa (blacklist)                            | Mốc dưới cùng                       |
| 2   | XLM-R fine-tune trên ViHSD                            | Đại diện giải pháp quốc tế          |
| 3   | ViSoBERT fine-tune, không chuẩn hóa, không đối kháng  | Đo mức sụp hiệu năng trước biến thể |
| 4   | ViSoBERT fine-tune + chuẩn hóa + huấn luyện đối kháng | Hệ đầy đủ — kết quả đề xuất         |

Ghi chú: ViHateT5 (T5-based, arxiv.org/abs/2405.14141) đạt điểm cao hơn trên
benchmark công bố, nhưng kiến trúc encoder-decoder nặng hơn, khó lượng tử
hóa gọn cho demo on-device — nên chỉ trích dẫn làm mốc SOTA tham khảo, không
dùng làm baseline chính.

## 6 nhóm biến thể né lọc

| Nhóm                 | Mô tả                                            | Ví dụ (từ "mày ngu quá") |
| -------------------- | ------------------------------------------------ | ------------------------ |
| Loại bỏ dấu          | Bỏ dấu thanh, dấu phụ                            | may ngu qua              |
| Teencode             | Viết tắt phổ biến                                | m ngu qá                 |
| Chèn ký tự phân tách | Thêm dấu chấm/gạch/khoảng trắng                  | m.à.y ng.u qu.á          |
| Ký tự đồng dạng      | Thay chữ bằng số/ký hiệu giống hình              | m4y n9u qu4              |
| Unicode nhìn giống   | Thay bằng ký tự bảng mã khác, hiển thị giống hệt | "a" Latin → "а" Cyrillic |
| Nhiễu chính tả       | Đảo/lặp/bỏ ký tự                                 | mya ngu quaa             |

## Chỉ số đánh giá

- Precision, Recall, F1 theo lớp + trung bình có trọng số
- Tỷ lệ báo động giả
- Độ chính xác định vị cụm từ (đối chiếu ViHOS)
- Độ trễ suy luận + kích thước model sau quantize

## Kịch bản demo (5 bước)

1. Gõ trực tiếp câu né lọc vào ô nhập liệu
2. Hiển thị real-time văn bản sau chuẩn hóa
3. Hiển thị nhãn rủi ro + cụm từ tô sáng
4. Mở tab Network — chứng minh không gửi request nào
5. (Tuỳ chọn) mock màn hình phụ huynh — chỉ hiện nhãn, không hiện nguyên văn

## Hạn chế đã biết

- Báo động giả với ngôn ngữ đùa cợt giữa bạn bè
- Biến thể ngoài phân bố (né lọc thay đổi liên tục)
- Hệ thống có thể bị vô hiệu hóa (gỡ cài, đổi thiết bị)
- Kết quả chỉ phản ánh hiệu năng trên ViHSD/ViHOS và biến thể tự sinh

## Hướng mở rộng tương lai

- Nhánh xử lý ảnh + kiến trúc đa chuyên gia
- Lớp chuẩn hóa bằng mạng nơ-ron (xem ViSoLex — arxiv.org/abs/2501.07020)
- Vòng lặp LLM red-teaming sinh biến thể mới
- Federated learning
- Tích hợp thu thập nội dung thật (Accessibility Service / Screen Time)
- Hiệu chỉnh ngưỡng cảnh báo theo độ tuổi
- Ứng dụng phụ huynh hoàn chỉnh
