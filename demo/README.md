# SafeViChi Web Demo (On-Device)

Ứng dụng web minh họa khả năng phát hiện nội dung độc hại tiếng Việt và khoanh vùng giải thích chạy trực tiếp trong trình duyệt bằng **ONNX Runtime Web**.

Hệ thống hoạt động 100% ở phía client:
- Không cần backend Python
- Không cần database
- Không gửi bất kỳ request mạng nào ra ngoài khi phân loại

---

## Cách chạy thử

1. Di chuyển vào thư mục demo:
   ```bash
   cd demo
   ```

2. Khởi động một web server nội bộ (ví dụ bằng Python):
   ```bash
   python -m http.server 8000
   ```

3. Mở trình duyệt tại địa chỉ:
   ```
   http://localhost:8000
   ```

---

## Kiến trúc demo

- **Giao diện & Điều khiển:** `index.html` và `js/ai_scanner.js`
- **Bộ chuẩn hóa phía trình duyệt:** `js/normalizer.js` (chuyển thể từ `src/normalization/normalizer.py`)
- **Từ điển teencode:** `teencode_dict.json`
- **Mô hình ONNX:** Nằm trong `onnx_model/`, gồm encoder và decoder của ViHateT5 đã tối ưu cho suy luận trên trình duyệt.

---

## Kịch bản kiểm thử on-device

1. Nhập một câu tiếng Việt có sử dụng teencode, chèn ký tự phân cách hoặc bỏ dấu (ví dụ: `m co bị n.gu l k z ma`).
2. Nhấn **Kiểm tra**.
3. Xem kết quả:
   - Văn bản sau khi chuẩn hóa.
   - Nhãn dự đoán (Độc hại / Sạch) kèm độ tin cậy.
   - Các từ ngữ vi phạm được tô sáng theo vị trí.
4. Mở tab **Network** trong Developer Tools (F12) của trình duyệt để kiểm chứng: Sau khi tải xong model ban đầu, không có bất kỳ gói tin HTTP/WebSocket nào được gửi ra ngoài khi bạn bấm kiểm tra câu mới.
