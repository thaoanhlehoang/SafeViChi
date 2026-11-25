# SafeViChi v2 – web chạy thử (on-device)

Khung web mới cho v2: 5 nhãn, tô bằng chứng, mô hình chạy trong Web Worker trên máy người dùng.
Hiện dùng **mô hình giả** (`src/engine/mock.js`, dò từ khóa) để thiết kế UI song song khi chưa có ONNX.

```bash
cd web-v2
npm install
npm run dev      # http://127.0.0.1:5173
```

## Cấu trúc

- `src/main.js`, `index.html`, `src/style.css`: giao diện (thay tự do).
- `src/labels.js`: tên, màu, hành động của 5 nhãn.
- `src/model.worker.js`: hợp đồng tin nhắn UI <-> mô hình. **Giữ nguyên hợp đồng này** khi thay mô hình thật.
- `src/engine/mock.js`: mô hình giả. Sẽ thêm `src/engine/onnx.js` (onnxruntime-web + tokenizer mmBERT
  + bộ chuẩn hóa + giải mã bằng chứng theo `calibration.json`).

## Việc còn lại

1. Xuất mmBERT-DualORC-CRF sang ONNX (cần trọng số + file code định nghĩa mô hình), lượng tử hóa INT8.
2. Viết lại bộ chuẩn hóa và giải mã bằng chứng (ngưỡng marginal 0,40, T = 1,1807...) bằng JS, so khớp với Python.
3. Deploy: web lên Cloudflare Pages, file ONNX lên Cloudflare R2 (sửa `MODEL_ORIGIN_CHUA_DAT` trong `public/_headers`).

## Bảo mật

- Không backend, không lưu, không gửi tin nhắn. Sau khi tải mô hình, kiểm tra trong tab Network sẽ không thấy request nào.
- CSP chặt trong `public/_headers`, chỉ cho tải từ chính site và nơi để mô hình.
- Chữ người dùng chỉ đưa vào DOM bằng `textContent` (chống XSS).
- Không có khóa API nào trong mã web.
