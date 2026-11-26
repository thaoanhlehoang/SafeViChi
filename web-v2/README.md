# SafeViChi v2 – Ứng dụng Web suy luận ONNX trên thiết bị (On-Device)

Giao diện trò chuyện thử nghiệm SafeViChi v2 chạy mô hình thật **`SafeVi-DualORC-CRF` (`mmBERT-base` Hybrid INT8)** ngay trong Web Worker trên trình duyệt qua **ONNX Runtime Web (WASM SIMD)**, suy luận trực tiếp trên văn bản gốc (không dùng bộ chuẩn hóa v1).

## Khởi động nhanh

```bash
cd web-v2
npm install
npm run dev      # http://127.0.0.1:5173
```

Hoặc chạy bản đã build:

```bash
cd web-v2
npm run build
python serve.py  # http://localhost:8000
```

### Nguồn mô hình ONNX

1. **Chạy cục bộ (mặc định):** `vite.config.js` (dev/preview) và `serve.py` phục vụ trực tiếp đường dẫn `/model/*` từ `../models/dualorc_mmbert_base/onnx/` (hoặc biến môi trường `SAFEVICHI_ONNX_DIR`), không cần sao chép ~334 MiB vào `public/`.
2. **Triển khai tĩnh (Cloudflare Pages) hoặc khi máy chưa tải sẵn `models/`:** trình duyệt tự động tải từ Hugging Face Hub (`https://huggingface.co/dminhnhat06/safevichi_v2/resolve/main/onnx`) và lưu vào **Cache API** để các lần sau chạy ngoại tuyến.
3. **Nguồn tùy chỉnh (Cloudflare R2 / CDN):** đặt `VITE_MODEL_BASE_URL` trong `.env` (xem `.env.example`) trước khi `npm run build` — `dist/_headers` sẽ tự động thêm origin này vào `connect-src`.

## Cấu trúc mã nguồn

- `src/engine/tokenizer.js`: Tokenizer BPE + Metaspace + ByteFallback cho `mmBERT-base`, đọc thẳng `tokenizer.json` và giữ offsets UTF-16 chính xác.
- `src/engine/dualorc.js`: Tiền xử lý hội thoại 1–8 lượt (8 tensor đầu vào) + giải mã BIO-CRF Forward-Backward / Viterbi khớp với `src/v2/predict.py`.
- `src/engine/onnx.js`: Nạp `safevichi_v2_int8.onnx` trên ONNX Runtime Web (WASM SIMD, tự bật đa luồng khi có COOP/COEP), báo tiến độ theo byte và hỗ trợ Cache API.
- `src/model.worker.js`: Web Worker tuần tự hóa hàng đợi suy luận (ưu tiên tương tác người dùng).
- `src/main.js`, `src/chat.js`, `src/landing.js`, `src/labels.js`: Giao diện 2 màn (Trang chủ → Trò chuyện 3 cột), gộp tin liên tiếp cùng người nói theo đúng quy ước lượt thoại của mô hình v2.
- `scripts/verify_onnx_parity.mjs`: Kiểm thử đối chiếu trực tiếp JS (`onnxruntime-web` WASM + `MMBertTokenizer` + `dualorc.js`) với tập test v2 và kết quả khóa.

## Kiểm thử đối chiếu mô hình

```bash
cd web-v2
npm run verify:parity
```

## Bảo mật & Quyền riêng tư

- Không backend, không lưu, không gửi nội dung tin nhắn ra ngoài. Sau khi tải xong mô hình, mở F12 → Network khi gửi tin nhắn sẽ không phát sinh request nào.
- Chữ người dùng chỉ đưa vào DOM qua `textContent` / text node (chống XSS).
- Header bảo mật chặt trong `public/_headers` (và sinh tự động vào `dist/_headers`): CSP, COOP/COEP, CORP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`.
