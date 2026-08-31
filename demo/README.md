# Demo web on-device

Tương ứng Mục 5 trong `docs/plan.md`. Mô hình sau khi fine-tune được lượng
tử hóa (quantize) và export sang ONNX, chạy trực tiếp trong trình duyệt
bằng **ONNX Runtime Web** — không backend, không database, không request
mạng nào được gửi đi khi phân loại.

## Việc cần làm để hoàn thiện demo

1. Fine-tune model (`src/classifier/train.py`) → checkpoint PyTorch trong `models/`
2. Export sang ONNX + quantize:
   ```bash
   optimum-cli export onnx --model models/visobert-classifier --task text-classification demo/model_onnx/
   ```
3. Copy tokenizer files (`tokenizer.json`, `sentencepiece.bpe.model`, `config.json`) vào `demo/model_onnx/`
4. Load model trong `index.html` bằng thư viện `onnxruntime-web` (CDN hoặc npm bundle)
5. Chạy lớp chuẩn hóa (port logic từ `src/normalization/normalizer.py` sang JavaScript, hoặc dùng WASM/Pyodide nếu muốn tái sử dụng code Python trực tiếp)

## Kịch bản demo 5 bước (Mục 5.2 trong plan)

1. Người dùng gõ trực tiếp câu tiếng Việt né lọc vào ô nhập liệu
2. Hiển thị real-time văn bản sau khi qua lớp chuẩn hóa
3. Hiển thị nhãn rủi ro + cụm từ được tô sáng (module giải thích)
4. Mở tab Network của DevTools — chứng minh không có request nào được gửi đi
5. (Tuỳ chọn) mock màn hình "phía phụ huynh" — chỉ hiện nhãn + cụm từ, không hiện nguyên văn

## Chạy thử local

```bash
cd demo
python -m http.server 8000
# mở http://localhost:8000
```
