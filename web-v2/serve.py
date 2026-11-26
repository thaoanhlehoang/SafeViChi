"""Chạy thử bản build trên Windows: python serve.py rồi mở http://localhost:8000

- Gắn đúng MIME type cho .js/.mjs/.wasm trên Windows.
- Bật COOP/COEP để trình duyệt cấp SharedArrayBuffer cho ONNX Runtime Web đa luồng.
- Phục vụ trực tiếp gói ONNX cục bộ tại `/model/*` từ `../models/dualorc_mmbert_base/onnx/`
  (hoặc biến môi trường `SAFEVICHI_ONNX_DIR`) nếu đã có trên máy.
"""
import functools
import http.server
import os
import posixpath
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "dist"
ONNX_ROOT = Path(
    os.environ.get("SAFEVICHI_ONNX_DIR", "").strip()
    or (HERE.parent / "models" / "dualorc_mmbert_base" / "onnx")
).resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".svg": "image/svg+xml",
    }

    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlsplit(path).path
        if parsed.startswith("/model/"):
            rel = posixpath.normpath(urllib.parse.unquote(parsed[len("/model/") :])).lstrip("/")
            if rel and not rel.startswith(".."):
                candidate = (ONNX_ROOT / rel).resolve()
                try:
                    candidate.relative_to(ONNX_ROOT)
                    return str(candidate)
                except ValueError:
                    pass
            return str(ROOT / "__not_found__")
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()


if __name__ == "__main__":
    if not ROOT.is_dir():
        raise SystemExit("Chưa có thư mục dist/. Hãy chạy `npm run build` trong web-v2 trước.")
    handler = functools.partial(Handler, directory=str(ROOT))
    print(f"Phục vụ dist/ từ {ROOT}")
    print(f"Nguồn mô hình cục bộ (/model): {ONNX_ROOT} ({'có sẵn' if (ONNX_ROOT / 'crf_params.json').is_file() else 'chưa có — trình duyệt sẽ tự tải từ Hugging Face'})")
    print("Mở http://localhost:8000  (Ctrl+C để dừng)")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()
