"""Chạy thử bản build trên Windows: python serve.py rồi mở http://localhost:8000

Dùng file này thay cho `python -m http.server` vì Windows đôi khi gán sai kiểu file .js
(text/plain), làm trình duyệt chặn script và trang không có hình.
"""
import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).parent / "dist"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".svg": "image/svg+xml",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=str(ROOT))
    print("Mở http://localhost:8000  (Ctrl+C để dừng)")
    http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()
