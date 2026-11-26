"""Điểm vào mô-đun cho phép chạy `python -m src.v2` mà không cần cài gói."""

from src.v2.predict import main

if __name__ == "__main__":
    raise SystemExit(main())
