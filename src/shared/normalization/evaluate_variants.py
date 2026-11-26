"""
Kiểm tra hiệu năng của bộ chuẩn hóa (Normalizer) trên 100 mẫu biến thể.

Quy trình:
1. Đọc ngẫu nhiên 100 mẫu từ file v1/data/processed/vietnamese_nonstandard_v1_2/train.jsonl
2. Lấy original_text và text (đã bị làm nhiễu).
3. Chạy text nhiễu qua bộ chuẩn hóa (normalizer.py).
4. In kết quả ra màn hình (Original -> Variant -> Normalized).

Chạy:
    python -m src.shared.normalization.evaluate_variants
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

if __package__:
    from .normalizer import normalize
else:
    _project_root = Path(__file__).resolve().parents[3]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    try:
        from src.shared.normalization.normalizer import normalize
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "src",
            "src.shared",
            "src.shared.normalization",
            "src.shared.normalization.normalizer",
        }:
            raise
        _spec = importlib.util.spec_from_file_location(
            "_safevichi_normalizer",
            Path(__file__).resolve().parent / "normalizer.py",
        )
        if _spec is None or _spec.loader is None:
            raise
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        normalize = _mod.normalize

project_root = Path(__file__).resolve().parents[3]
JSONL_PATH = project_root / "v1" / "data" / "processed" / "vietnamese_nonstandard_v1_2" / "train.jsonl"


def _ensure_utf8_stdout() -> None:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("_", "-")
    if enc not in {"utf-8", "utf8"} and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _ensure_utf8_stdout()
    print("=" * 100)
    print("KIỂM TRA HIỆU NĂNG NORMALIZER TRÊN 100 MẪU (RULE-BASED)")
    print("=" * 100)
    print("Cấu trúc:")
    print("  [O] Văn bản gốc (original_text)")
    print("  [V] Biến thể (text)")
    print("  [N] Chuẩn hóa (sau khi qua normalizer)")
    print("-" * 100)

    if not JSONL_PATH.exists():
        print(f"Lỗi: Không tìm thấy file {JSONL_PATH}")
        return

    # Đọc tất cả các dòng hợp lệ vào list (bỏ qua dòng trắng)
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    # Lấy ngẫu nhiên 100 mẫu (không làm thay đổi trạng thái random toàn cục)
    rng = random.Random(42)
    sampled_lines = rng.sample(lines, min(100, len(lines)))

    count = 0

    for line in sampled_lines:
        data = json.loads(line)
        original_text = data.get("original_text") or ""
        variant_text = data.get("text") or ""
        applied_types = data.get("perturbation_types") or []

        normalized_text = normalize(variant_text)
        count += 1
        label = data.get("label") or "UNKNOWN"

        print(f"Mẫu #{count:03d} | Nhãn: {label} | Nhiễu: {', '.join(map(str, applied_types))}")
        if original_text:
            print(f"  [O] {original_text}")
        print(f"  [V] {variant_text}")
        print(f"  [N] {normalized_text}")
        print("-" * 100)

if __name__ == "__main__":
    main()
