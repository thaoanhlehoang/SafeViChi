"""
Tạo dataset đã chuẩn hóa từ dữ liệu biến thể.

Đọc train.jsonl (và validation/test), lấy cột text (biến thể) + label,
chạy qua bộ normalize, xuất ra file JSONL mới với 2 cột: text, label.

Chạy:
    python -m src.shared.normalization.build_normalized_dataset
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
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
INPUT_DIR = project_root / "v1" / "data" / "processed" / "vietnamese_nonstandard_v1_2"
OUTPUT_DIR = project_root / "v1" / "data" / "processed" / "normalized"


def _ensure_utf8_stdout() -> None:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("_", "-")
    if enc not in {"utf-8", "utf8"} and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def process_split(split_name: str) -> None:
    """Đọc 1 split (train/validation/test), chuẩn hóa, ghi ra file mới."""
    _ensure_utf8_stdout()
    input_path = INPUT_DIR / f"{split_name}.jsonl"
    if not input_path.exists():
        print(f"Khong tim thay {input_path}, bo qua.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    print(f"\n{'='*60}")
    print(f"Đang xử lý: {split_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    start_time = time.perf_counter()
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            variant_text = data["text"]
            label = data["label"]

            normalized_text = normalize(variant_text)

            row = {
                "text": normalized_text,
                "label": label,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count % 10000 == 0:
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                speed = count / elapsed
                print(f"  ... {count:,} dòng ({speed:.0f} dòng/s)")

    elapsed = time.perf_counter() - start_time
    print(f"  Hoan thanh {split_name}: {count:,} dong trong {elapsed:.1f}s")


def main():
    _ensure_utf8_stdout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TAO DATASET CHUAN HOA TU DU LIEU BIEN THE")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    for split in ["train", "validation", "test"]:
        process_split(split)

    print(f"\n{'='*60}")
    print("Hoan thanh tat ca!")
    print(f"Du lieu chuan hoa nam o: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
