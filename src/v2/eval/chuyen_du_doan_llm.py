"""Đổi file dự đoán LLM API (`nhan_llm`, `bang_chung_llm`) sang định dạng bộ chấm đọc (`nhan`, `bang_chung`).

Không sửa nội dung dự đoán, chỉ giữ `id` và đổi tên hai trường.

    python -m src.v2.eval.chuyen_du_doan_llm \
        --src results/v2/llm_api/test_predictions.jsonl \
        --dst results/v2/llm_api/cham_lai/C0_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open("r", encoding="utf-8-sig") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            out = {"id": row["id"], "nhan": row.get("nhan_llm"), "bang_chung": row.get("bang_chung_llm") or []}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args(argv)
    n = convert(args.src, args.dst)
    print(f"Đã ghi {n} dòng vào {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
