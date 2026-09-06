"""Build bộ dữ liệu ViHOS cho Bước 4 (occlusion localization eval).

Tải qua HF datasets: `load_dataset("htdung167/ViHOS")` — 3 split sẵn
train(8844)/validation(1106)/test(1106), cột:
  - `content`: câu text gốc (KHÔNG bị biến thể/normalize, câu sạch tiếng Việt).
  - `index_spans`: chuỗi kiểu `"[9, 10]"` — chỉ số KÝ TỰ (không phải mức từ)
    trong `content` thuộc về vùng gây hại. `"[]"` = không có span (CLEAN).

Occlusion (Bước 4) trượt theo TỪ (whitespace-tokenized), nên script này map
span ký tự -> span từ: 1 từ được coi là gold nếu có ít nhất 1 ký tự của nó
nằm trong `index_spans`. Output mỗi dòng:
  {"words": [...], "span_mask": [0/1,...], "label": "HATE"|"CLEAN"}
(span_mask cùng độ dài words).

Chạy: python -m src.dataset_builder.build_vihos
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"\S+")


def char_spans_to_word_mask(content: str, char_indices: set[int]) -> tuple[list[str], list[int]]:
    """Tokenize theo whitespace, gán span_mask[i]=1 nếu từ i chứa >=1 index gold."""
    words: list[str] = []
    span_mask: list[int] = []
    for match in WORD_RE.finditer(content):
        words.append(match.group())
        word_chars = range(match.start(), match.end())
        span_mask.append(1 if any(i in char_indices for i in word_chars) else 0)
    return words, span_mask


def build_split(dataset) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    malformed = 0
    for row in dataset:
        content = row["content"]
        try:
            char_indices = set(ast.literal_eval(row["index_spans"]))
        except (ValueError, SyntaxError):
            malformed += 1
            char_indices = set()

        words, span_mask = char_spans_to_word_mask(content, char_indices)
        label = "HATE" if char_indices else "CLEAN"
        rows.append({"words": words, "span_mask": span_mask, "label": label})

    stats = {
        "total": len(rows),
        "malformed_index_spans": malformed,
        "by_label": dict(Counter(r["label"] for r in rows)),
        "empty_words": sum(1 for r in rows if not r["words"]),
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bộ dữ liệu ViHOS (word-level span_mask) cho Bước 4")
    parser.add_argument("--out_dir", type=str, default="data/data_explain")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = parser.parse_args()

    from datasets import load_dataset

    print("Tải htdung167/ViHOS ...")
    dataset = load_dataset("htdung167/ViHOS")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"source": "htdung167/ViHOS", "splits": {}}
    for split in args.splits:
        rows, stats = build_split(dataset[split])
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = stats
        print(f"{split:11s} total={stats['total']:5d} labels={stats['by_label']} "
              f"malformed={stats['malformed_index_spans']}")

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {out_dir}/ ({', '.join(args.splits)} + manifest.json)")


if __name__ == "__main__":
    main()
