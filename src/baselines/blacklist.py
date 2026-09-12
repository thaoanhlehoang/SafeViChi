"""B1 — Blacklist Filter (floor baseline).

Không train gì cả: xây 1 danh sách từ khóa từ chính tần suất xuất hiện của
chúng trong HATE so với CLEAN (trên data/data_final/train.jsonl —
câu SẠCH, chưa perturbation, để danh sách từ khóa không lẫn biến thể né lọc),
rồi phân loại bằng cách match trực tiếp: câu chứa >= 1 từ trong danh sách =>
HATE, ngược lại => CLEAN.

Vì đây là baseline "sàn" (floor), nó CỐ TÌNH không xử lý né lọc gì cả — điểm
yếu này là chủ đích: B1 dùng để đo B4 (normalize + adversarial) hơn được bao
nhiêu so với cách làm ngây thơ nhất.

Build danh sách:
    python -m src.baselines.blacklist build
Eval trên 1 file jsonl (mặc định C1_perturbed.jsonl — câu bị né lọc, RAW):
    python -m src.baselines.blacklist eval --data_path data/eval_baseline/test/C1_perturbed.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from src.classifier.metrics import score_labels

DEFAULT_WORDLIST_PATH = Path(__file__).parent / "blacklist_words.json"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def build_wordlist(
    train_path: Path,
    min_count: int = 5,
    min_hate_ratio: float = 0.75,
    max_words: int = 400,
) -> list[str]:
    hate_count: Counter[str] = Counter()
    clean_count: Counter[str] = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            counter = hate_count if d["label"] == "HATE" else clean_count
            counter.update(set(tokenize(d["text"])))

    candidates = []
    for word, h in hate_count.items():
        c = clean_count.get(word, 0)
        total = h + c
        if total < min_count:
            continue
        ratio = h / total
        if ratio >= min_hate_ratio:
            candidates.append((ratio, h, word))

    candidates.sort(reverse=True)
    return [word for _, _, word in candidates[:max_words]]


def predict(text: str, wordlist: set[str]) -> str:
    tokens = set(tokenize(text))
    return "HATE" if tokens & wordlist else "CLEAN"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B1: Blacklist Filter baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="xây danh sách từ khóa từ data_final/train.jsonl")
    p_build.add_argument("--train_path", type=str, default="data/data_final/train.jsonl")
    p_build.add_argument("--out_path", type=str, default=str(DEFAULT_WORDLIST_PATH))
    p_build.add_argument("--min_count", type=int, default=5)
    p_build.add_argument("--min_hate_ratio", type=float, default=0.75)
    p_build.add_argument("--max_words", type=int, default=400)

    p_eval = sub.add_parser("eval", help="eval danh sách từ khóa trên 1 file jsonl")
    p_eval.add_argument("--data_path", type=str, default="data/eval_baseline/test/C1_perturbed.jsonl")
    p_eval.add_argument("--wordlist_path", type=str, default=str(DEFAULT_WORDLIST_PATH))

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.command == "build":
        wordlist = build_wordlist(Path(args.train_path), args.min_count, args.min_hate_ratio, args.max_words)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(wordlist, f, ensure_ascii=False, indent=2)
        print(f"Đã xây {len(wordlist)} từ khóa -> {args.out_path}")
        print("Top 30:", wordlist[:30])

    elif args.command == "eval":
        wordlist = set(json.load(open(args.wordlist_path, encoding="utf-8")))
        print(f"Danh sách: {len(wordlist)} từ khóa ({args.wordlist_path})")

        rows = [json.loads(l) for l in open(args.data_path, encoding="utf-8") if l.strip()]
        print(f"Eval trên {args.data_path}: {len(rows)} dòng")

        from src.classifier.metrics import to_label_id
        label_ids = np.array([to_label_id(r["label"]) for r in rows])
        pred_ids = np.array([to_label_id(predict(r["text"], wordlist)) for r in rows])
        sources = [r.get("source", "unknown") for r in rows]

        metrics = score_labels(label_ids, pred_ids, sources)
        from src.classifier.metrics import report
        report(f"B1 blacklist trên {args.data_path}", metrics)


if __name__ == "__main__":
    main()
