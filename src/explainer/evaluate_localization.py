"""Bước 4.3 — đo độ chính xác ĐỊNH VỊ của occlusion trên ViHOS.

Câu hỏi: khi model kêu 1 câu là HATE, cụm từ mà occlusion chỉ ra có trùng với
cụm từ con người đã khoanh (ViHOS gold span) không?

Cách làm:
  1. Đọc data/data_explain/{split}.jsonl (words / span_mask / label — do
     src/dataset_builder/build_vihos.py dựng).
  2. Chỉ chạy trên câu HATE có span gold KHÔNG rỗng — câu CLEAN không có gì
     để định vị, đưa vào chỉ làm loãng số liệu.
  3. Mỗi câu: occlude_words() -> điểm quan trọng từng từ -> chọn tập từ "gây
     cảnh báo" theo NGƯỠNG -> so với span_mask gold ở mức TỪ.
  4. Báo cáo micro (gộp TP/FP/FN toàn tập) + macro (trung bình F1 theo câu).

NGƯỠNG được quét trên VALIDATION rồi ĐÓNG BĂNG áp lên TEST — đúng kỷ luật đã
dùng ở Bước 3 (src/baselines/tune_threshold.py), để số liệu không bị "chọn
ngưỡng đẹp trên chính tập test". Điểm occlusion là logit margin (xem
occlusion.py) nên ngưỡng có đơn vị margin, KHÔNG phải xác suất — dải quét mặc
định 0..5 phản ánh điều đó.

Chạy (GPU khuyến nghị, ~531 câu HATE trong test):
    python -m src.explainer.evaluate_localization \
        --model_name models/vihatet5-v2-best \
        --val_path  data/data_explain/validation.jsonl \
        --test_path data/data_explain/test.jsonl \
        --out_dir results/step4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classifier.metrics import report
from src.explainer.occlusion import (
    OcclusionScorer, normalize_importance, occlude_words, top_k_spans,
)

# absolute: ngưỡng tính bằng đơn vị logit margin (không phải xác suất).
# relative: điểm đã chia cho từ mạnh nhất trong câu -> ngưỡng nằm trong 0..1,
#   nghĩa là "mạnh bằng ít nhất X% từ mạnh nhất của chính câu này".
THRESHOLD_GRIDS = {
    "absolute": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "relative": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}


def read_span_rows(path: str | Path) -> list[dict]:
    """Chỉ giữ câu HATE có span gold không rỗng (câu duy nhất có thể định vị)."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["label"] == "HATE" and any(row["span_mask"]) and row["words"]:
                rows.append(row)
    return rows


def compute_importances(rows: list[dict], scorer: OcclusionScorer, window: int,
                        scoring: str) -> list[list[float]]:
    """Chạy occlusion 1 lần cho mọi câu — tái dùng cho MỌI ngưỡng khi quét."""
    importances: list[list[float]] = []
    for i, row in enumerate(rows, 1):
        importance = occlude_words(row["words"], scorer, window=window)
        if scoring == "relative":
            importance = normalize_importance(importance)
        importances.append(importance)
        if i % 100 == 0:
            print(f"    ...{i}/{len(rows)} câu")
    return importances


def score_at_threshold(rows: list[dict], importances: list[list[float]], threshold: float) -> dict:
    """P/R/F1 mức từ: micro (gộp toàn tập) + macro (trung bình theo câu)."""
    tp_total = fp_total = fn_total = 0
    per_sentence_f1: list[float] = []

    for row, importance in zip(rows, importances):
        predicted = set(top_k_spans(importance, threshold=threshold))
        gold = {i for i, m in enumerate(row["span_mask"]) if m}

        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_sentence_f1.append(2 * p * r / (p + r) if p + r else 0.0)

    micro_p = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_r = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    return {
        "threshold": threshold,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": sum(per_sentence_f1) / len(per_sentence_f1) if per_sentence_f1 else 0.0,
        "n_sentences": len(rows),
        "predicted_words": tp_total + fp_total,
        "gold_words": tp_total + fn_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo độ chính xác định vị của occlusion trên ViHOS")
    parser.add_argument("--model_name", type=str, default="models/vihatet5-v2-best",
                         help="checkpoint đã fine-tune ở Bước 3 (hoặc HF hub id để smoke-test)")
    parser.add_argument("--val_path", type=str, default="data/data_explain/validation.jsonl",
                         help="dùng để QUÉT ngưỡng, không dùng để báo cáo số liệu chính")
    parser.add_argument("--test_path", type=str, default="data/data_explain/test.jsonl",
                         help="chỉ chạy ở ngưỡng ĐÃ đóng băng từ val")
    parser.add_argument("--out_dir", type=str, default="results/step4")
    parser.add_argument("--window", type=int, default=2, help="số từ bị che mỗi lần (n-gram)")
    parser.add_argument("--scoring", choices=list(THRESHOLD_GRIDS), default="relative",
                         help="relative (mặc định): chuẩn hóa điểm theo từ mạnh nhất TRONG TỪNG CÂU "
                              "rồi mới áp ngưỡng — thang margin lệch rất mạnh giữa các câu nên ngưỡng "
                              "tuyệt đối gạt sạch câu 'model nói nhỏ'. absolute: ngưỡng thẳng trên margin.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="chỉ lấy N câu đầu mỗi split (smoke-test)")
    args = parser.parse_args()

    scorer = OcclusionScorer(args.model_name, batch_size=args.batch_size)
    print(f"Model: {args.model_name} (device={scorer.device})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[VAL] quét ngưỡng trên {args.val_path}")
    val_rows = read_span_rows(args.val_path)[: args.limit]
    print(f"  {len(val_rows)} câu HATE có span gold")
    val_importances = compute_importances(val_rows, scorer, args.window, args.scoring)

    sweep = [score_at_threshold(val_rows, val_importances, t) for t in THRESHOLD_GRIDS[args.scoring]]
    for row in sweep:
        print(f"  threshold={row['threshold']:<5} micro_f1={row['micro_f1']:.4f} "
              f"P={row['micro_precision']:.4f} R={row['micro_recall']:.4f}")
    best = max(sweep, key=lambda r: r["micro_f1"])
    frozen_threshold = best["threshold"]
    print(f"  -> ngưỡng chọn (tốt nhất trên VAL): {frozen_threshold}")

    print(f"\n[TEST] áp ngưỡng đã đóng băng lên {args.test_path}")
    test_rows = read_span_rows(args.test_path)[: args.limit]
    print(f"  {len(test_rows)} câu HATE có span gold")
    test_importances = compute_importances(test_rows, scorer, args.window, args.scoring)
    test_metrics = score_at_threshold(test_rows, test_importances, frozen_threshold)

    report(f"Localization trên ViHOS test (ngưỡng {frozen_threshold} đóng băng từ val)", test_metrics)

    results = {
        "model": args.model_name,
        "window": args.window,
        "scoring": args.scoring,
        "frozen_threshold": frozen_threshold,
        "val_sweep": sweep,
        "val_best": best,
        "test": test_metrics,
    }
    with open(out_dir / "localization.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_dir / "predictions_test.jsonl", "w", encoding="utf-8") as f:
        for row, importance in zip(test_rows, test_importances):
            predicted = top_k_spans(importance, threshold=frozen_threshold)
            f.write(json.dumps({
                "words": row["words"],
                "span_mask": row["span_mask"],
                "importance": [round(v, 4) for v in importance],
                "predicted_indices": predicted,
                "predicted_words": [row["words"][i] for i in predicted],
                "gold_words": [w for w, m in zip(row["words"], row["span_mask"]) if m],
            }, ensure_ascii=False) + "\n")

    print(f"\nĐã ghi {out_dir}/localization.json + predictions_test.jsonl")


if __name__ == "__main__":
    main()
