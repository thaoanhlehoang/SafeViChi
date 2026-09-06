"""Quét ngưỡng quyết định trên tập VALIDATION, đóng băng, rồi áp lên TEST.

Vì sao cần bước này: tập train lệch prior nặng (~49% HATE) so với tập test
(~18% HATE). Model học "cứ 2 câu thì 1 câu độc hại" rồi bị đem chấm ở nơi chỉ
1/5 câu độc hại, nên ở ngưỡng mặc định 0.5 nó báo HATE quá tay và precision
sập. Ngưỡng 0.5 cũng mang ý nghĩa KHÁC NHAU với B2 và B3 vì hai model học dưới
hai prior khác nhau — so chúng ở cùng 0.5 là so ở hai điểm vận hành không
tương đương.

Cách làm đúng: mỗi hệ thống được chọn ngưỡng tối ưu F1 của RIÊNG nó trên tập
validation (công bằng cho cả B2 lẫn B3), đóng băng ngưỡng đó, rồi mới mở tập
test ra chấm. Ngưỡng KHÔNG BAO GIỜ được chọn trên tập test — làm vậy thì con
số báo cáo mất giá trị.

Chạy trên CPU, không cần GPU:
    python -m src.baselines.tune_threshold \
        --val_scores results/step3/val/scores.jsonl \
        --test_scores results/step3/test/scores.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.baselines.stats import bootstrap_ci
from src.utils.seed import SEED_DEFAULT, set_global_seed


def load_scores(path: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """-> {(system, condition): {"y": ..., "p": ..., "row_id": ...}}"""
    buckets: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"y": [], "p": [], "row_id": []})
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            b = buckets[(d["system"], d["condition"])]
            b["y"].append(d["label"])
            b["p"].append(d["p_hate"])
            b["row_id"].append(d["row_id"])
    return {k: {"y": np.array(v["y"]), "p": np.array(v["p"]), "row_id": v["row_id"]}
            for k, v in buckets.items()}


def sweep(y: np.ndarray, p: np.ndarray, n_steps: int = 200) -> tuple[float, float]:
    """Ngưỡng cho F1 lớp HATE cao nhất. Trả (ngưỡng, F1 tại đó)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, n_steps):
        f1 = f1_score(y, (p > t).astype(int), pos_label=1, average="binary", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def score_at(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p > t).astype(int)
    m = {
        "threshold": round(t, 4),
        "hate_f1": float(f1_score(y, pred, pos_label=1, average="binary", zero_division=0)),
        "hate_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "hate_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "predicted_hate_rate": float(pred.mean()),
    }
    m.update(bootstrap_ci(y, pred))
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Quét ngưỡng trên val, áp lên test")
    parser.add_argument("--val_scores", type=str, default="results/step3/val/scores.jsonl")
    parser.add_argument("--test_scores", type=str, default="results/step3/test/scores.jsonl")
    parser.add_argument("--out_dir", type=str, default="results/step3/threshold")
    parser.add_argument("--pool_conditions", action="store_true", default=True,
                        help="chọn MỘT ngưỡng / hệ thống trên toàn bộ điều kiện gộp lại "
                             "(mặc định). Sát thực tế triển khai: lúc chạy thật không "
                             "biết trước đang gặp kiểu tấn công nào để đổi ngưỡng.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    set_global_seed(args.seed)

    val = load_scores(Path(args.val_scores))
    test = load_scores(Path(args.test_scores))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = sorted({s for s, _ in val})
    conditions = sorted({c for _, c in test})

    # --- 1. Chọn ngưỡng trên VALIDATION ---
    print("=== Quét ngưỡng trên VALIDATION ===")
    chosen: dict[str, float] = {}
    for sysname in systems:
        ys, ps = [], []
        for cond in sorted({c for s, c in val if s == sysname}):
            b = val[(sysname, cond)]
            ys.append(b["y"])
            ps.append(b["p"])
        y_all, p_all = np.concatenate(ys), np.concatenate(ps)
        t, f1 = sweep(y_all, p_all)
        chosen[sysname] = t
        f1_at_half = f1_score(y_all, (p_all > 0.5).astype(int), pos_label=1,
                              average="binary", zero_division=0)
        print(f"  {sysname:20s} ngưỡng={t:.3f}  val F1={f1:.4f}  "
              f"(ở 0.5 thì F1={f1_at_half:.4f}, chênh {f1 - f1_at_half:+.4f})")

    # --- 2. Áp ngưỡng đã đóng băng lên TEST ---
    print("\n=== Áp ngưỡng đã đóng băng lên TEST ===")
    results: dict[str, dict] = {}
    for sysname in systems:
        results[sysname] = {}
        for cond in conditions:
            if (sysname, cond) not in test:
                continue
            b = test[(sysname, cond)]
            tuned = score_at(b["y"], b["p"], chosen[sysname])
            default = score_at(b["y"], b["p"], 0.5)
            results[sysname][cond] = {"tuned": tuned, "at_0.5": default}
            print(f"  {sysname:20s} {cond:15s} F1={tuned['hate_f1']:.4f} "
                  f"[{tuned['f1_ci_low']:.3f}-{tuned['f1_ci_high']:.3f}] "
                  f"P={tuned['hate_precision']:.3f} R={tuned['hate_recall']:.3f}  "
                  f"(ở 0.5: {default['hate_f1']:.4f}, chênh "
                  f"{tuned['hate_f1'] - default['hate_f1']:+.4f})")

    payload = {"chosen_thresholds": chosen, "results": results,
               "val_scores": args.val_scores, "test_scores": args.test_scores}
    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = ["| Hệ thống | Ngưỡng | Điều kiện | hate F1 | KTC 95% | P | R | macro F1 | F1 ở 0.5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sysname, per_cond in results.items():
        for cond, m in per_cond.items():
            t_, d_ = m["tuned"], m["at_0.5"]
            lines.append(
                f"| {sysname} | {t_['threshold']:.3f} | {cond} | **{t_['hate_f1']:.4f}** | "
                f"{t_['f1_ci_low']:.3f}–{t_['f1_ci_high']:.3f} | {t_['hate_precision']:.3f} | "
                f"{t_['hate_recall']:.3f} | {t_['macro_f1']:.4f} | {d_['hate_f1']:.4f} |")
    (out_dir / "threshold.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {out_dir / 'threshold.json'} và {out_dir / 'threshold.md'}")


if __name__ == "__main__":
    main()
