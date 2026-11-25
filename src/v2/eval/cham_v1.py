"""Chấm baseline SafeViChi v1 (nhãn nhị phân HATE/CLEAN) trên bộ test v2.

v1 không có 5 nhãn và không trả bằng chứng, nên chỉ chấm nhị phân nguy hại/an toàn
(HATE ứng với 4 nhãn nguy hại của v2, CLEAN ứng với AN_TOAN), kèm recall theo từng nhãn v2
và các chỉ số theo giọng vùng.

    python -m src.v2.eval.cham_v1 \
        --pred results/v2/v1_baseline/test_v1_predictions.jsonl \
        --meta data/v2/test_day_du.jsonl \
        --out results/v2/v1_baseline/cham_lai/C0_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAFE_LABEL = "AN_TOAN"
HARMFUL_LABELS = ("XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
REGIONS = ("bac", "trung", "nam")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_metrics(gold_harm: list[bool], pred_harm: list[bool]) -> dict[str, Any]:
    tp = sum(g and p for g, p in zip(gold_harm, pred_harm))
    fp = sum((not g) and p for g, p in zip(gold_harm, pred_harm))
    fn = sum(g and (not p) for g, p in zip(gold_harm, pred_harm))
    tn = sum((not g) and (not p) for g, p in zip(gold_harm, pred_harm))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(gold_harm),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr_an_toan": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "fnr_nguy_hai": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate(rows: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold = [str(r["nhan_goc"]) for r in rows]
    pred_harm = [str(r["nhan_v1"]).upper() == "HATE" for r in rows]
    gold_harm = [g != SAFE_LABEL for g in gold]
    regions = [meta.get(str(r.get("goc_id") or r["id"]), {}).get("giong_vung") for r in rows]

    recall_by_label = {}
    for lbl in HARMFUL_LABELS:
        idx = [i for i, g in enumerate(gold) if g == lbl]
        recall_by_label[lbl] = {
            "n": len(idx),
            "recall": round(sum(pred_harm[i] for i in idx) / len(idx), 4) if idx else 0.0,
        }

    by_region = {}
    for reg in REGIONS:
        idx = [i for i, v in enumerate(regions) if v == reg]
        by_region[reg] = binary_metrics([gold_harm[i] for i in idx], [pred_harm[i] for i in idx])
    f1s = [by_region[r]["f1"] for r in REGIONS if by_region[r]["n"]]

    return {
        "he_thong": "SafeViChi v1 (ViHateT5 FGM, nhị phân HATE/CLEAN)",
        "ghi_chu": "v1 không có 5 nhãn và bằng chứng: chỉ chấm nhị phân nguy hại/an toàn.",
        "binary_harmful_vs_safe": binary_metrics(gold_harm, pred_harm),
        "recall_theo_nhan": recall_by_label,
        "giong_vung": {
            **by_region,
            "worst_group_binary_f1": min(f1s) if f1s else 0.0,
            "region_gap_binary_f1": round(max(f1s) - min(f1s), 4) if f1s else 0.0,
            "khong_ro_vung": sum(v is None for v in regions),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=Path("data/v2/test_day_du.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    meta = {str(r["id"]): r for r in read_jsonl(args.meta)}
    result = evaluate(read_jsonl(args.pred), meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi kết quả vào {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
