"""CLI đánh giá chuẩn cho mọi mô hình / baseline / LLM trên bộ test SafeViChi v2.

Ví dụ chạy trên C0 + C1 + C2 (tự động lọc in-memory mọi lát cắt vùng miền & kịch bản):
    python -m src.v2.eval.evaluate \
        --gold data/v2/test_day_du.jsonl data/v2/test_perturb_C1.jsonl data/v2/test_perturb_C2.jsonl \
        --pred results/v2/<cau_hinh>/predictions.jsonl \
        --out results/v2/<cau_hinh>/metrics.json

So sánh ghép cặp với một mốc đối chiếu (vd. SafeViChi v1 hoặc LLM API):
    python -m src.v2.eval.evaluate \
        --gold data/v2/test_day_du.jsonl \
        --pred results/v2/v2_full/predictions.jsonl \
        --reference-pred results/v2/llm_api/predictions.jsonl \
        --out results/v2/v2_full/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.v2.eval.evidence import HE_SO_PHU_MAC_DINH
from src.v2.eval.metrics import SEED_DEFAULT, compare_systems, evaluate_bundle


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_predictions(
    records: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ghép dự đoán theo trường `id` khớp chính xác thứ tự của tập gold."""
    by_id: dict[str, dict[str, Any]] = {}
    for row in pred_rows:
        rid = str(row["id"])
        if rid in by_id:
            raise ValueError(f"Trùng id dự đoán: {rid}")
        by_id[rid] = row
    aligned: list[dict[str, Any]] = []
    missing: list[str] = []
    for rec in records:
        rid = str(rec["id"])
        if rid not in by_id:
            missing.append(rid)
        else:
            aligned.append(by_id[rid])
    if missing:
        raise ValueError(f"Thiếu {len(missing)} id trong file dự đoán (vd. {missing[:5]})")
    return aligned


def enrich_metadata(records: list[dict[str, Any]], meta_source: Path | None) -> list[dict[str, Any]]:
    """Bổ sung metadata (giong_vung, phan_test, cach_gan, the, so_dong) từ test_day_du.jsonl nếu thiếu."""
    if meta_source is None or not meta_source.exists():
        return records
    meta_by_id = {str(r["id"]): r for r in read_jsonl(meta_source)}
    fields = ("giong_vung", "phan_test", "kich_ban", "cach_gan", "the", "so_dong")
    enriched: list[dict[str, Any]] = []
    for rec in records:
        lookup_id = str(rec.get("goc_id") or rec["id"])
        src = meta_by_id.get(lookup_id)
        if src is None:
            enriched.append(rec)
            continue
        merged = dict(rec)
        for k in fields:
            if not merged.get(k) and src.get(k) is not None:
                merged[k] = src[k]
        enriched.append(merged)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, nargs="+", required=True, help="1 hoặc nhiều file JSONL gold (C0/C1/C2)")
    parser.add_argument("--pred", type=Path, required=True, help="File JSONL dự đoán (có id, nhan, bang_chung)")
    parser.add_argument("--meta", type=Path, default=None, help="File test_day_du.jsonl để bổ sung metadata nếu gold thiếu")
    parser.add_argument("--reference-pred", type=Path, default=None, help="File dự đoán mốc đối chiếu để chạy McNemar")
    parser.add_argument("--out", type=Path, default=None, help="Đường dẫn ghi metrics.json")
    parser.add_argument(
        "--he-so-phu", type=float, default=HE_SO_PHU_MAC_DINH,
        help="Hệ số recall F1 theo ý: tô trúng 1/he_so_phu số từ của ý gán là đủ điểm (mặc định 1,5)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Số lần bootstrap CI 95% (0 để tắt)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    records = [row for path in args.gold for row in read_jsonl(path)]
    records = enrich_metadata(records, args.meta)
    preds = align_predictions(records, read_jsonl(args.pred))

    report = evaluate_bundle(
        records,
        preds,
        n_bootstrap=args.n_bootstrap,
        he_so_phu=args.he_so_phu,
        seed=args.seed,
        include_slices=True,
    )
    if args.reference_pred is not None:
        ref_preds = align_predictions(records, read_jsonl(args.reference_pred))
        report["comparison_vs_reference"] = compare_systems(records, preds, ref_preds)

    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"Đã ghi kết quả đánh giá vào {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
