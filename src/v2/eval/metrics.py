"""Bộ độ đo đánh giá toàn diện SafeViChi v2 (phân loại, lát cắt, thống kê, so sánh LLM).

Đáp ứng trực tiếp 3 yêu cầu cải tiến của ban tổ chức:
1. Tách phân loại khỏi bằng chứng, đo P/R/F1 từng nhãn, FPR trên AN_TOAN, Recall DE_DOA/LUA_DAO, ECE/PR-AUC.
2. Cắt lát in-memory theo C0/C1/C2 (kèm ΔF1), vùng miền (kèm Worst-group F1), kịch bản trong/ngoài phân bố,
   thẻ ca khó và số lượt hội thoại.
3. Kiểm định thống kê Bootstrap 95% CI (seed 42) + McNemar ghép cặp, cùng độ tin cậy đầu ra & đánh đổi chi phí
   khi so sánh với LLM API.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.v2.eval.evidence import HE_SO_PHU_MAC_DINH, SAFE_LABEL, evaluate_evidence

LABELS: tuple[str, ...] = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
LABEL_TO_IDX: dict[str, int] = {lbl: idx for idx, lbl in enumerate(LABELS)}
SEED_DEFAULT = 42


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s in LABEL_TO_IDX else None


def bootstrap_ci_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED_DEFAULT,
) -> dict[str, float]:
    """Khoảng tin cậy Bootstrap percentile (seed cố định để tái hiện 100%)."""
    n = len(y_true)
    if n == 0:
        return {"low": 0.0, "high": 0.0, "halfwidth": 0.0}
    rng = np.random.default_rng(seed)
    scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = float(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {
        "low": round(float(lo), 4),
        "high": round(float(hi), 4),
        "halfwidth": round(float((hi - lo) / 2), 4),
    }


def mcnemar_exact(y_true: Sequence[Any], pred_a: Sequence[Any], pred_b: Sequence[Any]) -> dict[str, Any]:
    """Kiểm định McNemar chính xác (binomial test) ghép cặp 2 hệ thống trên cùng tập mẫu."""
    yt = np.asarray(list(y_true))
    pa = np.asarray(list(pred_a))
    pb = np.asarray(list(pred_b))
    a_ok = pa == yt
    b_ok = pb == yt
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b_only_a_correct": b, "c_only_b_correct": c, "p_value": 1.0, "significant": False}
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {
        "b_only_a_correct": b,
        "c_only_b_correct": c,
        "p_value": round(p, 6),
        "significant": p < 0.05,
    }


def compute_ece(y_true_idx: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) trên xác suất lớp dự đoán cao nhất."""
    if len(y_true_idx) == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true_idx).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_idx)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return round(float(ece), 4)


def evaluate_output_reliability(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Đo độ ổn định đầu ra (đặc biệt quan trọng cho SLM sinh và LLM API).

    Các mẫu từ chối trả lời (refusal=True), hỏng JSON (parse_error=True), hoặc nhãn ngoài 5 lớp
    đều được thống kê minh bạch và giữ lại trong mẫu số (tính sai, không được loại bỏ).
    """
    n = len(predictions)
    if n == 0:
        return {
            "n": 0,
            "valid_output_rate": 0.0,
            "invalid_label_rate": 0.0,
            "parse_error_rate": 0.0,
            "refusal_rate": 0.0,
        }
    n_invalid_label = 0
    n_parse_error = 0
    n_refusal = 0
    for pred in predictions:
        if pred.get("refusal"):
            n_refusal += 1
        if pred.get("parse_error"):
            n_parse_error += 1
        if _normalize_label(pred.get("nhan")) is None:
            n_invalid_label += 1
    n_bad = sum(
        1
        for pred in predictions
        if pred.get("refusal") or pred.get("parse_error") or _normalize_label(pred.get("nhan")) is None
    )
    return {
        "n": n,
        "valid_output_rate": round((n - n_bad) / n, 4),
        "invalid_label_rate": round(n_invalid_label / n, 4),
        "parse_error_rate": round(n_parse_error / n, 4),
        "refusal_rate": round(n_refusal / n, 4),
    }


def evaluate_classification(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = SEED_DEFAULT,
) -> dict[str, Any]:
    """Đo đầy đủ chỉ số phân loại 5 nhãn + nhị phân (so với v1) + hiệu chỉnh xác suất (nếu có)."""
    if len(records) != len(predictions):
        raise ValueError(f"Số mẫu gold ({len(records)}) khác số mẫu dự đoán ({len(predictions)})")
    n = len(records)
    if n == 0:
        return {"n": 0}

    y_true_idx = np.array([LABEL_TO_IDX[str(r["nhan"])] for r in records], dtype=np.int64)
    # Dự đoán lỗi/từ chối/nhãn lạ gán mã -1 để luôn bị tính là sai ở mọi lớp
    y_pred_idx = np.array(
        [
            -1
            if (p.get("refusal") or p.get("parse_error") or _normalize_label(p.get("nhan")) is None)
            else LABEL_TO_IDX[str(p["nhan"]).strip()]
            for p in predictions
        ],
        dtype=np.int64,
    )

    label_ids = list(range(len(LABELS)))
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true_idx, y_pred_idx, labels=label_ids, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1s[i]), 4),
            "support": int(sup[i]),
        }
        for i, lbl in enumerate(LABELS)
    }

    present_mask = sup > 0
    macro_f1 = float(f1s[present_mask].mean()) if present_mask.any() else 0.0
    weighted_f1 = float(f1_score(y_true_idx, y_pred_idx, labels=label_ids, average="weighted", zero_division=0))
    accuracy = float((y_true_idx == y_pred_idx).mean())
    cm = confusion_matrix(y_true_idx, np.where(y_pred_idx >= 0, y_pred_idx, 0), labels=label_ids).tolist()

    # Nhị phân: Nguy hại (1..4) vs AN_TOAN (0) — so sánh trực tiếp với SafeViChi v1
    # Nếu mô hình lỗi/từ chối (-1): coi là dự đoán sai so với nhãn thật (gây FN với câu nguy hại, FP với câu AN_TOAN)
    y_true_bin = (y_true_idx != 0).astype(np.int64)
    y_pred_bin = np.where(y_pred_idx < 0, 1 - y_true_bin, (y_pred_idx != 0).astype(np.int64))
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    bin_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_f1 = (2 * bin_p * bin_r / (bin_p + bin_r)) if (bin_p + bin_r) > 0 else 0.0
    fpr_an_toan = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_nguy_hai = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Calibration & PR-AUC (chỉ tính khi toàn bộ dự đoán có mảng probs hợp lệ độ dài 5; với LLM sẽ là None)
    calibration: dict[str, Any] | None = None
    if all(isinstance(p.get("probs"), (list, tuple)) and len(p["probs"]) == len(LABELS) for p in predictions):
        probs = np.asarray([p["probs"] for p in predictions], dtype=np.float64)
        probs = np.clip(probs, 1e-7, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        pr_auc_per_class: dict[str, float | None] = {}
        valid_aucs: list[float] = []
        for i, lbl in enumerate(LABELS):
            bin_lbl = (y_true_idx == i).astype(np.int64)
            if bin_lbl.sum() > 0 and bin_lbl.sum() < n:
                ap = float(average_precision_score(bin_lbl, probs[:, i]))
                pr_auc_per_class[lbl] = round(ap, 4)
                valid_aucs.append(ap)
            else:
                pr_auc_per_class[lbl] = None
        calibration = {
            "ece": compute_ece(y_true_idx, probs),
            "nll": round(float(log_loss(y_true_idx, probs, labels=label_ids)), 4),
            "pr_auc_macro": round(float(np.mean(valid_aucs)), 4) if valid_aucs else None,
            "pr_auc_per_class": pr_auc_per_class,
        }

    result: dict[str, Any] = {
        "n": n,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "binary_harmful_vs_safe": {
            "precision": round(bin_p, 4),
            "recall": round(bin_r, 4),
            "f1": round(bin_f1, 4),
            "fpr_an_toan": round(fpr_an_toan, 4),
            "fnr_nguy_hai": round(fnr_nguy_hai, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "confusion_matrix": {"labels": list(LABELS), "matrix": cm},
        "calibration": calibration,
    }

    if n_bootstrap > 0:
        present_labels = [i for i in label_ids if sup[i] > 0]
        result["bootstrap_95"] = {
            "macro_f1": bootstrap_ci_metric(
                y_true_idx,
                y_pred_idx,
                lambda yt, yp: f1_score(yt, yp, labels=present_labels, average="macro", zero_division=0),
                n_resamples=n_bootstrap,
                seed=seed,
            ),
            "binary_f1": bootstrap_ci_metric(
                y_true_bin,
                y_pred_bin,
                lambda yt, yp: f1_score(yt, yp, pos_label=1, average="binary", zero_division=0),
                n_resamples=n_bootstrap,
                seed=seed,
            ),
        }
    return result


def evaluate_bundle(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    n_bootstrap: int = 1000,
    he_so_phu: float = HE_SO_PHU_MAC_DINH,
    seed: int = SEED_DEFAULT,
    include_slices: bool = True,
) -> dict[str, Any]:
    """Chạy đánh giá đầy đủ (phân loại + bằng chứng + độ ổn định đầu ra + toàn bộ lát cắt in-memory)."""
    summary: dict[str, Any] = {
        "classification": evaluate_classification(records, predictions, n_bootstrap=n_bootstrap, seed=seed),
        "evidence": evaluate_evidence(records, predictions, he_so_phu=he_so_phu),
        "output_reliability": evaluate_output_reliability(predictions),
    }
    if not include_slices:
        return summary

    def _slice_report(indices: list[int]) -> dict[str, Any]:
        sub_r = [records[i] for i in indices]
        sub_p = [predictions[i] for i in indices]
        cls = evaluate_classification(sub_r, sub_p, n_bootstrap=0, seed=seed)
        ev = evaluate_evidence(sub_r, sub_p, he_so_phu=he_so_phu)
        return {
            "n": len(indices),
            "macro_f1": cls["macro_f1"],
            "binary_f1": cls["binary_harmful_vs_safe"]["f1"],
            "fpr_an_toan": cls["binary_harmful_vs_safe"]["fpr_an_toan"],
            "evidence_f1_theo_y": ev["f1_theo_y"]["f1"],
            "evidence_f1_muc_tu": ev["f1_muc_tu"]["f1"],
            "safe_false_highlight_rate": ev["safe_false_highlight_rate"],
        }

    def _group_by(key_fn) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx, rec in enumerate(records):
            val = key_fn(rec)
            if val is None or val == "":
                continue
            groups.setdefault(str(val), []).append(idx)
        return dict(sorted(groups.items()))

    slices: dict[str, Any] = {}

    # 1. Theo mức biến dạng C0 / C1 / C2 + độ tụt ΔF1
    by_perturb = _group_by(lambda r: r.get("bien_dang") or "C0")
    if by_perturb:
        perturb_rep = {k: _slice_report(v) for k, v in by_perturb.items()}
        if "C0" in perturb_rep:
            c0_f1 = perturb_rep["C0"]["macro_f1"]
            deltas = {
                f"{k}_minus_C0": round(perturb_rep[k]["macro_f1"] - c0_f1, 4)
                for k in perturb_rep
                if k != "C0"
            }
            if deltas:
                perturb_rep["delta_macro_f1"] = deltas
        slices["bien_dang"] = perturb_rep

    # 2. Theo nhóm biến dạng đã thấy / chưa thấy và từng kiểu biến dạng
    by_nhom_bd = _group_by(lambda r: r.get("nhom_bien_dang"))
    if by_nhom_bd:
        slices["nhom_bien_dang"] = {k: _slice_report(v) for k, v in by_nhom_bd.items()}

    by_kieu_bd = _group_by(lambda r: r.get("kieu_bien_dang"))
    if by_kieu_bd:
        slices["kieu_bien_dang"] = {k: _slice_report(v) for k, v in by_kieu_bd.items()}

    # 3. Theo giọng vùng miền (bac / trung / nam) + Worst-group F1 + chênh max-min
    by_region = _group_by(lambda r: r.get("giong_vung"))
    if by_region:
        region_rep = {k: _slice_report(v) for k, v in by_region.items()}
        f1_vals = [region_rep[k]["macro_f1"] for k in ("bac", "trung", "nam") if k in region_rep]
        if f1_vals:
            region_rep["worst_group_macro_f1"] = round(min(f1_vals), 4)
            region_rep["region_gap_max_min"] = round(max(f1_vals) - min(f1_vals), 4)
        slices["giong_vung"] = region_rep

    # 4. Theo phân bố kịch bản (trong_phan_phoi / chua_thay)
    by_part = _group_by(lambda r: r.get("phan_test"))
    if by_part:
        slices["phan_test"] = {k: _slice_report(v) for k, v in by_part.items()}

    # 5. Lát cắt tổ hợp (bien_dang x giong_vung và bien_dang x phan_test — thay thế 15 file cắt lát trên đĩa)
    if len(by_perturb) > 1 or by_region or by_part:
        combo: dict[str, Any] = {}
        for bd, bd_idx in by_perturb.items():
            for region in ("bac", "trung", "nam"):
                sub = [i for i in bd_idx if records[i].get("giong_vung") == region]
                if sub:
                    combo[f"{bd}_{region}"] = _slice_report(sub)
            for src_val, dst_name in (("trong_phan_phoi", "da_thay"), ("chua_thay", "chua_thay")):
                sub = [i for i in bd_idx if records[i].get("phan_test") == src_val]
                if sub:
                    combo[f"{bd}_{dst_name}"] = _slice_report(sub)
        if combo:
            slices["bien_dang_x_nhom"] = combo

    # 6. Theo số lượt hội thoại (1 dòng vs nhiều dòng)
    by_turns = _group_by(
        lambda r: "1_luot" if int(r.get("so_dong") or len(str(r["text"]).splitlines())) <= 1 else "nhieu_luot"
    )
    if by_turns:
        slices["so_luot"] = {k: _slice_report(v) for k, v in by_turns.items()}

    # 7. Theo cách gán (nguoi vs llm_nguoi_duyet — hỗ trợ tách thiên lệch LLM)
    by_annot = _group_by(lambda r: r.get("cach_gan"))
    if by_annot:
        slices["cach_gan"] = {k: _slice_report(v) for k, v in by_annot.items()}

    # 8. Theo thẻ ca khó (tieng_long, vung_mien, an_y, trich_dan...)
    tag_groups: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        for tag in rec.get("the") or []:
            tag_groups.setdefault(str(tag), []).append(idx)
    if tag_groups:
        slices["the"] = {k: _slice_report(v) for k, v in sorted(tag_groups.items())}

    summary["slices"] = slices
    return summary


def compare_systems(
    records: Sequence[dict[str, Any]],
    preds_candidate: Sequence[dict[str, Any]],
    preds_reference: Sequence[dict[str, Any]],
    candidate_cost: dict[str, float] | None = None,
    reference_cost: dict[str, float] | None = None,
) -> dict[str, Any]:
    """So sánh ghép cặp mô hình ứng viên (vd. SafeViChi v2) với mốc đối chiếu (vd. v1 / ablation / LLM API)."""
    cand = evaluate_bundle(records, preds_candidate, n_bootstrap=0, include_slices=False)
    ref = evaluate_bundle(records, preds_reference, n_bootstrap=0, include_slices=False)

    y_true = [str(r["nhan"]) for r in records]
    pa = [
        "__INVALID__" if (p.get("refusal") or p.get("parse_error") or _normalize_label(p.get("nhan")) is None)
        else str(p["nhan"]).strip()
        for p in preds_candidate
    ]
    pb = [
        "__INVALID__" if (p.get("refusal") or p.get("parse_error") or _normalize_label(p.get("nhan")) is None)
        else str(p["nhan"]).strip()
        for p in preds_reference
    ]

    y_true_bin = [0 if y == SAFE_LABEL else 1 for y in y_true]
    pa_bin = [1 - yt if p == "__INVALID__" else (0 if p == SAFE_LABEL else 1) for yt, p in zip(y_true_bin, pa)]
    pb_bin = [1 - yt if p == "__INVALID__" else (0 if p == SAFE_LABEL else 1) for yt, p in zip(y_true_bin, pb)]

    ref_macro = ref["classification"]["macro_f1"]
    cand_macro = cand["classification"]["macro_f1"]
    retention = round(100.0 * cand_macro / ref_macro, 2) if ref_macro > 0 else None

    out: dict[str, Any] = {
        "n": len(records),
        "delta_macro_f1": round(cand_macro - ref_macro, 4),
        "quality_retention_pct": retention,
        "delta_binary_f1": round(
            cand["classification"]["binary_harmful_vs_safe"]["f1"]
            - ref["classification"]["binary_harmful_vs_safe"]["f1"],
            4,
        ),
        "delta_fpr_an_toan": round(
            cand["classification"]["binary_harmful_vs_safe"]["fpr_an_toan"]
            - ref["classification"]["binary_harmful_vs_safe"]["fpr_an_toan"],
            4,
        ),
        "delta_evidence_f1_theo_y": round(
            cand["evidence"]["f1_theo_y"]["f1"] - ref["evidence"]["f1_theo_y"]["f1"], 4
        ),
        "mcnemar_5class": mcnemar_exact(y_true, pa, pb),
        "mcnemar_binary": mcnemar_exact(y_true_bin, pa_bin, pb_bin),
        "candidate_output_reliability": cand["output_reliability"],
        "reference_output_reliability": ref["output_reliability"],
    }

    if candidate_cost and reference_cost:
        tradeoff: dict[str, Any] = {"candidate": candidate_cost, "reference": reference_cost}
        for key in ("latency_p50_ms", "latency_p95_ms"):
            c_val = candidate_cost.get(key)
            r_val = reference_cost.get(key)
            if c_val and r_val and c_val > 0:
                tradeoff[f"{key}_speedup_x"] = round(r_val / c_val, 2)
        out["cost_latency_tradeoff"] = tradeoff
    return out
