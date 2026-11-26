"""Metric dùng chung cho train.py và evaluate.py (B2/B3/B4 đều gọi qua đây,
để số liệu giữa các baseline được tính đúng cùng 1 công thức, so sánh được).

Target text sinh ra ở chữ thường (xem src/classifier/data.py) vì tokenizer
của ViHateT5 map "HATE"/"CLEAN" viết hoa cả hai về cùng <unk>.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LABEL2ID = {"clean": 0, "hate": 1}
INVALID_LABEL_ID = -1


def _binary_scores(label_ids: np.ndarray, pred_ids: np.ndarray) -> dict[str, float]:
    label_ids = np.asarray(label_ids)
    pred_ids = np.asarray(pred_ids)
    if len(label_ids) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    pos = LABEL2ID["hate"]
    # sklearn từ chối average="binary" nếu pred_ids có giá trị ngoài {0,1} (vd.
    # INVALID_LABEL_ID khi model sinh ra text không phải "hate"/"clean" — hay
    # gặp với model CHƯA fine-tune như baseline B3). Coi mọi output không hợp
    # lệ là SAI: gán về nhãn ngược với label thật, để chắc chắn bị tính sai
    # (không vô tình "đúng" khi -1 == label thật), đồng thời giữ pred_ids chỉ
    # còn 2 giá trị {0,1} để sklearn tính binary được bình thường.
    clean_pred_ids = np.where(
        np.isin(pred_ids, list(LABEL2ID.values())), pred_ids, 1 - label_ids
    )
    return {
        "accuracy": accuracy_score(label_ids, clean_pred_ids),
        "f1": f1_score(label_ids, clean_pred_ids, pos_label=pos, average="binary", zero_division=0),
        "precision": precision_score(label_ids, clean_pred_ids, pos_label=pos, average="binary", zero_division=0),
        "recall": recall_score(label_ids, clean_pred_ids, pos_label=pos, average="binary", zero_division=0),
    }


def to_label_id(text: str) -> int:
    return LABEL2ID.get(text.strip().lower(), INVALID_LABEL_ID)


def score_labels(label_ids: np.ndarray, pred_ids: np.ndarray, sources: list[str] | None = None) -> dict:
    """accuracy/f1/precision/recall tổng + tách riêng theo từng nhóm `sources`."""
    metrics = _binary_scores(label_ids, pred_ids)
    metrics["invalid_generation_rate"] = float(np.mean(pred_ids == INVALID_LABEL_ID))

    if sources is not None and len(sources) == len(pred_ids):
        sources_arr = np.array(sources)
        for source in sorted(set(sources)):
            mask = sources_arr == source
            for key, value in _binary_scores(label_ids[mask], pred_ids[mask]).items():
                metrics[f"{source}_{key}"] = value

    return metrics


def build_compute_metrics(tokenizer, eval_sources: list[str] | None = None):
    """eval_sources: cột source của tập eval, theo đúng thứ tự dòng, để báo cáo
    thêm metric TÁCH RIÊNG cho từng nhóm (perturbed/clean/blindspot)."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_ids = np.array([to_label_id(p) for p in decoded_preds])
        label_ids = np.array([to_label_id(l) for l in decoded_labels])

        return score_labels(label_ids, pred_ids, eval_sources)

    return compute_metrics


def report(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    for key in sorted(metrics):
        value = metrics[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
