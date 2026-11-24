"""Mô-đun đánh giá chuẩn cho SafeViChi v2."""

from src.v2.eval.evidence import evaluate_evidence, score_evidence_item, tokenize_words_and_clauses
from src.v2.eval.metrics import (
    LABELS,
    SEED_DEFAULT,
    bootstrap_ci_metric,
    compare_systems,
    compute_ece,
    evaluate_bundle,
    evaluate_classification,
    evaluate_output_reliability,
    mcnemar_exact,
)

__all__ = [
    "LABELS",
    "SEED_DEFAULT",
    "bootstrap_ci_metric",
    "compare_systems",
    "compute_ece",
    "evaluate_bundle",
    "evaluate_classification",
    "evaluate_evidence",
    "evaluate_output_reliability",
    "mcnemar_exact",
    "score_evidence_item",
    "tokenize_words_and_clauses",
]
