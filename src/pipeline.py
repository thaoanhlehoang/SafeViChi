"""
Ghép toàn bộ pipeline một đường thẳng (Mục 3.1 trong docs/plan.md):

    Văn bản đầu vào
        -> [1] Lớp chuẩn hóa (rule-based)
        -> [2] Bộ phân loại (ViSoBERT fine-tune)
        -> [3] Module giải thích (occlusion)
        -> Đầu ra: nhãn rủi ro + cụm từ gây cảnh báo

Đây là điểm vào duy nhất dùng cho demo web (demo/) khi đã export model
sang ONNX. Không rẽ nhánh, không router.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from src.normalization.normalizer import normalize
from src.explainer.occlusion import explain

LABEL_NAMES = ["clean", "offensive", "hate"]


@dataclass
class PipelineResult:
    original_text: str
    normalized_text: str
    label: str
    confidence: float
    flagged_phrases: List[Tuple[str, float]]


class SafeViChiPipeline:
    def __init__(self, classifier_predict_fn, classifier_score_fn):
        """
        classifier_predict_fn: nhận văn bản đã chuẩn hóa, trả về (label_index, confidence)
        classifier_score_fn:   nhận văn bản, trả về điểm số của nhãn dự đoán gốc
                                (dùng cho module giải thích occlusion)
        """
        self.predict_fn = classifier_predict_fn
        self.score_fn = classifier_score_fn

    def run(self, text: str, explain_top_k: int = 3) -> PipelineResult:
        normalized = normalize(text)
        label_idx, confidence = self.predict_fn(normalized)
        label = LABEL_NAMES[label_idx]

        flagged = []
        if label != "clean":
            flagged = explain(normalized, self.score_fn, n=1, top_k=explain_top_k)

        return PipelineResult(
            original_text=text,
            normalized_text=normalized,
            label=label,
            confidence=confidence,
            flagged_phrases=flagged,
        )


if __name__ == "__main__":
    # TODO: thay bằng model thật sau khi fine-tune (src/classifier/train.py)
    def dummy_predict_fn(text: str):
        bad = {"ngu", "óc", "chó"}
        hit = any(w in text.lower() for w in bad)
        return (1, 0.87) if hit else (0, 0.95)

    def dummy_score_fn(text: str) -> float:
        bad = {"ngu", "óc", "chó"}
        return sum(1.0 for w in text.split() if w.lower() in bad) / max(len(text.split()), 1)

    pipeline = SafeViChiPipeline(dummy_predict_fn, dummy_score_fn)
    result = pipeline.run("m4y ng.u qu4")
    print(result)
