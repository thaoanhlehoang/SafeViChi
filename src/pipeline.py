"""
Ghép toàn bộ pipeline một đường thẳng (Mục 3.1 trong docs/plan.md):

    Văn bản đầu vào
        -> [1] Lớp chuẩn hóa (rule-based)
        -> [2] Bộ phân loại (ViSoBERT fine-tune / ViHateT5)
        -> [3] Module giải thích (occlusion)
        -> Đầu ra: nhãn rủi ro + cụm từ gây cảnh báo
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from src.normalization.normalizer import normalize
from src.explainer.occlusion import OcclusionScorer, split_words, occlude_words, normalize_importance, top_k_spans

LABEL_NAMES = ["clean", "offensive", "hate"]


@dataclass
class PipelineResult:
    original_text: str
    normalized_text: str
    label: str
    confidence: float
    flagged_phrases: List[Tuple[str, float]]


class SafeViChiPipeline:
    def __init__(self, model_path: str):
        """
        Load model và khởi tạo OcclusionScorer.
        """
        self.scorer = OcclusionScorer(model_path)

    def run(self, text: str, explain_top_k: int = 3) -> PipelineResult:
        normalized = normalize(text)
        
        # Lấy điểm margin (hate - clean)
        margin = self.scorer.score(normalized)
        
        # Tính nhãn và độ tin cậy (sigmoid của margin)
        # Giả sử margin > 0 thì là hate/offensive, < 0 là clean
        # (cần điều chỉnh lại logic ánh xạ nhãn tùy theo cách bạn định nghĩa offensive vs hate)
        confidence = self.scorer.probability(normalized)
        if margin > 0:
            label = "hate" # hoặc offensive tùy ngưỡng
        else:
            label = "clean"

        flagged = []
        if label != "clean":
            words = split_words(normalized)
            importance = occlude_words(words, self.scorer)
            norm_imp = normalize_importance(importance)
            spans = top_k_spans(norm_imp, k=explain_top_k, threshold=0.1)
            flagged = [(words[i], norm_imp[i]) for i in spans]

        return PipelineResult(
            original_text=text,
            normalized_text=normalized,
            label=label,
            confidence=confidence,
            flagged_phrases=flagged,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn tới thư mục model")
    parser.add_argument("--text", type=str, default="m4y ng.u qu4", help="Văn bản cần kiểm tra")
    args = parser.parse_args()
    
    pipeline = SafeViChiPipeline(args.model)
    result = pipeline.run(args.text)
    print(result)
