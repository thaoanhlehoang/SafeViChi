"""
Module giải thích định vị cụm từ gây cảnh báo (occlusion-based).

Tương ứng Mục 3.6 trong docs/plan.md.

Đây là kỹ thuật giải thích HẬU KỲ (post-hoc): dùng chính bộ phân loại đã
huấn luyện (src/classifier) làm hàm chấm điểm, KHÔNG cần huấn luyện thêm
model/head nào khác. Xem phần "Làm rõ nội dung" trong lịch sử trao đổi về
lý do không dùng token classification / span detection cho việc này.

Cách hoạt động:
  1. Trượt qua từng cụm n-gram trong câu
  2. Che (mask/remove) cụm đó, chạy lại qua bộ phân loại
  3. Đo mức giảm điểm số của nhãn dự đoán gốc
  4. Cụm nào làm điểm giảm nhiều nhất -> cụm gây cảnh báo

Đối chiếu định lượng với nhãn mức cụm từ của ViHOS để tính độ chính xác
định vị (Mục 4.2).
"""

from __future__ import annotations
from typing import Callable, List, Tuple

ScoreFn = Callable[[str], float]  # nhận văn bản, trả về điểm số của nhãn dự đoán gốc


def occlude_ngrams(tokens: List[str], n: int = 1) -> List[Tuple[int, int, List[str]]]:
    """Sinh danh sách (start, end, tokens_còn_lại) sau khi che từng n-gram."""
    spans = []
    for i in range(len(tokens) - n + 1):
        remaining = tokens[:i] + tokens[i + n:]
        spans.append((i, i + n, remaining))
    return spans


def explain(
    text: str,
    score_fn: ScoreFn,
    n: int = 1,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Trả về top_k cụm từ (n-gram) gây ảnh hưởng lớn nhất tới quyết định
    phân loại, kèm mức điểm giảm tương ứng.

    score_fn: hàm nhận 1 câu, trả về điểm số (xác suất) của nhãn gốc
              -> cần wrap quanh model đã fine-tune ở src/classifier
    """
    tokens = text.split()
    base_score = score_fn(text)

    results = []
    for start, end, remaining_tokens in occlude_ngrams(tokens, n=n):
        occluded_text = " ".join(remaining_tokens)
        occluded_score = score_fn(occluded_text) if occluded_text else 0.0
        drop = base_score - occluded_score
        phrase = " ".join(tokens[start:end])
        results.append((phrase, drop))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Ví dụ minh họa với hàm chấm điểm giả (thay bằng model thật khi có checkpoint)
    def dummy_score_fn(text: str) -> float:
        bad_words = {"ngu", "óc", "chó"}
        return sum(1.0 for w in text.split() if w.lower() in bad_words) / max(len(text.split()), 1)

    sample = "mày ngu như con chó"
    print(explain(sample, dummy_score_fn, n=1, top_k=3))
