from src.explainer.occlusion import (
    normalize_importance, occlude_words, split_words, top_k_spans,
)


class FakeScorer:
    """Scorer giả: P(hate) = số từ 'xấu' còn lại trong câu / tổng số từ ban đầu."""

    def __init__(self, bad_words: set[str], total_words: int):
        self.bad_words = bad_words
        self.total_words = total_words

    def score_batch(self, texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            words = text.split() if text else []
            n_bad = sum(1 for w in words if w in self.bad_words)
            scores.append(n_bad / self.total_words)
        return scores


def test_split_words_whitespace():
    assert split_words("Bấp bênh vl thế") == ["Bấp", "bênh", "vl", "thế"]


def test_occlude_words_same_length_as_input():
    words = ["a", "b", "c"]
    scorer = FakeScorer(bad_words={"b"}, total_words=len(words))
    importance = occlude_words(words, scorer, window=1)
    assert len(importance) == len(words)


def test_occlude_words_flags_the_bad_word():
    words = ["câu", "này", "ngu", "vậy"]
    scorer = FakeScorer(bad_words={"ngu"}, total_words=len(words))
    importance = occlude_words(words, scorer, window=1)
    # che "ngu" làm P(hate) giảm 1/4 -> importance ở đúng vị trí đó cao nhất
    assert importance.index(max(importance)) == words.index("ngu")
    assert importance == [0.0, 0.0, 0.25, 0.0]


def test_occlude_words_empty_input():
    assert occlude_words([], FakeScorer(set(), 1)) == []


def test_top_k_spans_threshold():
    assert top_k_spans([0.0, 0.3, 0.0, 0.5], threshold=0.0) == [1, 3]


def test_top_k_spans_fixed_k():
    assert top_k_spans([0.1, 0.9, 0.05, 0.4], k=2) == [1, 3]


def test_normalize_importance_scales_to_peak():
    assert normalize_importance([1.0, 4.0, 2.0]) == [0.25, 1.0, 0.5]


def test_normalize_importance_all_negative_flags_nothing():
    # Không từ nào làm điểm giảm -> trả toàn 0.0, không được lật dấu
    assert normalize_importance([-1.0, -4.0]) == [0.0, 0.0]


def test_normalize_importance_empty():
    assert normalize_importance([]) == []
