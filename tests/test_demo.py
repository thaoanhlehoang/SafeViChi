from src.explainability.demo import bucket, render


def test_bucket_levels():
    assert bucket(0.8, 0.1) == "strong"
    assert bucket(0.3, 0.1) == "medium"
    assert bucket(0.15, 0.1) == "weak"
    assert bucket(0.05, 0.1) == "none"


def test_render_markdown_marks_only_flagged_words():
    words = ["câu", "này", "ngu", "vậy"]
    importance = [0.0, 0.05, 0.9, 0.3]
    assert render(words, importance, 0.1, "markdown") == "câu này **ngu** *vậy*"


def test_render_plain_brackets_flagged_words():
    assert render(["a", "b"], [0.0, 0.5], 0.1, "plain") == "a [b]"


def test_render_nothing_flagged_returns_original_sentence():
    assert render(["a", "b"], [0.0, 0.0], 0.1, "markdown") == "a b"
