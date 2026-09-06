from src.dataset_builder.build_vihos import char_spans_to_word_mask


def test_span_mask_same_length_as_words():
    words, span_mask = char_spans_to_word_mask("Bấp bênh vl thế", {9, 10})
    assert len(words) == len(span_mask)
    assert words == ["Bấp", "bênh", "vl", "thế"]
    assert span_mask == [0, 0, 1, 0]


def test_empty_span_all_zero():
    words, span_mask = char_spans_to_word_mask("Dừa lắm :))", set())
    assert words == ["Dừa", "lắm", ":))"]
    assert span_mask == [0, 0, 0]


def test_index_in_whitespace_flags_no_word():
    # "Má nứng quá": M=0 á=1 (space)=2 n=3... -> index 2 là khoảng trắng,
    # không thuộc từ nào cả.
    words, span_mask = char_spans_to_word_mask("Má nứng quá", {2})
    assert words == ["Má", "nứng", "quá"]
    assert span_mask == [0, 0, 0]
