"""
Unit tests cho bộ chuẩn hóa SafeViChi.

Bao phủ 11 loại perturbation:
  - Xử lý: surface_obfuscation, boundary_variation, expressive_lengthening,
           diacritic_variation, abbreviation_clipping, intentional_spelling,
           phonetic_spelling, dialectal_writing, slang_lexical
  - Bỏ qua: context_dependent, typographical_noise

Nguyên tắc kiểm tra an toàn: THÀ KHÔNG SỬA CHỨ ĐỪNG SỬA SAI.
  - Số thuần phải giữ nguyên
  - Từ tiếng Việt hợp lệ không bị thay đổi
  - Dấu câu được bảo toàn
  - Viết hoa được bảo toàn

Chạy:
    pytest tests/test_normalizer.py -v
    # hoặc
    python -m pytest tests/test_normalizer.py -v
"""

import pytest
from src.normalization.normalizer import (
    normalize,
    normalize_unicode,
    map_lookalike_chars,
    remove_inserted_separators,
    collapse_elongation,
    restore_teencode,
    get_dict_size,
)


# ============================================================
# Bước 1: Unicode NFC
# ============================================================

class TestNormalizeUnicode:
    def test_nfc_basic(self):
        # Chữ ề có thể biểu diễn bằng 1 hoặc 2 code point
        composed = "\u1EC1"       # ề (NFC - 1 code point)
        decomposed = "e\u0302\u0300"  # e + ̂ + ̀ (NFD - 3 code points)
        assert normalize_unicode(decomposed) == composed

    def test_already_nfc(self):
        text = "xin chào"
        assert normalize_unicode(text) == text

    def test_empty(self):
        assert normalize_unicode("") == ""


# ============================================================
# Bước 2: Lookalike — AN TOÀN
# ============================================================

class TestLookalike:
    """Chỉ chuyển số→chữ khi số NẰM TRONG từ có chữ cái."""

    def test_basic(self):
        assert map_lookalike_chars("m4y") == "may"

    def test_multiple(self):
        assert map_lookalike_chars("n9u qu4") == "ngu qua"

    def test_all_lookalike(self):
        # "m4y" chứa chữ "m" → chuyển "4" → "a"
        assert map_lookalike_chars("m4y") == "may"

    # --- AN TOÀN: giữ nguyên số đứng một mình ---

    def test_pure_number_preserved(self):
        """Số thuần túy KHÔNG bị chuyển."""
        assert map_lookalike_chars("400") == "400"
        assert map_lookalike_chars("123") == "123"
        assert map_lookalike_chars("100000") == "100000"

    def test_number_with_unit_preserved(self):
        """Số + đơn vị: digits > alpha → giữ nguyên."""
        assert map_lookalike_chars("300k") == "300k"
        assert map_lookalike_chars("400k") == "400k"
        assert map_lookalike_chars("50m") == "50m"

    def test_at_sign(self):
        assert map_lookalike_chars("m@y") == "may"

    def test_no_change_normal_text(self):
        """Văn bản bình thường không bị ảnh hưởng."""
        assert map_lookalike_chars("xin chào") == "xin chào"


# ============================================================
# Bước 3: Separator removal — BẢO THỦ
# ============================================================

class TestSeparator:
    """Chỉ xóa .-_ giữa 2 chữ cái. KHÔNG xóa khoảng trắng."""

    def test_dots(self):
        assert remove_inserted_separators("n.g.u") == "ngu"

    def test_dashes(self):
        assert remove_inserted_separators("n-g-u") == "ngu"

    def test_underscores(self):
        assert remove_inserted_separators("n_g_u") == "ngu"

    def test_mixed(self):
        assert remove_inserted_separators("m.à.y n-g.u") == "mày ngu"

    # --- AN TOÀN ---

    def test_space_preserved(self):
        """Khoảng trắng hợp lệ KHÔNG bị xóa."""
        assert remove_inserted_separators("tôi đi học") == "tôi đi học"

    def test_number_dot_preserved(self):
        """Dấu chấm giữa 2 số → giữ nguyên."""
        assert remove_inserted_separators("3.14") == "3.14"

    def test_sentence_dot_preserved(self):
        """Dấu chấm cuối câu → giữ nguyên."""
        assert remove_inserted_separators("tôi đi.") == "tôi đi."

    def test_vietnamese_diacritics(self):
        """Chữ Việt có dấu vẫn được nhận diện là chữ cái."""
        assert remove_inserted_separators("q.u.á") == "quá"


# ============================================================
# Bước 4: Collapse elongation — BẢO THỦ
# ============================================================

class TestElongation:
    """Chỉ gộp khi >= 3 ký tự giống nhau liên tiếp."""

    def test_triple(self):
        assert collapse_elongation("nàooo") == "nào"

    def test_many(self):
        assert collapse_elongation("nàoooooo") == "nào"

    def test_multiple_words(self):
        assert collapse_elongation("đẹppppp quááááá") == "đẹp quá"

    def test_diacritic_base_letter(self):
        """Ký tự có dấu + base letter lặp: quáaaaa → quá."""
        assert collapse_elongation("quáaaaa") == "quá"
        assert collapse_elongation("quáa") == "quá"

    def test_diacritic_base_letter_o(self):
        assert collapse_elongation("nàooo") == "nào"

    # --- AN TOÀN ---

    def test_double_preserved(self):
        """2 ký tự giống nhau → giữ nguyên (VD: 'oo' trong 'xoong')."""
        assert collapse_elongation("xoong") == "xoong"

    def test_normal_text_preserved(self):
        assert collapse_elongation("xin chào") == "xin chào"

    def test_exclamation_preserved(self):
        """Dấu câu lặp KHÔNG bị gộp (để bảo toàn ... hay !!!)."""
        assert collapse_elongation("trời!!!!!!") == "trời!!!!!!"


# ============================================================
# Bước 5: Dictionary lookup — AN TOÀN
# Các test dưới đây CHỈ pass khi đã chạy build_dictionary.py
# để tạo teencode_dict.json. Skip nếu dict rỗng.
# ============================================================

_has_dict = get_dict_size() > 0


@pytest.mark.skipif(not _has_dict, reason="Chưa có teencode_dict.json")
class TestTeencodeDictionary:

    # --- abbreviation_clipping ---
    def test_abbrev_ko(self):
        result = restore_teencode("ko")
        assert result == "không", f"Expected 'không', got '{result}'"

    def test_abbrev_dc(self):
        result = restore_teencode("đc")
        assert result in ("được",), f"Expected 'được', got '{result}'"

    # --- dấu câu dính liền ---
    def test_punct_trailing_comma(self):
        result = restore_teencode("ko,")
        assert result == "không,", f"Expected 'không,', got '{result}'"

    def test_punct_trailing_dot(self):
        result = restore_teencode("ko.")
        assert result == "không.", f"Expected 'không.', got '{result}'"

    def test_punct_trailing_question(self):
        result = restore_teencode("ko?")
        assert result == "không?", f"Expected 'không?', got '{result}'"

    # --- viết hoa ---
    def test_uppercase_preserved(self):
        result = restore_teencode("Ko")
        assert result == "Không", f"Expected 'Không', got '{result}'"

    # --- AN TOÀN: từ hợp lệ KHÔNG bị thay đổi ---
    def test_valid_word_preserved(self):
        """Từ tiếng Việt hợp lệ không có trong dict → giữ nguyên."""
        for word in ["mèo", "ăn", "học", "trường", "đẹp"]:
            assert restore_teencode(word) == word, f"'{word}' bị thay đổi!"

    def test_ambiguous_preserved(self):
        """Từ mơ hồ (trong blocklist) KHÔNG bị thay đổi."""
        # Những từ này phải giữ nguyên vì có nghĩa riêng
        for word in ["dậy", "con", "cho", "ma"]:
            assert restore_teencode(word) == word, f"'{word}' bị thay đổi!"


# ============================================================
# Pipeline tổng hợp
# ============================================================

class TestNormalizePipeline:
    """Test toàn bộ pipeline normalize()."""

    # --- surface_obfuscation ---
    def test_lookalike_basic(self):
        result = normalize("m4y n9u qu4")
        # Sau B2: "may ngu qua", B5 có thể thêm dấu nếu trong dict
        assert "4" not in result
        assert "9" not in result

    def test_separator_basic(self):
        result = normalize("n.g.u")
        assert result in ("ngu", "ngủ", "ngũ")  # tùy dict có entry ko

    def test_lookalike_plus_separator(self):
        result = normalize("m4.y")
        assert "4" not in result
        assert "." not in result or result.endswith(".")

    # --- expressive_lengthening ---
    def test_elongation(self):
        result = normalize("quááááá")
        assert result in ("quá", "quá")

    # --- AN TOÀN: không sửa nhầm ---
    def test_pure_numbers_safe(self):
        """Số thuần PHẢI giữ nguyên qua toàn bộ pipeline."""
        assert normalize("400") == "400"
        assert normalize("100000") == "100000"
        assert normalize("400k") == "400k"
        assert normalize("300k") == "300k"

    def test_decimal_safe(self):
        assert normalize("3.14") == "3.14"
        assert normalize("9.5") == "9.5"

    def test_clean_sentence_safe(self):
        """Câu tiếng Việt bình thường KHÔNG bị thay đổi."""
        clean = "tôi đi học ở trường"
        assert normalize(clean) == clean

    def test_sentence_with_punctuation_safe(self):
        """Dấu câu cuối câu không bị xóa."""
        text = "tôi đi học."
        result = normalize(text)
        assert result.endswith(".")

    def test_emoticon_preserved(self):
        """Emoticon giữ nguyên."""
        text = "vui quá :))"
        result = normalize(text)
        assert ":))" in result

    # --- Kết hợp nhiều loại ---
    @pytest.mark.skipif(not _has_dict, reason="Chưa có dict")
    def test_combined_obfuscation(self):
        """Kết hợp lookalike + separator + teencode."""
        result = normalize("m4.y n9.u")
        # B2: m4.y → ma.y, n9.u → ng.u
        # B3: ma.y → may, ng.u → ngu
        # B5: tra dict nếu có
        assert "4" not in result
        assert "9" not in result


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_spaces(self):
        assert normalize("   ") == ""  # split + join

    def test_single_char(self):
        result = normalize("a")
        assert result == "a"

    def test_only_punctuation(self):
        result = normalize("...")
        assert result == "..."

    def test_unicode_emoji(self):
        """Emoji không bị ảnh hưởng."""
        text = "vui quá 😊"
        result = normalize(text)
        assert "😊" in result
