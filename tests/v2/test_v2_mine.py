"""Test lọc ứng viên bằng từ khóa."""

from src.v2.data import mine_candidates as m


def test_match_keywords_accent_insensitive_and_word_bounded():
    assert m.match_keywords("Tao biết nhà mày ở đâu đấy")["DE_DOA"] == ["biet nha may"]
    assert m.match_keywords("tao biet nha may o dau")["DE_DOA"] == ["biet nha may"]
    assert "QUAY_ROI" in m.match_keywords("Gửi anh ảnh nóng đi em")
    assert m.match_keywords("Hôm nay trời đẹp") == {}
    assert m.match_keywords("ctvx") == {}  # không khớp giữa chữ


def test_mine_dedups_text():
    rows = [{"text": "nạp thẻ nhận quà", "nguon": "vihsd"}, {"text": "nạp thẻ nhận quà", "nguon": "vihos"},
            {"text": "chào bạn", "nguon": "vihsd"}]
    out, stats = m.mine(rows)
    assert len(out) == 1 and out[0]["ung_vien"] == ["LUA_DAO"]
    assert stats["so_cau_da_quet"] == 3 and stats["theo_nhan_ung_vien"] == {"LUA_DAO": 1}
