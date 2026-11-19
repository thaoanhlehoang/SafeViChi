"""Test sinh câu v2 không gọi API thật: đọc kich_ban.yaml, lập ô, kiểm đầu ra, khử trùng, resume."""

import json
from collections import Counter

import pytest
import yaml

from src.v2.labeling import generate as g


@pytest.fixture(scope="module")
def bang():
    return g.load_kich_ban()


def test_kich_ban_yaml_matches_plan(bang):
    codes = [kb.ma for kb in bang.kich_ban]
    assert codes[:11] == [f"LD{i}" for i in range(1, 12)]
    assert [c for c in codes if c.startswith("DD")] == [f"DD{i}" for i in range(1, 6)]
    assert [c for c in codes if c.startswith("QR")] == [f"QR{i}" for i in range(1, 6)]
    held = [kb.ma for kb in bang.kich_ban if kb.giu_ngoai_train]
    assert held == ["LD7"]
    assert set(bang.kenh) == {"messenger_zalo", "nhom_lop", "chat_game", "binh_luan", "sms"}
    assert set(bang.do_lo) == {"lo_lieu", "an_y"}
    for kb in bang.kich_ban:
        assert len(kb.kenh) == 3 and len(kb.giong) == 4 and kb.sinh_doi


def test_plan_counts_and_held_out(bang):
    cells = g.plan_cells(bang)
    per_label = Counter(c.nhan for c in cells)
    assert per_label == {"LUA_DAO": 240, "DE_DOA": 120, "QUAY_ROI": 120, "AN_TOAN": 480}
    assert not any(c.ma == "LD7" for c in cells)
    assert len({c.o_id for c in cells}) == len(cells)
    # mỗi ô nguy hại đi liền ô sinh đôi cùng kênh, giọng, độ lộ
    assert cells[1].o_id == cells[0].o_id + "-SD" and cells[1].nhan == "AN_TOAN"
    only = g.plan_cells(bang, ["DD1"], so_cau=3, co_sinh_doi=False)
    assert len(only) == 24 and all(c.ma == "DD1" and c.so_cau == 3 for c in only)
    assert g.plan_cells(bang, ["LD7"]) == []


def _write_yaml(tmp_path, **override):
    row = {"ma": "X1", "nhan": "LUA_DAO", "mo_ta": "m", "sinh_doi": "s", "kenh": ["sms"], "giong": ["a"],
           "vi_du": []}
    row.update(override)
    data = {"kenh": {"sms": "SMS"}, "giong": {"a": "A"}, "do_lo": {"lo_lieu": "x", "an_y": "y"},
            "do_lo_sinh_doi": {"lo_lieu": "x", "an_y": "y"}, "kich_ban": [row]}
    path = tmp_path / "kb.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_load_kich_ban_rejects_bad_rows(tmp_path):
    assert g.load_kich_ban(_write_yaml(tmp_path)).kich_ban[0].ma == "X1"
    with pytest.raises(ValueError, match="nhãn"):
        g.load_kich_ban(_write_yaml(tmp_path, nhan="AN_TOAN"))
    with pytest.raises(ValueError, match="kênh/giọng"):
        g.load_kich_ban(_write_yaml(tmp_path, kenh=["fax"]))
    with pytest.raises(ValueError, match="5 câu"):
        g.load_kich_ban(_write_yaml(tmp_path, vi_du=["a"] * 6))


def test_examples_must_not_come_from_test(tmp_path):
    path = _write_yaml(tmp_path, vi_du=["Ib mình nhận Robux free nha"])
    bang = g.load_kich_ban(path)
    g.check_examples_not_in_test(bang, {"câu khác"})
    with pytest.raises(ValueError, match="tập test"):
        g.check_examples_not_in_test(bang, {"ib mình nhận robux free nha"})
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"text": "Ib mình  nhận Robux free nha", "split": "test"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    assert "ib mình nhận robux free nha" in g.load_test_texts(tmp_path)


def test_user_prompt_contents(bang):
    cells = g.plan_cells(bang, ["QR1"])
    harmful = g.build_user_prompt(cells[0], bang)
    assert "QR1" in harmful and "QUAY_ROI" in harmful and "JSON list" in harmful
    assert "học sinh THCS, THPT hoặc sinh viên" in harmful
    twin = g.build_user_prompt(cells[1], bang)
    assert "SINH ĐÔI AN TOÀN" in twin and '"nhan": "AN_TOAN"' in twin
    system = g.build_system_prompt()
    assert "QUY TẮC AN TOÀN" in system and "xxx[.]com" in system and "## 5. Quy tắc bằng chứng" in system


def test_parse_items_variants():
    items = [{"text": "a b c", "nhan": "LUA_DAO"}]
    assert g.parse_items("```json\n" + json.dumps(items) + "\n```") == items
    assert g.parse_items("Dưới đây là:\n" + json.dumps(items)) == items
    assert g.parse_items('{"text": "x y z"},\n{"text": "u v w"}') == [{"text": "x y z"}, {"text": "u v w"}]
    assert g.parse_items("Xin lỗi, tôi không thể giúp với yêu cầu này.") == []


def test_defang():
    assert g.defang("vào https://nhanqua-ff.com/abc ngay") == "vào nhanqua-ff[.]com/abc ngay"
    assert g.defang("vào nhanqua[.]com") == "vào nhanqua[.]com"
    assert g.defang("điểm 8.5 nha") == "điểm 8.5 nha"


def _cell(nhan="LUA_DAO", sinh_doi=False):
    return g.O("LD1-sms-a-lo_lieu" + ("-SD" if sinh_doi else ""), "LD1", nhan, sinh_doi, "sms", "a", "lo_lieu", 3)


def test_check_item_keeps_verbatim_evidence_and_defangs():
    checked, reason = g.check_item({
        "text": "Nhận 500 kc free, đăng nhập tại kcfree.com nha",
        "nhan": "LUA_DAO",
        "bang_chung": ["Nhận 500 kc free", "đăng nhập tại kcfree.com", "không có trong câu"],
    }, _cell())
    assert reason is None
    assert checked["text"] == "Nhận 500 kc free, đăng nhập tại kcfree[.]com nha"
    assert checked["bang_chung"] == ["Nhận 500 kc free", "đăng nhập tại kcfree[.]com"]
    assert checked["bang_chung_bi_bo"] == ["không có trong câu"] and not checked["can_duyet"]


def test_check_item_flags_lost_evidence_and_drops_bad_items():
    checked, _ = g.check_item({"text": "ib e có quà nè", "nhan": "LUA_DAO", "bang_chung": ["sai"]}, _cell())
    assert checked["can_duyet"] and checked["ly_do_duyet"] == ["mat_het_bang_chung"]
    assert g.check_item({"text": "ok", "nhan": "LUA_DAO"}, _cell())[1] == "qua_ngan"
    assert g.check_item({"text": "Dưới đây là 10 tin nhắn mẫu", "nhan": "LUA_DAO"}, _cell())[1] == "giong_tro_ly"
    assert g.check_item({"text": "a b c d", "nhan": "HATE"}, _cell())[1] == "nhan_khong_hop_le"
    assert g.check_item({"text": "a b c d", "nhan": "DE_DOA"}, _cell())[1] == "nhan_khong_khop_kich_ban"
    assert g.check_item({"nhan": "LUA_DAO"}, _cell())[1] == "khong_co_text"


def test_twin_is_an_toan_without_evidence():
    checked, reason = g.check_item(
        {"text": "t vừa nạp thẻ trên web chính thức nè", "nhan": "an_toan", "bang_chung": ["nạp thẻ"]},
        _cell("AN_TOAN", True))
    assert reason is None and checked["nhan"] == "AN_TOAN" and checked["bang_chung"] == []
    assert not checked["can_duyet"]


class FakeLLM:
    def __init__(self, refuse=()):
        self.calls = 0
        self.refuse = refuse

    def __call__(self, system, user):
        self.calls += 1
        if any(code in user for code in self.refuse):
            return "Xin lỗi, tôi không thể viết nội dung này."
        nhan = "AN_TOAN" if "SINH ĐÔI" in user else user.split("nhãn ", 1)[1].split(":", 1)[0]
        base = f"tin số {self.calls} của {nhan}"
        items = [
            {"text": f"{base} mau chuyển khoản trước nha bạn", "nhan": nhan,
             "bang_chung": [] if nhan == "AN_TOAN" else ["chuyển khoản trước"]},
            {"text": f"{base} mau chuyển khoản trước nha bạn", "nhan": nhan,
             "bang_chung": [] if nhan == "AN_TOAN" else ["chuyển khoản trước"]},
            {"text": "Lưu ý: đây là dữ liệu mẫu", "nhan": nhan},
        ]
        return json.dumps(items, ensure_ascii=False)


def test_run_collect_resume_and_refusal(tmp_path, bang):
    cells = g.plan_cells(bang, ["LD1", "QR1"], so_cau=3)[:2] + g.plan_cells(bang, ["QR1"], so_cau=3)[:1]
    raw = tmp_path / "raw.jsonl"
    meta = {"provider": "fake", "model": "m", "prompt_version": "v"}
    llm = FakeLLM(refuse=("QR1",))
    done = g.run_cells(cells, bang, system="s", complete=llm, raw_path=raw, meta=meta, log=lambda _: None)
    assert llm.calls == 3 and len(done) == 3

    records, stats = g.collect(cells, done)
    assert stats["o_bi_tu_choi"] == [cells[2].o_id]
    assert stats["cau_bi_bo"]["giong_tro_ly"] == 2
    assert stats["cau_bi_bo"]["trung_gan_dung"] == 2  # câu lặp trong mỗi ô
    assert stats["theo_nhan"] == {"AN_TOAN": 1, "LUA_DAO": 1}
    for r in records:
        assert r["nguon"] == "sinh_llm" and r["cach_gan"] == "llm" and r["split"] == "train"
        assert r["kich_ban"] == "LD1" and r["llm_model"] == "m"
    assert next(r for r in records if r["nhan"] == "LUA_DAO")["bang_chung"] == ["chuyển khoản trước"]

    # chạy lại: không gọi lại ô đã xong, kể cả ô bị từ chối
    again = FakeLLM()
    g.run_cells(cells, bang, system="s", complete=again, raw_path=raw, meta=meta, log=lambda _: None)
    assert again.calls == 0


def test_api_error_is_retried_next_run(tmp_path, bang):
    cells = g.plan_cells(bang, ["DD1"], so_cau=2)[:1]
    raw = tmp_path / "raw.jsonl"

    def broken(system, user):
        raise ConnectionError("mất mạng")

    done = g.run_cells(cells, bang, system="s", complete=broken, raw_path=raw, meta={}, log=lambda _: None)
    assert done[cells[0].o_id]["loi"]
    assert g.collect(cells, done)[1]["o_loi_api"] == [cells[0].o_id]
    llm = FakeLLM()
    done = g.run_cells(cells, bang, system="s", complete=llm, raw_path=raw, meta={}, log=lambda _: None)
    assert llm.calls == 1 and g.collect(cells, done)[1]["theo_nhan"] == {"DE_DOA": 1}
