"""Test sinh hội thoại v2 không gọi API thật: đọc kich_ban.yaml, lập ô, kiểm đầu ra, khử trùng, resume."""

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
    assert [kb.ma for kb in bang.kich_ban if kb.giu_ngoai_train] == ["LD7"]
    assert set(bang.kenh) == {"messenger_zalo", "nhom_lop", "chat_game", "binh_luan", "sms"}
    assert set(bang.do_lo) == {"lo_lieu", "an_y"}
    assert set(bang.kieu_lo) == {"ngay_dau", "lo_dan", "mot_dong"}
    assert set(bang.kenh_mot_dong) == {"binh_luan", "sms"}
    for kb in bang.kich_ban:
        assert len(kb.kenh) == 3 and len(kb.giong) == 4 and kb.sinh_doi


def test_plan_counts_kieu_lo_and_held_out(bang):
    cells = g.plan_cells(bang)
    assert len(cells) == 800  # không vượt ngân sách lượt gọi
    assert Counter(c.nhan for c in cells) == {"LUA_DAO": 160, "DE_DOA": 120, "QUAY_ROI": 120, "AN_TOAN": 400}
    assert bang.muc_tieu_doan == {"LUA_DAO": 800, "DE_DOA": 600, "QUAY_ROI": 600}
    ld = [c for c in cells if c.nhan == "LUA_DAO"]
    assert sum(c.so_doan for c in ld) == 800
    assert Counter(c.do_lo for c in ld) == {"lo_lieu": 80, "an_y": 80}
    assert set(Counter(c.ma for c in ld).values()) == {16}
    # chọn ô không phụ thuộc --kich-ban
    assert [c.o_id for c in g.plan_cells(bang, ["LD3"])] == [c.o_id for c in cells if c.ma == "LD3"]
    assert not any(c.ma == "LD7" for c in cells)
    assert len({c.o_id for c in cells}) == len(cells)
    for c in cells:
        assert (c.kieu_lo == "mot_dong") == (c.kenh in bang.kenh_mot_dong)
        assert c.so_doan == g.DEFAULT_SO_DOAN
    multi = Counter(c.kieu_lo for c in cells if c.kieu_lo != "mot_dong")
    assert multi["ngay_dau"] == multi["lo_dan"]
    twin = cells[1]
    assert twin.o_id == cells[0].o_id + "-SD" and twin.nhan == "AN_TOAN" and twin.kieu_lo == cells[0].kieu_lo
    only = g.plan_cells(bang, ["DD1"], so_doan=3, co_sinh_doi=False)
    assert len(only) == 24 and all(c.ma == "DD1" and c.so_doan == 3 for c in only)
    assert g.plan_cells(bang, ["LD7"]) == []


def _write_yaml(tmp_path, **override):
    row = {"ma": "X1", "nhan": "LUA_DAO", "mo_ta": "m", "sinh_doi": "s", "kenh": ["sms"], "giong": ["a"],
           "vi_du": []}
    row.update(override)
    data = {"kenh": {"sms": "SMS"}, "giong": {"a": "A"}, "do_lo": {"lo_lieu": "x", "an_y": "y"},
            "do_lo_sinh_doi": {"lo_lieu": "x", "an_y": "y"},
            "kieu_lo": {"ngay_dau": "a", "lo_dan": "b", "mot_dong": "c"}, "kenh_mot_dong": ["sms"],
            "kich_ban": [row]}
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
    bang = g.load_kich_ban(_write_yaml(tmp_path, vi_du=["Ib mình nhận Robux free nha"]))
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
    assert "QR1" in harmful and "QUAY_ROI" in harmful and "JSON list" in harmful and "nguoi_noi" not in harmful
    assert "từ 2 đến 6 dòng" in harmful and "Kiểu lộ" in harmful
    twin = g.build_user_prompt(cells[1], bang)
    assert "SINH ĐÔI AN TOÀN" in twin and '"nhan": "AN_TOAN"' in twin and "Vị trí chữ dễ nhầm" in twin
    one_line = next(c for c in cells if c.kieu_lo == "mot_dong")
    assert "đúng 1 dòng" in g.build_user_prompt(one_line, bang)
    system = g.build_system_prompt()
    assert "QUY TẮC AN TOÀN" in system and "xxx[.]com" in system and "## 5. Quy tắc bằng chứng" in system
    assert "TOÀN BỘ LƯỢT NÓI CỦA MỘT NGƯỜI" in system and "LUÂN PHIÊN" in system and "DE_DOA > LUA_DAO > QUAY_ROI > XUC_PHAM > AN_TOAN" in system


def test_parse_items_variants():
    items = [{"text": "a\nb c", "nhan": "LUA_DAO"}]
    assert g.parse_items("```json\n" + json.dumps(items) + "\n```") == items
    assert g.parse_items("Dưới đây là:\n" + json.dumps(items)) == items
    assert g.parse_items('{"text": "x y z"},\n{"text": "u v w"}') == [{"text": "x y z"}, {"text": "u v w"}]
    assert g.parse_items("Xin lỗi, tôi không thể giúp với yêu cầu này.") == []


def test_defang():
    assert g.defang("vào https://nhanqua-ff.com/abc ngay") == "vào nhanqua-ff[.]com/abc ngay"
    assert g.defang("vào nhanqua[.]com") == "vào nhanqua[.]com"
    assert g.defang("điểm 8.5 nha") == "điểm 8.5 nha"


def test_clean_conversation_keeps_lines_and_strips_speakers():
    text, stripped = g.clean_conversation("A: ê m ơi\n\nB:  sao   thế\nNgười 1: vào kc.com nhận quà\n2 - 3 ngày nữa nha")
    assert text == "ê m ơi\nsao thế\nvào kc[.]com nhận quà\n2 - 3 ngày nữa nha"
    assert stripped == 3
    assert g.clean_conversation(["dòng một", "dòng hai"])[0] == "dòng một\ndòng hai"


def _cell(nhan="LUA_DAO", sinh_doi=False, kieu_lo="ngay_dau"):
    return g.O("LD1-chat_game-a-lo_lieu-" + kieu_lo + ("-SD" if sinh_doi else ""), "LD1", nhan, sinh_doi,
               "chat_game", "a", "lo_lieu", kieu_lo, 3)


def test_check_item_conversation_with_evidence_per_line():
    checked, reason = g.check_item({
        "text": "ê bro\nB: gì\nnhận 500 kc free, đăng nhập tại kcfree.com nha\nnhanh k hết",
        "nhan": "LUA_DAO",
        "bang_chung": ["đăng nhập tại kcfree.com", "nhanh k hết", "gì\nnhận 500 kc", "không có trong câu"],
    }, _cell())
    assert reason is None
    assert checked["text"] == "ê bro\ngì\nnhận 500 kc free, đăng nhập tại kcfree[.]com nha\nnhanh k hết"
    assert checked["bo_tien_to_dau_dong"] == 1
    assert checked["bang_chung"] == ["đăng nhập tại kcfree[.]com", "nhanh k hết"]
    assert set(checked["bang_chung_bi_bo"]) == {"gì nhận 500 kc", "không có trong câu"}
    assert not checked["can_duyet"]


def test_check_item_drops_bad_items():
    ok = {"text": "ib e có quà nè\nthật k\nthật", "nhan": "LUA_DAO"}
    checked, _ = g.check_item({**ok, "bang_chung": ["sai"]}, _cell())
    assert checked["can_duyet"] and checked["ly_do_duyet"] == ["mat_het_bang_chung"]
    assert g.check_item({**ok, "nguoi_noi": ["A", "C"]}, _cell())[1] is None  # trường thừa bị bỏ qua
    assert g.check_item({**ok, "text": "chỉ một dòng thôi nha"}, _cell())[1] == "sai_so_dong"
    assert g.check_item({**ok, "text": "\n".join(["dòng nè bạn"] * 9)}, _cell())[1] == "sai_so_dong"
    assert g.check_item(ok, _cell(kieu_lo="mot_dong"))[1] == "sai_so_dong"
    assert g.check_item({"text": "ok", "nhan": "LUA_DAO"}, _cell())[1] == "qua_ngan"
    assert g.check_item({**ok, "text": "Dưới đây là 5 đoạn\nmẫu nha\nok"}, _cell())[1] == "giong_tro_ly"
    assert g.check_item({**ok, "nhan": "HATE"}, _cell())[1] == "nhan_khong_hop_le"
    assert g.check_item({**ok, "nhan": "DE_DOA"}, _cell())[1] == "nhan_khong_khop_kich_ban"
    assert g.check_item({"nhan": "LUA_DAO"}, _cell())[1] == "khong_co_text"


def test_one_line_item():
    checked, reason = g.check_item({"text": "Nhận Robux free tại rb.com", "nhan": "LUA_DAO",
                                    "bang_chung": ["Nhận Robux free"]}, _cell(kieu_lo="mot_dong"))
    assert reason is None and checked["text"] == "Nhận Robux free tại rb[.]com"


def test_twin_is_an_toan_without_evidence():
    checked, reason = g.check_item(
        {"text": "t vừa nạp thẻ trên web chính thức nè\nngon, leo rank thôi",
         "nhan": "an_toan", "bang_chung": ["nạp thẻ"]}, _cell("AN_TOAN", True))
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
        one = "đúng 1 dòng" in user
        base = f"đoạn số {self.calls} của {nhan} mau chuyển khoản trước nha bạn"
        text = base if one else f"ê bạn ơi\n{base}\nok"
        evidence = [] if nhan == "AN_TOAN" else ["chuyển khoản trước"]
        items = [
            {"text": text, "nhan": nhan, "bang_chung": evidence},
            {"text": text, "nhan": nhan, "bang_chung": evidence},
            {"text": "Lưu ý: đây là dữ liệu mẫu\nok", "nhan": nhan},
        ]
        return json.dumps(items, ensure_ascii=False)


def test_run_collect_resume_and_refusal(tmp_path, bang):
    cells = g.plan_cells(bang, ["LD1"], so_doan=3)[:2] + g.plan_cells(bang, ["QR1"], so_doan=3)[:1]
    raw = tmp_path / "raw.jsonl"
    meta = {"provider": "fake", "model": "m", "prompt_version": "v"}
    llm = FakeLLM(refuse=("QR1",))
    done = g.run_cells(cells, bang, system="s", complete=llm, raw_path=raw, meta=meta, log=lambda _: None)
    assert llm.calls == 3 and len(done) == 3

    records, stats = g.collect(cells, done)
    assert stats["o_bi_tu_choi"] == [cells[2].o_id]
    assert stats["doan_bi_bo"]["giong_tro_ly"] == 2
    assert stats["doan_bi_bo"]["trung_gan_dung"] == 2  # đoạn lặp trong mỗi ô
    assert stats["theo_nhan"] == {"AN_TOAN": 1, "LUA_DAO": 1}
    for r in records:
        assert r["nguon"] == "sinh_llm" and r["cach_gan"] == "llm" and r["split"] == "train"
        assert r["kich_ban"] == "LD1" and r["llm_model"] == "m"
        assert r["so_dong"] == len(r["text"].split("\n")) and "nguoi_noi" not in r
    harmful = next(r for r in records if r["nhan"] == "LUA_DAO")
    assert harmful["bang_chung"] == ["chuyển khoản trước"] and "\n" in harmful["text"]

    again = FakeLLM()
    g.run_cells(cells, bang, system="s", complete=again, raw_path=raw, meta=meta, log=lambda _: None)
    assert again.calls == 0  # không gọi lại ô đã xong, kể cả ô bị từ chối


def test_api_error_is_retried_next_run(tmp_path, bang):
    cells = g.plan_cells(bang, ["DD1"], so_doan=2)[:1]
    raw = tmp_path / "raw.jsonl"

    def broken(system, user):
        raise ConnectionError("mất mạng")

    done = g.run_cells(cells, bang, system="s", complete=broken, raw_path=raw, meta={}, log=lambda _: None)
    assert done[cells[0].o_id]["loi"]
    assert g.collect(cells, done)[1]["o_loi_api"] == [cells[0].o_id]
    llm = FakeLLM()
    done = g.run_cells(cells, bang, system="s", complete=llm, raw_path=raw, meta={}, log=lambda _: None)
    assert llm.calls == 1 and g.collect(cells, done)[1]["theo_nhan"] == {"DE_DOA": 1}
