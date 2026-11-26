"""Test tách kho test và chọn câu cần gán lại (không gọi API)."""

import json

from src.v2.data import chon_gan_lai as c
from src.v2.labeling import prelabel as p


def _r(i, nhan_goc="CLEAN", splits=(("vihsd", "train"),), text=None):
    return {"id": f"v2_{i}", "text": text or f"câu {i}", "nhan_goc": nhan_goc,
            "nguon_goc": [{"nguon": n, "split_goc": s, "nhan_goc": nhan_goc} for n, s in splits]}


def test_tach_kho_keeps_groups_together_and_respects_v1_and_llm_seen():
    records = [
        _r(1, splits=(("vihsd", "test"),)),                      # v1 chưa thấy -> test
        _r(2, splits=(("vihsd", "test"), ("vihos", "train"))),    # ViHOS train -> v1 đã thấy -> train
        _r(3, splits=(("vihsd", "test"),)),                      # cùng nhóm với 4 (v1 đã thấy) -> train
        _r(4, splits=(("vihsd", "train"),)),
        _r(5, splits=(("vihsd", "test"), ("vithsd", "train"))),   # ViTHSD train không tính -> test
        _r(6, splits=(("vihsd", "test"),)),                      # đã gửi LLM -> train
    ]
    groups = ["g1", "g2", "g34", "g34", "g5", "g6"]
    train, test = c.tach_kho(records, groups, da_gui_llm={"v2_6"})
    assert [r["id"] for r in test] == ["v2_1", "v2_5"]
    assert {r["id"] for r in train} == {"v2_2", "v2_3", "v2_4", "v2_6"}
    assert all("nhom_trung" in r for r in train + test)


def test_chon_takes_all_harmful_candidates_blacklist_and_random_clean():
    train = [_r(1, "HATE"), _r(2, "OFFENSIVE"), _r(3, text="đồ khùng vui ghê"), _r(4), _r(5), _r(6), _r(7)]
    chosen = c.chon(train, ung_vien_ids={"v2_4"}, blacklist={"khùng"}, so_clean=2)
    reasons = {r["id"]: r["ly_do_chon"] for r in chosen}
    assert reasons["v2_1"] == reasons["v2_2"] == "nhan_goc_nguy_hai"
    assert reasons["v2_3"] == "clean_co_tu_blacklist" and reasons["v2_4"] == "ung_vien_tu_khoa"
    assert sum(v == "clean_ngau_nhien" for v in reasons.values()) == 2
    again = c.chon(train, ung_vien_ids={"v2_4"}, blacklist={"khùng"}, so_clean=2)
    assert [r["id"] for r in again] == [r["id"] for r in chosen]  # seed cố định


def test_prelabel_never_sends_test_pool(tmp_path, monkeypatch, capsys):
    kho = tmp_path / "kho_test.jsonl"
    kho.write_text(json.dumps({"id": "v2_t"}) + "\n", encoding="utf-8")
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(json.dumps(_r(i) | {"nguon": "vihsd"}, ensure_ascii=False) + "\n"
                           for i in ("t", "a")), encoding="utf-8")
    monkeypatch.setattr(p, "KHO_TEST_PATH", kho)
    assert p.load_kho_test_ids(kho) == {"v2_t"}
    sent: list[str] = []
    monkeypatch.setattr(p, "run", lambda records, **kw: sent.extend(r["id"] for r in records) or [])
    monkeypatch.setattr(p, "load_kho_test_ids", lambda path=kho: {"v2_t"})
    p.main(["--input", str(inp), "--dry-run"])
    out = capsys.readouterr()
    # prompt lô đầu chỉ chứa câu ngoài kho test
    assert '[1] "câu a"' in out.out and '"câu t"' not in out.out
    assert "Bỏ 1 câu thuộc kho test" in out.err
