"""Test gán sơ bộ v2 không gọi API thật: kiểm đầu ra, cờ duyệt, phân tích lô, resume."""

import json

from src.v2.labeling import prelabel as p

TEXT = "VCB thông báo tài khoản của quý khách sẽ bị khóa, xác minh tại vcb-xacminh.com"


def test_validate_keeps_verbatim_evidence_and_drops_others():
    v = p.validate_output({
        "nhan": "LUA_DAO",
        "bang_chung": ["tài khoản của quý khách sẽ bị khóa", "bấm link ngay", "", 5,
                       "tài khoản của quý khách sẽ bị khóa"],
        "the": ["an_y", "khong_co"], "do_chac": 0.9, "ly_do": "giả danh ngân hàng",
    }, TEXT)
    assert v.nhan == "LUA_DAO"
    assert v.bang_chung == ["tài khoản của quý khách sẽ bị khóa"]
    assert "bấm link ngay" in v.bang_chung_bi_bo and "5" in v.bang_chung_bi_bo
    assert v.the == ["an_y"] and v.do_chac == 0.9 and v.loi == []


def test_validate_normalizes_nfd_evidence():
    v = p.validate_output({"nhan": "LUA_DAO", "bang_chung": ["sẽ bị khóa".replace("ẽ", "ẽ")],
                           "do_chac": 1}, TEXT)
    assert v.bang_chung == ["sẽ bị khóa"]


def test_validate_rejects_bad_label_and_confidence():
    v = p.validate_output({"nhan": "HATE", "bang_chung": "x", "do_chac": "cao"}, TEXT)
    assert v.nhan is None and v.do_chac is None
    assert any(e.startswith("nhan_khong_hop_le") for e in v.loi)
    assert "bang_chung_khong_phai_list" in v.loi
    assert p.validate_output({"nhan": "an_toan", "do_chac": 1.5}, TEXT).do_chac is None
    assert p.validate_output({"nhan": "an_toan", "do_chac": 1}, TEXT).nhan == "AN_TOAN"


def test_an_toan_has_no_evidence():
    v = p.validate_output({"nhan": "AN_TOAN", "bang_chung": ["xác minh"], "do_chac": 0.95}, TEXT)
    assert v.bang_chung == [] and v.bang_chung_bi_bo == ["xác minh"]


def _v(nhan, do_chac=0.9):
    return p.Validated(nhan, [], [], [], do_chac, "", [])


def test_review_reasons():
    assert p.review_reasons(_v("AN_TOAN"), "CLEAN") == []
    assert p.review_reasons(_v("XUC_PHAM"), "HATE") == []
    assert p.review_reasons(_v("AN_TOAN", 0.69), "CLEAN") == ["do_chac_thap"]
    assert p.review_reasons(_v("AN_TOAN", 0.7), "CLEAN") == []
    assert "lech_nhan_goc_sach_thanh_nguy_hai" in p.review_reasons(_v("DE_DOA"), "CLEAN")
    assert "lech_nhan_goc_nguy_hai_thanh_an_toan" in p.review_reasons(_v("AN_TOAN"), "OFFENSIVE")
    assert p.review_reasons(_v("XUC_PHAM"), "KHONG_RO") == []
    assert "loi_dau_ra" in p.review_reasons(p.Validated(None, [], [], [], None, "", ["x"]), "CLEAN")


def test_parse_batch_response():
    content = ('```json\n{"stt": 2, "nhan": "AN_TOAN"}\nrác\n{"stt": 1, "nhan": "DE_DOA"}\n'
               '{"stt": 9, "nhan": "X"}\n{hỏng\n```')
    parsed = p.parse_batch_response(content, 2)
    assert parsed[1]["nhan"] == "DE_DOA" and parsed[2]["nhan"] == "AN_TOAN"
    assert p.parse_batch_response('{"nhan": "AN_TOAN"}', 1)[1]["nhan"] == "AN_TOAN"


def test_system_prompt_uses_guide_sections_and_appendix():
    system = p.build_system_prompt()
    assert "## 2. Năm nhãn" in system and "## 5. Quy tắc bằng chứng" in system
    assert "## 6." not in system
    assert "Bạn là người gán nhãn dữ liệu an toàn tin nhắn tiếng Việt" in system
    assert '"stt"' in system


def test_user_prompt_escapes_newlines():
    prompt = p.build_user_prompt(['a\n"b"', "c"])
    assert prompt.splitlines() == ['[1] "a\\n\\"b\\""', '[2] "c"']


class FakeLLM:
    """Gán DE_DOA nếu có chữ 'đánh', còn lại AN_TOAN; bỏ qua tin có 'im lặng' khi đi theo lô."""

    def __init__(self):
        self.calls = 0

    def __call__(self, system, user):
        self.calls += 1
        lines = []
        items = user.splitlines()
        for line in items:
            stt, text = line.split(" ", 1)
            stt = int(stt.strip("[]"))
            text = json.loads(text)
            if "im lặng" in text and len(items) > 1:
                continue
            if "đánh" in text:
                lines.append({"stt": stt, "nhan": "DE_DOA", "bang_chung": ["tao đánh", "không có"],
                              "the": [], "do_chac": 0.95, "ly_do": "dọa"})
            else:
                lines.append({"stt": stt, "nhan": "AN_TOAN", "bang_chung": [], "the": [],
                              "do_chac": 0.9, "ly_do": "bình thường"})
        return "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)


def _records():
    return [
        {"id": "a", "text": "mai tao đánh mày", "nhan_goc": "CLEAN", "nguon": "vihsd"},
        {"id": "b", "text": "chào bạn", "nhan_goc": "CLEAN", "nguon": "vihsd"},
        {"id": "c", "text": "chào bạn", "nhan_goc": "CLEAN", "nguon": "vihos"},
        {"id": "d", "text": "im lặng là vàng", "nhan_goc": "HATE", "nguon": "vihos"},
    ]


def test_run_batches_dedups_text_and_resumes(tmp_path):
    cache = tmp_path / "cache.jsonl"
    llm = FakeLLM()
    meta = {"provider": "fake", "model": "m", "prompt_version": "v"}
    results = p.run(_records(), system="s", complete=llm, cache_path=cache, meta=meta,
                    batch_size=10, log=lambda _: None)
    # 1 lô cho 3 text khác nhau + 1 lần gọi lẻ cho tin bị bỏ sót trong lô
    assert llm.calls == 2
    by_id = {r["id"]: r for r in results}
    assert by_id["a"]["nhan"] == "DE_DOA" and by_id["a"]["bang_chung"] == ["tao đánh"]
    assert by_id["a"]["llm_bang_chung_bi_bo"] == ["không có"]
    assert by_id["a"]["can_duyet"] is True
    assert "lech_nhan_goc_sach_thanh_nguy_hai" in by_id["a"]["ly_do_duyet"]
    assert by_id["b"]["can_duyet"] is False and by_id["c"]["nhan"] == "AN_TOAN"
    assert by_id["d"]["ly_do_duyet"] == ["lech_nhan_goc_nguy_hai_thanh_an_toan"]
    assert all(r["cach_gan"] == "llm" and r["llm_model"] == "m" for r in results)

    # chạy lại: không gọi API thêm
    again = p.run(_records(), system="s", complete=llm, cache_path=cache, meta=meta, log=lambda _: None)
    assert llm.calls == 2 and [r["nhan"] for r in again] == [r["nhan"] for r in results]


def test_run_resumes_after_truncated_cache_line(tmp_path):
    cache = tmp_path / "cache.jsonl"
    meta = {"provider": "fake", "model": "m", "prompt_version": "v"}
    p.run(_records()[:1], system="s", complete=FakeLLM(), cache_path=cache, meta=meta, log=lambda _: None)
    with cache.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "b", "llm_ra')  # bị ngắt giữa chừng
    llm = FakeLLM()
    results = p.run(_records(), system="s", complete=llm, cache_path=cache, meta=meta, log=lambda _: None)
    assert len(results) == 4 and llm.calls >= 1
    assert {r["id"] for r in results} == {"a", "b", "c", "d"}


def test_failed_item_is_flagged_and_retried(tmp_path):
    cache = tmp_path / "cache.jsonl"
    meta = {"provider": "fake", "model": "m", "prompt_version": "v"}
    results = p.run(_records()[:1], system="s", complete=lambda s, u: "không phải json", cache_path=cache,
                    meta=meta, log=lambda _: None)
    assert results[0]["nhan"] == "" and results[0]["can_duyet"] and "khong_co_ket_qua" in results[0]["llm_loi"]
    llm = FakeLLM()
    results = p.run(_records()[:1], system="s", complete=llm, cache_path=cache, meta=meta, log=lambda _: None)
    assert llm.calls == 1 and results[0]["nhan"] == "DE_DOA"


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nSAFEVICHI_TEST_A='x'\nexport SAFEVICHI_TEST_B=y\n", encoding="utf-8")
    monkeypatch.setenv("SAFEVICHI_TEST_B", "giu")
    monkeypatch.delenv("SAFEVICHI_TEST_A", raising=False)
    p.load_dotenv(env)
    import os
    assert os.environ["SAFEVICHI_TEST_A"] == "x" and os.environ["SAFEVICHI_TEST_B"] == "giu"
    monkeypatch.delenv("SAFEVICHI_TEST_A")


def test_sample_clean_keeps_all_harmful_and_samples_clean_by_text():
    rows = [{"id": f"c{i}", "text": f"câu sạch {i}", "nhan_goc": "CLEAN"} for i in range(10)]
    rows += [{"id": "d0", "text": "câu sạch 3", "nhan_goc": "CLEAN"},       # trùng text với c3
             {"id": "h0", "text": "đồ ngu", "nhan_goc": "HATE"},
             {"id": "o0", "text": "câu sạch 5", "nhan_goc": "OFFENSIVE"}]  # text có bản nguy hại
    assert p.sample_clean(rows, None) is rows
    picked = p.sample_clean(rows, 3)
    assert picked == p.sample_clean(rows, 3)  # seed cố định
    texts = {r["text"] for r in picked}
    assert {"đồ ngu", "câu sạch 5"} <= texts
    clean_only = texts - {"đồ ngu", "câu sạch 5"}
    assert len(clean_only) == 3
    # bản ghi trùng text đi cùng nhau
    assert ("câu sạch 3" in texts) == ({"c3", "d0"} <= {r["id"] for r in picked})
    assert [r["id"] for r in picked] == [r["id"] for r in rows if r in picked]  # giữ thứ tự
    assert len({r["text"] for r in p.sample_clean(rows, 100)}) == 11
