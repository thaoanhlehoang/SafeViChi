"""Test sinh biến dạng cho test v2: giữ bằng chứng, xếp nhóm đã thấy/chưa thấy, không chạy trên train."""

import json

from src.shared.normalization.normalizer import normalize
from src.v2.data import perturb as p

REC = {"id": "v2_x", "text": "Mày không trả tiền thì tao đánh chết mày\nbiết chưa, ra cổng trường gặp tao",
       "bang_chung": ["tao đánh chết mày", "ra cổng trường gặp tao"], "nhan": "DE_DOA", "split": "test"}
SAFE = {"id": "v2_y", "text": "Link nè: shop-abc[.]com, gọi <SDT> nha", "bang_chung": [], "nhan": "AN_TOAN",
        "split": "test"}


def _build(records=(REC, SAFE), coverage=None):
    groups = p.classify(coverage, 0.01, ("go_sai",))
    return p.build(list(records), groups)


def test_every_variant_keeps_evidence_as_substring_and_changes_it():
    out, stats = _build()
    variants = [r for r in out if r["goc_id"] == "v2_x" and r["bien_dang"] != "C0"]
    assert len(variants) == len(p.C1_KIEU) + len(p.C2_KIEU) + 4
    for r in variants:
        assert len(r["bang_chung"]) == 2
        assert all(b in r["text"] for b in r["bang_chung"]), r["kieu_bien_dang"]
        assert r["text"].count("\n") == 1
        if r["kieu_bien_dang"] != "dinh_tu":
            assert r["bang_chung"] != REC["bang_chung"]  # bằng chứng luôn bị biến dạng ít nhất một từ
        assert r["id"] != REC["id"] and r["id"].startswith("v2_")


def test_protected_tokens_untouched():
    out, _ = _build()
    for r in out:
        if r["goc_id"] == "v2_y":
            assert "shop-abc[.]com" in r["text"] and "<SDT>" in r["text"]


def test_normalizer_recovers_c1_it_covers_but_not_c2():
    out, _ = _build([REC])
    by = {r["kieu_bien_dang"]: r for r in out}
    for kieu in ("chen_ky_tu", "so_thay_chu", "keo_dai"):
        assert normalize(by[kieu]["text"]) == REC["text"].replace("\n", " ") or \
            normalize(by[kieu]["text"].split("\n")[0]) == REC["text"].split("\n")[0], kieu
    for kieu in ("cyrillic_homoglyph", "zero_width", "fullwidth"):
        assert normalize(by[kieu]["text"].split("\n")[0]) != REC["text"].split("\n")[0], kieu


def test_classify_uses_train_coverage():
    rows = [{"text": "ko bik dc k, mai di hoc ko may"}, {"text": "nguuuu vãi"}, {"text": "bình thường thôi mà"}]
    cov = p.train_coverage(rows)
    assert cov["teencode"] > 0 and cov["bo_dau"] > 0 and cov["keo_dai"] > 0 and cov["chen_ky_tu"] == 0
    groups = p.classify(cov, 0.01, ("go_sai",))
    assert groups["teencode"] == "da_thay" and groups["go_sai"] == "da_thay"
    assert groups["chen_ky_tu"] == "chua_thay" and groups["cyrillic_homoglyph"] == "chua_thay"
    out, stats = _build([REC], cov)
    assert {"da_thay", "chua_thay", "tron", "goc"} == set(stats["theo_nhom"])


def test_deterministic_and_refuses_train(tmp_path):
    a, _ = _build()
    b, _ = _build()
    assert a == b
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(dict(REC, split="train"), ensure_ascii=False) + "\n", encoding="utf-8")
    assert p.main([str(src), "--out", str(tmp_path / "o.jsonl")]) == 1
    src.write_text(json.dumps(REC, ensure_ascii=False) + "\n", encoding="utf-8")
    assert p.main([str(src), "--out", str(tmp_path / "o.jsonl")]) == 0
    assert (tmp_path / "o_stats.json").exists()


def test_build_chon_two_random_kinds_per_sentence_no_original():
    groups = p.classify(None, 0.01, ("go_sai",))
    for nhom, kieu in (("C1", p.C1_KIEU), ("C2", p.C2_KIEU)):
        out, stats = p.build_chon([REC, SAFE], nhom, 2, groups)
        assert len(out) == 4 and stats["so_cau_khong_du_ban"] == 0
        for gid in ("v2_x", "v2_y"):
            rows = [r for r in out if r["goc_id"] == gid]
            assert len(rows) == 2 and len({r["kieu_bien_dang"] for r in rows}) == 2
            assert all(r["bien_dang"] == nhom and r["kieu_bien_dang"] in kieu for r in rows)
            assert all(r["text"] != (REC if gid == "v2_x" else SAFE)["text"] for r in rows)
        for r in out:
            for ev in r["bang_chung"]:
                assert ev in r["text"]
        assert out == p.build_chon([REC, SAFE], nhom, 2, groups)[0]


def test_main_nhom_option(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(REC, ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "c2.jsonl"
    assert p.main([str(src), "--nhom", "C2", "--so-ban", "2", "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {r["bien_dang"] for r in rows} == {"C2"}
