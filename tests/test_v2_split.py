"""Test ghép tập train (không gọi API)."""

from src.v2.data import split as s


def _that(i, nhan, dong_thuan=True, ly_do_chon="clean_ngau_nhien", text=None):
    return {"id": f"t{i}", "text": text or f"câu thật số {i} khá dài để khỏi trùng", "nhan": nhan,
            "bang_chung": [] if nhan == "AN_TOAN" else ["câu thật"], "nguon": "vihsd", "nguon_id": str(i),
            "cach_gan": "llm", "the": [], "nhan_goc": "CLEAN", "dong_thuan": dong_thuan,
            "ly_do_chon": ly_do_chon, "ly_do_dong_thuan": "khop_nhan_goc"}


def _sinh(i, nhan, kb="LD1"):
    return {"id": f"s{i}", "text": f"đoạn sinh {i} dòng một\ndòng hai của đoạn {i}", "nhan": nhan,
            "bang_chung": [] if nhan == "AN_TOAN" else ["dòng hai"], "nguon": "sinh_llm", "nguon_id": str(i),
            "cach_gan": "llm", "the": [], "kich_ban": kb, "sinh_doi": nhan == "AN_TOAN"}


def test_build_respects_rules(monkeypatch):
    monkeypatch.setattr(s, "SO_AN_TOAN_THAT", 3)
    monkeypatch.setattr(s, "SO_AN_TOAN_SINH", 2)
    monkeypatch.setattr(s, "SO_XUC_PHAM", 3)
    that = ([_that(i, "AN_TOAN") for i in range(5)] + [_that(5, "AN_TOAN", ly_do_chon="clean_co_tu_blacklist")]
            + [_that(i, "XUC_PHAM") for i in range(10, 15)] + [_that(20, "DE_DOA"), _that(21, "LUA_DAO", False)])
    sinh = [_sinh(i, "QUAY_ROI") for i in range(3)] + [_sinh(10, "XUC_PHAM")] + \
           [_sinh(20 + i, "AN_TOAN", kb) for i, kb in enumerate(["LD1", "LD1", "DD1", "DD1"])]
    kho_test = [{"id": "k1", "text": "câu thật số 0 khá dài để khỏi trùng"}]  # gần trùng t0
    train, stats = s.build(that, sinh, kho_test)
    ids = {r["id"] for r in train}
    assert "t21" not in ids                       # dong_thuan=false không vào train
    assert "t5" in ids                            # ca khó AN_TOAN được ưu tiên
    assert "t0" not in ids                        # gần trùng kho test thì bỏ
    assert {r["meta"]["kich_ban"] for r in train if r["id"].startswith("s2")} == {"LD1", "DD1"}  # chia đều kịch bản
    assert sum(r["nhan"] == "XUC_PHAM" for r in train) == 3   # 1 sinh + 2 thật
    assert all(r["split"] == "train" and r["bien_dang"] == "C0" for r in train)
    assert all(r["bang_chung"] == [] for r in train if r["nhan"] == "AN_TOAN")
    assert s.validate(train) == []
    assert stats["tong"] == len(train)


def test_main_writes_full_and_slim(tmp_path, monkeypatch):
    import json

    that = [_that(i, "AN_TOAN") for i in range(3)] + [_that(10, "XUC_PHAM")]
    sinh = [_sinh(1, "DE_DOA")]
    for name, rows in (("that.jsonl", that), ("sinh.jsonl", sinh), ("kho.jsonl", [])):
        s.write_jsonl(tmp_path / name, rows)
    monkeypatch.setattr(s, "THAT_PATH", tmp_path / "that.jsonl")
    monkeypatch.setattr(s, "SINH_PATH", tmp_path / "sinh.jsonl")
    monkeypatch.setattr(s, "KHO_TEST_PATH", tmp_path / "kho.jsonl")
    monkeypatch.setattr(s, "DATA_V2", tmp_path)
    assert s.main(["--out", str(tmp_path / "train.jsonl"), "--full", str(tmp_path / "so_sanh.jsonl")]) == 0

    full = s.read_jsonl(tmp_path / "so_sanh.jsonl")
    slim = s.read_jsonl(tmp_path / "train.jsonl")
    assert [r["id"] for r in full] == [r["id"] for r in slim]          # cùng dòng, cùng thứ tự
    assert all(set(r) == set(s.TRAIN_FIELDS) for r in slim)
    assert all("meta" in r and "nhom_trung" in r for r in full)
    assert set(json.loads((tmp_path / "CHECKSUMS.json").read_text(encoding="utf-8"))) == {"so_sanh.jsonl",
                                                                                          "train.jsonl"}


def test_validate_catches_bad_evidence():
    bad = s.to_v2(_sinh(1, "DE_DOA") | {"bang_chung": ["một\ndòng"]}, {"loai": "sinh"})
    assert s.validate([bad])
