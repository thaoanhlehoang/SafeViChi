"""Test gộp nguồn v2: ghép span ViHOS, định dạng bản ghi, ViTHSD khi thiếu hoặc có."""

import json

import pytest

from src.v2.data import merge as m

REQUIRED_FIELDS = {
    "id", "text", "nhan", "bang_chung", "nguon", "nguon_id", "cach_gan", "the",
    "bien_dang", "nhom_trung", "split", "nhan_goc", "split_goc", "bang_chung_goi_y",
}


def test_mask_to_phrases_groups_consecutive_words():
    words = ["mày", "ngu", "như", "bò", "thật", "đấy", "đồ", "chó"]
    mask = [0, 1, 1, 1, 0, 0, 1, 1]
    assert m.mask_to_phrases(words, mask) == ["ngu như bò", "đồ chó"]


def test_mask_to_phrases_edges_and_empty():
    assert m.mask_to_phrases(["a", "b"], [1, 1]) == ["a b"]
    assert m.mask_to_phrases(["a", "b", "c"], [1, 0, 1]) == ["a", "c"]
    assert m.mask_to_phrases(["a", "b"], [0, 0]) == []
    assert m.mask_to_phrases([], []) == []


def test_mask_to_phrases_dedup_and_length_check():
    assert m.mask_to_phrases(["vl", "x", "vl"], [1, 0, 1]) == ["vl"]
    with pytest.raises(ValueError):
        m.mask_to_phrases(["a"], [1, 0])


def test_phrases_are_substrings_of_joined_text():
    words = ["Bấp", "bênh", "vl", "thế", "đồ", "dở"]
    mask = [0, 0, 1, 0, 1, 1]
    text = " ".join(words)
    assert all(p in text for p in m.mask_to_phrases(words, mask))


def test_make_record_format_and_stable_id():
    record = m.make_record(text="  Xìn  chào ", nguon="vihsd", nguon_id="train:1",
                           nhan_goc="CLEAN", split_goc="train")
    assert REQUIRED_FIELDS <= set(record)
    assert record["nhan"] == "" and record["cach_gan"] == "" and record["bang_chung"] == []
    assert record["text"] == "Xìn chào"  # NFC + gộp khoảng trắng
    assert record["id"].startswith("v2_")
    assert record["id"] == m.make_id("vihsd", "train:1")
    assert record["id"] != m.make_id("vihos", "train:1")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def fake_sources(tmp_path):
    vihsd = tmp_path / "vihsd"
    vihos = tmp_path / "vihos"
    vihsd.mkdir()
    vihos.mkdir()
    base = {"source_dataset": "uitnlp/vihsd", "source_split": "train"}
    _write_jsonl(vihsd / "train.jsonl", [
        {**base, "source_row_id": "1", "original_text": "câu một", "text": "cau mot", "label_original": "CLEAN"},
        {**base, "source_row_id": "1", "original_text": "câu một", "text": "kau mot", "label_original": "CLEAN"},
        {"source_dataset": "tarudesu/VOZ-HSD", "source_split": "train", "source_row_id": "9",
         "original_text": "voz", "label_original": "HATE"},
    ])
    _write_jsonl(vihsd / "validation.jsonl", [
        {**base, "source_row_id": "2", "original_text": "đồ ngu", "label_original": "OFFENSIVE"},
    ])
    _write_jsonl(vihsd / "test.jsonl", [])
    _write_jsonl(vihos / "train.jsonl", [{"words": ["đồ", "ngu"], "span_mask": [0, 1], "label": "HATE"}])
    _write_jsonl(vihos / "validation.jsonl", [])
    _write_jsonl(vihos / "test.jsonl", [{"words": ["ok"], "span_mask": [0], "label": "CLEAN"}])
    return vihsd, vihos, tmp_path / "vithsd_missing", tmp_path / "vihsd_raw_missing"


def test_merge_skips_voz_dedups_variants_and_handles_missing_vithsd(fake_sources):
    records, warnings = m.merge(*fake_sources)
    assert any("ViHSD gốc" in w for w in warnings)
    assert [r["nguon"] for r in records] == ["vihsd", "vihsd", "vihos", "vihos"]
    assert all(r["nguon"] != "voz" and "voz" not in r["text"] for r in records)
    assert records[0]["nguon_id"] == "train:1" and records[0]["text"] == "câu một"
    assert records[1]["nhan_goc"] == "OFFENSIVE"
    assert records[2]["bang_chung_goi_y"] == ["ngu"]
    assert records[3]["split_goc"] == "test"
    assert any("ViTHSD" in w for w in warnings)
    stats = m.compute_stats(records)
    assert stats["theo_nguon"] == {"vihos": 2, "vihsd": 2}
    assert stats["so_ban_ghi_trung_text_chinh_xac"] == 1  # "đồ ngu" có ở cả hai nguồn


def test_merge_prefers_raw_vihsd_with_all_splits(fake_sources):
    vihsd, vihos, vithsd, raw = fake_sources
    raw.mkdir()
    (raw / "train.csv").write_text("free_text,label_id\ncâu không,0\ncâu một,0\n", encoding="utf-8")
    (raw / "dev.csv").write_text("free_text,label_id\nđồ ngu,1\n", encoding="utf-8")
    (raw / "test.csv").write_text("free_text,label_id\n  ,2\nchết đi,2\n", encoding="utf-8")
    records, warnings = m.merge(vihsd, vihos, vithsd, raw)
    vihsd_records = [r for r in records if r["nguon"] == "vihsd"]
    assert [r["nguon_id"] for r in vihsd_records] == ["train:0", "train:1", "validation:0", "test:1"]
    assert [r["nhan_goc"] for r in vihsd_records] == ["CLEAN", "CLEAN", "OFFENSIVE", "HATE"]
    # id của split train trùng cách đánh số v1_2 (source_row_id) nên không đổi
    assert vihsd_records[1]["id"] == m.make_id("vihsd", "train:1")
    assert not any("ViHSD gốc" in w for w in warnings)


def test_collapse_by_text_keeps_one_record_per_text():
    a = m.make_record(text="đồ ngu", nguon="vihsd", nguon_id="train:1", nhan_goc="CLEAN", split_goc="train")
    b = m.make_record(text="đồ  ngu", nguon="vihos", nguon_id="train:0", nhan_goc="HATE", split_goc="train",
                      bang_chung_goi_y=["ngu"])
    c = m.make_record(text="đồ ngu", nguon="vithsd", nguon_id="test:3", nhan_goc="OFFENSIVE", split_goc="test",
                      nhan_goc_chi_tiet={"individual": "2"})
    d = m.make_record(text="chào", nguon="vihsd", nguon_id="train:2", nhan_goc="CLEAN", split_goc="train")
    out = m.collapse_by_text([a, b, c, d])
    assert len(out) == 2
    first = out[0]
    assert first["id"] == a["id"] and first["nguon"] == "vihsd"
    assert first["nhan_goc"] == "HATE" and first["nhan_goc_bat_dong"] is True
    assert first["bang_chung_goi_y"] == ["ngu"]
    assert [o["nguon"] for o in first["nguon_goc"]] == ["vihsd", "vihos", "vithsd"]
    assert first["nguon_goc"][2]["nhan_goc_chi_tiet"] == {"individual": "2"}
    assert out[1]["nhan_goc"] == "CLEAN" and out[1]["nhan_goc_bat_dong"] is False


def test_lfs_pointer_is_reported(tmp_path):
    for split in m.SPLITS:
        (tmp_path / f"{split}.jsonl").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        list(m.iter_vihsd(tmp_path))


def test_vithsd_row_severity():
    record = m.vithsd_row_to_record(
        {"content": "bọn này", "individual": "clean", "groups": "offensive", "politics": "0"}, "train", 3)
    assert record["nhan_goc"] == "OFFENSIVE" and record["nguon_id"] == "train:3"
    combined = m.vithsd_row_to_record({"content": "x", "labels": "individual#hate|groups#clean"}, "test", 0)
    assert combined["nhan_goc"] == "HATE"
    # mã số ViTHSD: 0 không nhắm tới, 1 clean, 2 offensive, 3 hate
    cols = ("individual", "groups", "religion/creed", "race/ethnicity", "politics")
    for values, expected in [("00000", "CLEAN"), ("10000", "CLEAN"), ("21000", "OFFENSIVE"), ("12030", "HATE")]:
        row = {"Unnamed: 0": "7", "content": "x", **dict(zip(cols, values))}
        record = m.vithsd_row_to_record(row, "train", 0)
        assert record["nhan_goc"] == expected and record["nguon_id"] == "train:7"
    weird = m.vithsd_row_to_record({"content": "x", "nhan": "khong ro"}, "test", 1)
    assert weird["nhan_goc"] == "KHONG_RO"
    assert m.vithsd_row_to_record({"content": "  ", "individual": "hate"}, "train", 0) is None
    with pytest.raises(ValueError):
        m.vithsd_row_to_record({"foo": "bar"}, "train", 0)


def test_vithsd_csv_loaded_when_present(tmp_path):
    (tmp_path / "train.xlsx.csv").write_text("content,individual\nchào,clean\n", encoding="utf-8")
    records = list(m.iter_vithsd(tmp_path))
    assert len(records) == 1 and records[0]["nguon"] == "vithsd" and records[0]["nhan_goc"] == "CLEAN"
