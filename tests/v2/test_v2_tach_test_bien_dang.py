"""Test tách tập test biến dạng theo giọng vùng, kiểu đã thấy/chưa thấy và kịch bản."""

import json

import pytest

from src.v2.data import tach_test_bien_dang as t

GOC = [
    {"id": "a", "nhan": "DE_DOA", "giong_vung": "bac", "phan_test": "trong_phan_phoi", "kich_ban": "DD1"},
    {"id": "b", "nhan": "AN_TOAN", "giong_vung": "trung", "phan_test": "chua_thay", "kich_ban": "moi"},
]
ROWS = [
    {"id": "a1", "goc_id": "a", "nhan": "DE_DOA", "text": "x", "bang_chung": [], "bien_dang": "C1",
     "nhom_bien_dang": "da_thay"},
    {"id": "a2", "goc_id": "a", "nhan": "DE_DOA", "text": "y", "bang_chung": [], "bien_dang": "C1",
     "nhom_bien_dang": "chua_thay"},
    {"id": "b1", "goc_id": "b", "nhan": "AN_TOAN", "text": "z", "bang_chung": [], "bien_dang": "C1",
     "nhom_bien_dang": "chua_thay"},
]


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_tach_and_attributes(tmp_path):
    goc, vao, ra = tmp_path / "goc.jsonl", tmp_path / "C1.jsonl", tmp_path / "ra"
    _write(goc, GOC)
    _write(vao, ROWS)
    assert t.main(["--goc", str(goc), "--vao", str(vao), "--ra", str(ra)]) == 0
    rows = t.read_jsonl(vao)
    assert [r["giong_vung"] for r in rows] == ["bac", "bac", "trung"]
    assert [r["id"] for r in t.read_jsonl(ra / "C1_bac.jsonl")] == ["a1", "a2"]
    assert [r["id"] for r in t.read_jsonl(ra / "C1_chua_thay.jsonl")] == ["b1"]
    assert [r["id"] for r in t.read_jsonl(ra / "C1_da_thay.jsonl")] == ["a1", "a2"]
    assert rows[2]["phan_test"] == "chua_thay"
    assert not (ra / "C1_nam.jsonl").exists()
    c0 = t.read_jsonl(ra / "C0_bac.jsonl")
    assert [(r["id"], r["goc_id"], r["bien_dang"]) for r in c0] == [("a", "a", "C0")]
    assert [r["id"] for r in t.read_jsonl(ra / "C0_chua_thay.jsonl")] == ["b"]
    stats = json.loads((ra / "thong_ke.json").read_text(encoding="utf-8"))
    assert stats["C1_nam"]["so_dong"] == 0 and stats["C1_da_thay"]["so_cau_goc"] == 1


def test_label_mismatch_is_refused():
    goc = {g["id"]: g for g in GOC}
    with pytest.raises(ValueError):
        t.gan_thuoc_tinh([dict(ROWS[0], nhan="AN_TOAN")], goc)
