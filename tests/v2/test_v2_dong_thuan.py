"""Test quy tắc đồng thuận và đo đồng thuận người gán (không gọi API)."""

import csv
import json

from src.v2.data import dong_thuan as d
from src.v2.labeling import agreement as ag


def _r(i, nhan, nhan_goc, goc_tung_bo):
    return {"id": f"v{i}", "text": f"câu {i}", "nhan": nhan, "nhan_goc": nhan_goc,
            "nguon_goc": [{"nguon": f"b{j}", "nhan_goc": g} for j, g in enumerate(goc_tung_bo)]}


def test_rule_keeps_llm_backed_by_at_least_one_annotator():
    cases = [
        (_r(1, "XUC_PHAM", "HATE", ["HATE"]), (True, "khop_nhan_goc")),
        (_r(2, "AN_TOAN", "CLEAN", ["CLEAN", "CLEAN"]), (True, "khop_nhan_goc")),
        (_r(3, "AN_TOAN", "HATE", ["HATE", "CLEAN"]), (True, "khop_it_nhat_mot_nguoi_gan")),
        (_r(4, "AN_TOAN", "HATE", ["HATE", "OFFENSIVE"]), (False, "lech_moi_nguoi_gan")),
        (_r(5, "LUA_DAO", "CLEAN", ["CLEAN"]), (False, "lech_moi_nguoi_gan")),
        (_r(6, "", "HATE", ["HATE"]), (False, "loi_llm")),
    ]
    for record, expected in cases:
        assert d.xet_dong_thuan(record) == expected, record["id"]


def test_sample_hides_labels_and_agreement_metrics(tmp_path):
    records = d.gan_co([_r(i, "XUC_PHAM", "HATE", ["HATE"]) for i in range(10)]
                       + [_r(i, "AN_TOAN", "HATE", ["HATE"]) for i in range(10, 20)])
    sample = d.chon_mau(records, n_moi_nhom=4)
    assert len(sample) == 8 and Counter_of(sample) == {"giu": 4, "bo": 4}
    assert [r["stt"] for r in sample] == list(range(1, 9))
    a_path, b_path = tmp_path / "A.csv", tmp_path / "B.csv"
    d.write_csv(sample, a_path)
    rows = list(csv.DictReader(a_path.open(encoding="utf-8-sig")))
    assert set(rows[0]) == set(d.CSV_COLUMNS) and all(r["nhan"] == "" for r in rows)  # không lộ nhãn
    assert "XUC_PHAM" not in a_path.read_text(encoding="utf-8-sig")

    key = [{"stt": r["stt"], "id": r["id"], "nhom_kiem": r["nhom_kiem"], "nhan_llm": r["nhan"],
            "nhan_goc": r["nhan_goc"], "text": r["text"]} for r in sample]
    # A và B đều đồng ý nhãn LLM ở nhóm giữ; ở nhóm bỏ, người đứng về phía nhãn gốc (nguy hại)
    a = {k["id"]: (k["nhan_llm"] if k["nhom_kiem"] == "giu" else "XUC_PHAM") for k in key}
    b = dict(a)
    metrics, dis = ag.analyse(key, a, b)
    assert metrics["kappa_5_nhan"] == 1.0 and dis == []
    assert metrics["nhom_giu"]["llm_dung_5_nhan"]["ti_le"] == 1.0
    assert metrics["nhom_bo"]["nguoi_dong_y_nhan_goc_sach_nguy_hai"]["ti_le"] == 1.0
    b[key[0]["id"]] = "DE_DOA"
    metrics, dis = ag.analyse(key, a, b)
    assert len(dis) == 1 and metrics["kappa_5_nhan"] < 1.0


def Counter_of(sample):
    from collections import Counter
    return dict(Counter(r["nhom_kiem"] for r in sample))


def test_read_labels_reports_invalid(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("stt,id,text,nhan,bang_chung,ghi_chu\n1,v1,a,xuc_pham,,\n2,v2,b,,,\n3,v3,c,ABC,,\n",
                    encoding="utf-8-sig")
    labels, errors = ag.read_labels(path)
    assert labels == {"v1": "XUC_PHAM"} and len(errors) == 1
