"""Test đọc câu test người viết, kappa, file mẫu, và loại câu sinh gần trùng câu test."""

import json
from pathlib import Path

import pytest

from src.v2.data import load_test_nguoi_viet as t
from src.v2.labeling import generate as g

ROOT = Path(__file__).resolve().parents[1]
HEADER = "text,nhan,bang_chung,kich_ban,kenh,nguoi_viet,ghi_chu,nhan_nguoi_kia\n"


def _csv(tmp_path, body, name="a.csv"):
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_sample_csv_is_valid():
    records, errors, stats = t.load([ROOT / "data" / "v2" / "raw" / "test_nguoi_viet_mau.csv"])
    assert errors == [] and len(records) == 2
    assert {r["nhan"] for r in records} == {"AN_TOAN", "LUA_DAO"}
    for r in records:
        assert r["nguon"] == "nguoi_viet" and r["cach_gan"] == "nguoi" and r["split"] == "test"


def test_row_errors_are_reported(tmp_path):
    path = _csv(tmp_path,
                'mai ra sau truong gap tao,DE_DOA,ra sau truong gap tao,DD1,nhom_lop,A,,\n'
                'câu sai nhãn nha,HATE,,moi,sms,A,,\n'
                'bằng chứng sai chỗ,DE_DOA,không có cụm này,DD1,sms,B,,\n'
                'an toàn mà có bằng chứng,AN_TOAN,an toàn,LD1,sms,B,,\n'
                'thiếu người viết nè,AN_TOAN,,LD1,sms,,,\n'
                'kênh lạ quá,AN_TOAN,,LD1,fax,A,,\n'
                'mai ra sau truong gap tao,DE_DOA,,DD1,sms,B,,\n')
    records, errors, stats = t.load([path])
    assert len(records) == 1 and records[0]["bang_chung"] == ["ra sau truong gap tao"]
    joined = "\n".join(errors)
    for needle in ("a.csv:3: nhan khong hop le", "a.csv:4: bang_chung khong co", "a.csv:5: cau AN_TOAN",
                   "a.csv:6: thieu nguoi_viet", "a.csv:7: kenh la", "a.csv:8: trung nguyen van"):
        assert needle in joined
    assert stats["so_loi"] == 6


def test_missing_column(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text("text,nhan\nx,AN_TOAN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nguoi_viet"):
        t.load([path])


def test_multiple_evidence_split_by_pipe(tmp_path):
    path = _csv(tmp_path, '"ck 200k phí giữ suất, hạn trong hôm nay",LUA_DAO,ck 200k phí giữ suất|hạn trong hôm nay,'
                          'LD10,sms,A,,LUA_DAO\n')
    records, errors, _ = t.load([path])
    assert errors == [] and records[0]["bang_chung"] == ["ck 200k phí giữ suất", "hạn trong hôm nay"]


def test_cohen_kappa():
    assert t.cohen_kappa([]) is None
    assert t.cohen_kappa([("A", "A"), ("B", "B")]) == 1.0
    assert t.cohen_kappa([("A", "A"), ("A", "A")]) == 1.0
    # ví dụ tay: po = 0.75, pe = 0.5 -> kappa = 0.5
    pairs = [("X", "X"), ("X", "X"), ("Y", "Y"), ("Y", "X")]
    assert t.cohen_kappa(pairs) == pytest.approx((0.75 - 0.5) / 0.5)


def test_kappa_and_disagreements_in_stats(tmp_path):
    path = _csv(tmp_path,
                'tao biết nhà mày rồi đấy,DE_DOA,tao biết nhà mày,DD1,sms,A,,DE_DOA\n'
                'gửi anh tấm ảnh riêng tư đi,QUAY_ROI,gửi anh tấm ảnh riêng tư,QR1,messenger_zalo,A,,DE_DOA\n'
                'tối nay leo rank không ae,AN_TOAN,,DD5,chat_game,B,,\n')
    records, errors, stats = t.load([path])
    assert errors == [] and stats["so_cau_co_nhan_nguoi_kia"] == 2
    assert len(stats["cau_bat_dong"]) == 1 and "QUAY_ROI vs DE_DOA" in stats["cau_bat_dong"][0]
    assert stats["cohen_kappa"] == pytest.approx(0.0)


def test_main_refuses_to_write_with_errors(tmp_path):
    path = _csv(tmp_path, 'câu sai nhãn nha,HATE,,moi,sms,A,,\nchào cậu nhé,AN_TOAN,,moi,sms,A,,\n')
    out = tmp_path / "out.jsonl"
    assert t.main([str(path), "--out", str(out)]) == 1 and not out.exists()
    assert t.main([str(path), "--out", str(out), "--bo-qua-loi"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_generated_near_duplicate_of_test_is_dropped(tmp_path):
    bang = g.load_kich_ban()
    cells = g.plan_cells(bang, ["DD1"], so_cau=2, co_sinh_doi=False)[:1]
    near = "Chiều nay tan học ra sau trường gặp tao, đứa nào trốn thì mai biết tay nha mày"
    other = "Tao nói rồi đấy, trưa mai cổng sau, đi một mình thôi không thì đừng trách"
    done = {cells[0].o_id: {"o_id": cells[0].o_id, "items": [
        {"text": near + "!", "nhan": "DE_DOA", "bang_chung": ["ra sau trường gặp tao"]},
        {"text": other, "nhan": "DE_DOA", "bang_chung": ["đừng trách"]},
    ], "loi": None, "tu_choi": False}}
    records, stats = g.collect(cells, done, cau_test=[near])
    assert [r["text"] for r in records] == [other]
    assert stats["cau_bi_bo"]["trung_cau_test_nguoi_viet"] == 1

    path = tmp_path / "test_nguoi_viet.jsonl"
    path.write_text(json.dumps({"text": near}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert g.load_test_nguoi_viet(path) == [near]
    assert g.load_test_nguoi_viet(tmp_path / "khong_co.jsonl") == []
