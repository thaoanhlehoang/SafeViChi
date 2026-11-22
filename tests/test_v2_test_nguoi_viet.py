"""Test đọc câu test người viết, kappa, file mẫu, và loại câu sinh gần trùng câu test."""

import json
from pathlib import Path

import pytest

from src.v2.data import load_test_nguoi_viet as t
from src.v2.labeling import generate as g

ROOT = Path(__file__).resolve().parents[1]
HEADER = "text,nhan,bang_chung,kich_ban,kenh,giong_vung,nguoi_viet,ghi_chu,nhan_nguoi_kia\n"


def _csv(tmp_path, body, name="a.csv"):
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_sample_csv_is_valid():
    records, errors, stats = t.load([ROOT / "data" / "v2" / "raw" / "test_nguoi_viet_mau.csv"])
    assert errors == [] and len(records) == 3
    assert {r["nhan"] for r in records} == {"AN_TOAN", "LUA_DAO"}
    assert [r["giong_vung"] for r in records] == ["chung", "chung", "trung"]
    for r in records:
        assert r["nguon"] == "nguoi_viet" and r["cach_gan"] == "nguoi" and r["split"] == "test"


def test_row_errors_are_reported(tmp_path):
    path = _csv(tmp_path,
                'mai ra sau truong gap tao,DE_DOA,ra sau truong gap tao,DD1,nhom_lop,chung,A,,\n'
                'câu sai nhãn nha,HATE,,moi,sms,chung,A,,\n'
                'bằng chứng sai chỗ,DE_DOA,không có cụm này,DD1,sms,chung,B,,\n'
                'an toàn mà có bằng chứng,AN_TOAN,an toàn,LD1,sms,chung,B,,\n'
                'thiếu người viết nè,AN_TOAN,,LD1,sms,chung,,,\n'
                'kênh lạ quá,AN_TOAN,,LD1,fax,chung,A,,\n'
                'mai ra sau truong gap tao,DE_DOA,,DD1,sms,chung,B,,\n')
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
                          'LD10,sms,chung,A,,LUA_DAO\n')
    records, errors, _ = t.load([path])
    assert errors == [] and records[0]["bang_chung"] == ["ck 200k phí giữ suất", "hạn trong hôm nay"]


def test_multiline_text_with_separator(tmp_path):
    path = _csv(tmp_path, '"ê có vở không || có || chiều ra sau trường gặp tao, mang tiền theo",DE_DOA,'
                          'ra sau trường gặp tao|mang tiền theo,DD1,nhom_lop,chung,A,,\n'
                          '"câu này || có cụm",DE_DOA,này có,moi,sms,chung,B,,\n'
                          '"lượt nói || lượt nói || lượt nói || lượt nói || lượt nói || lượt nói || lượt nói || lượt nói || lượt nói",AN_TOAN,,moi,sms,chung,B,,\n')
    records, errors, _ = t.load([path])
    assert len(records) == 1
    r = records[0]
    assert r["text"] == "ê có vở không\ncó\nchiều ra sau trường gặp tao, mang tiền theo"
    assert r["so_dong"] == 3 and "nguoi_noi" not in r
    assert any("a.csv:3: bang_chung" in e for e in errors)  # cụm vắt qua hai dòng
    assert any("a.csv:4: qua 8 dong" in e for e in errors)


def test_cohen_kappa():
    assert t.cohen_kappa([]) is None
    assert t.cohen_kappa([("A", "A"), ("B", "B")]) == 1.0
    assert t.cohen_kappa([("A", "A"), ("A", "A")]) == 1.0
    # ví dụ tay: po = 0.75, pe = 0.5 -> kappa = 0.5
    pairs = [("X", "X"), ("X", "X"), ("Y", "Y"), ("Y", "X")]
    assert t.cohen_kappa(pairs) == pytest.approx((0.75 - 0.5) / 0.5)


def test_kappa_and_disagreements_in_stats(tmp_path):
    path = _csv(tmp_path,
                'tao biết nhà mày rồi đấy,DE_DOA,tao biết nhà mày,DD1,sms,chung,A,,DE_DOA\n'
                'gửi anh tấm ảnh riêng tư đi,QUAY_ROI,gửi anh tấm ảnh riêng tư,QR1,messenger_zalo,chung,A,,DE_DOA\n'
                'tối nay leo rank không ae,AN_TOAN,,DD5,chat_game,chung,B,,\n')
    records, errors, stats = t.load([path])
    assert errors == [] and stats["so_cau_co_nhan_nguoi_kia"] == 2
    assert len(stats["cau_bat_dong"]) == 1 and "QUAY_ROI vs DE_DOA" in stats["cau_bat_dong"][0]
    assert stats["cohen_kappa"] == pytest.approx(0.0)


def test_main_refuses_to_write_with_errors(tmp_path):
    path = _csv(tmp_path, 'câu sai nhãn nha,HATE,,moi,sms,chung,A,,\nchào cậu nhé,AN_TOAN,,moi,sms,chung,A,,\n')
    out = tmp_path / "out.jsonl"
    assert t.main([str(path), "--out", str(out)]) == 1 and not out.exists()
    assert t.main([str(path), "--out", str(out), "--bo-qua-loi"]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_generated_near_duplicate_of_test_is_dropped(tmp_path):
    bang = g.load_kich_ban()
    cells = g.plan_cells(bang, ["DD1"], so_doan=2, co_sinh_doi=False)[:1]
    near = "ê mày\nChiều nay tan học ra sau trường gặp tao, đứa nào trốn thì mai biết tay nha mày"
    other = "nghe bảo m mách cô\nTao nói rồi đấy, trưa mai cổng sau, đi một mình thôi không thì đừng trách"
    done = {cells[0].o_id: {"o_id": cells[0].o_id, "items": [
        {"text": near + "!", "nhan": "DE_DOA", "bang_chung": ["ra sau trường gặp tao"]},
        {"text": other, "nhan": "DE_DOA", "bang_chung": ["đừng trách"]},
    ], "loi": None, "tu_choi": False}}
    records, stats = g.collect(cells, done, cau_test=[near])
    assert [r["text"] for r in records] == [other]
    assert stats["doan_bi_bo"]["trung_cau_test_nguoi_viet"] == 1

    path = tmp_path / "test_nguoi_viet.jsonl"
    path.write_text(json.dumps({"text": near}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert g.load_test_nguoi_viet(path) == [near]
    assert g.load_test_nguoi_viet(tmp_path / "khong_co.jsonl") == []


def test_phan_test_from_held_out(tmp_path):
    path = _csv(tmp_path,
                'vay 5 triệu chỉ cần ảnh căn cước,LUA_DAO,chỉ cần ảnh căn cước,LD7,sms,chung,A,,\n'
                'mày mách cô thì liệu hồn,DE_DOA,liệu hồn,DD6,nhom_lop,chung,A,,\n'
                'ra sau trường gặp tao,DE_DOA,ra sau trường gặp tao,DD1,nhom_lop,chung,B,,\n'
                'tình huống mới toanh nè,AN_TOAN,,moi,sms,chung,B,,\n')
    records, errors, stats = t.load([path])
    assert errors == []
    assert [r["phan_test"] for r in records] == ["chua_thay", "chua_thay", "trong_phan_phoi", "chua_thay"]
    assert stats["theo_phan_va_nhan"]["chua_thay/DE_DOA"] == 1


def test_giong_vung_required(tmp_path):
    path = _csv(tmp_path, 'tau nỏ cần mô,AN_TOAN,,moi,sms,,A,,\nmi ở mô rứa,AN_TOAN,,moi,sms,hue,A,,\n')
    records, errors, _ = t.load([path])
    assert records == [] and any("thieu giong_vung" in e for e in errors) and any("'hue'" in e for e in errors)


def test_semicolon_csv_from_excel(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_text("﻿" + HEADER.replace(",", ";") + "tau nỏ cần mô;AN_TOAN;;moi;sms;trung;A;;\n", encoding="utf-8")
    records, errors, _ = t.load([path])
    assert errors == [] and records[0]["giong_vung"] == "trung"


def test_nguon_cach_gan_and_xuat_test(tmp_path, monkeypatch):
    path = tmp_path / "c.csv"
    path.write_text(
        "stt;text;nhan;bang_chung;kich_ban;kenh;giong_vung;do_lo;nguoi_viet;nguon;cach_gan;nhan_nguoi;nhan_llm\n"
        "7;Mi chuyển 300k trước đi || Răng phải chuyển?;LUA_DAO;Mi chuyển 300k trước đi;LD7;sms;trung;lo_lieu;C;"
        "claude_viet;llm_nguoi_duyet;QUAY_ROI;LUA_DAO\n"
        "8;Tối ni đi ăn chè không;AN_TOAN;;moi;sms;trung;;C;nguoi_la;;;\n",
        encoding="utf-8")
    records, errors, _ = t.load([path])
    assert len(errors) == 1 and "nguon la" in errors[0]
    r = records[0]
    assert (r["nguon"], r["cach_gan"], r["nguoi_viet"], r["phan_test"]) == ("claude_viet", "llm_nguoi_duyet", "C",
                                                                           "chua_thay")
    assert (r["stt"], r["do_lo"], r["nhan_nguoi"], r["nhan_llm"]) == ("7", "lo_lieu", "QUAY_ROI", "LUA_DAO")

    monkeypatch.setattr(t, "TEST_DIR", tmp_path / "out")
    assert t.main([str(path), "--xuat-test", "--bo-qua-loi"]) == 0
    model = [json.loads(x) for x in (tmp_path / "out" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    full = [json.loads(x) for x in (tmp_path / "out" / "test_day_du.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [list(x) for x in model] == [["id", "text", "nhan", "bang_chung"]]
    assert model[0]["id"] == full[0]["id"] and full[0]["giong_vung"] == "trung"
    assert (tmp_path / "out" / "test_stats.json").exists()


def test_bo_test_ban_giao_khop_csv():
    """data/v2/test.jsonl phải tạo lại được đúng từ 3 file CSV trong data/v2/raw."""
    raw = ROOT / "data" / "v2" / "raw"
    records, errors, _ = t.load([raw / "test_A.csv", raw / "test_B.csv", raw / "test_C.csv"])
    assert errors == []
    expected = [{k: r[k] for k in t.MODEL_FIELDS} for r in records]
    lines = (ROOT / "data" / "v2" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == expected
