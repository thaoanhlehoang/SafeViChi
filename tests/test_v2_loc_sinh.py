"""Test lọc lại đoạn sinh, không gọi API."""

from src.v2.labeling import loc_sinh as ls


def _rec(i, nhan, sinh_doi=False):
    return {"id": f"s{i}", "text": f"đoạn {i}", "nhan": nhan, "kich_ban": "DD1", "sinh_doi": sinh_doi}


def test_split_keeps_only_matching_labels():
    records = [_rec(1, "DE_DOA"), _rec(2, "DE_DOA"), _rec(3, "AN_TOAN", True), _rec(4, "AN_TOAN", True)]
    checked = {
        "s1": {"nhan": "DE_DOA", "llm_do_chac": 0.9, "llm_ly_do": "dọa", "llm_loi": []},
        "s2": {"nhan": "AN_TOAN", "llm_do_chac": 0.8, "llm_ly_do": "nhẹ", "llm_loi": []},  # dọa quá nhẹ
        "s3": {"nhan": "AN_TOAN", "llm_do_chac": 0.9, "llm_ly_do": "ok", "llm_loi": []},
        # s4 không có kết quả kiểm
    }
    kept, dropped = ls.split_by_check(records, checked)
    assert [r["id"] for r in kept] == ["s1", "s3"]
    assert {r["id"]: r["ly_do_loai"] for r in dropped} == {"s2": "kiem_lai_ra_AN_TOAN", "s4": "chua_kiem_duoc"}
    assert kept[0]["nhan"] == "DE_DOA" and kept[0]["kiem_lai_nhan"] == "DE_DOA"
    stats = ls.summarize(kept, dropped)
    assert stats["giu"] == 2 and stats["loai"] == 2
    assert stats["theo_nhan"]["DE_DOA"] == {"giu": 1, "loai": 1, "ti_le_giu": 0.5}
    assert stats["theo_nhan"]["AN_TOAN(sinh_doi)"]["ti_le_giu"] == 0.5


def test_failed_check_is_dropped_not_kept():
    kept, dropped = ls.split_by_check([_rec(1, "LUA_DAO")], {"s1": {"nhan": "", "llm_loi": ["khong_co_ket_qua"]}})
    assert kept == [] and dropped[0]["ly_do_loai"] == "chua_kiem_duoc"


def test_chi_phi_parses_usage_lines_and_cost():
    from src.v2.labeling import chi_phi as cp
    log = ("xong 20/20\nToken đã dùng lượt này: 2 lượt gọi, 6,131 token vào, 1,336 token ra (trong đó 0 token suy luận)\n"
           "chạy lại\nToken đã dùng lượt này: 1 lượt gọi, 1,000 token vào, 664 token ra (trong đó 0 token suy luận)\n")
    usage = cp.parse_usage(log)
    assert usage == {"luot_goi": 3, "token_vao": 7131, "token_ra": 2000, "token_suy_luan": 0, "so_lan_chay": 2}
    money = cp.cost({"token_vao": 1_000_000, "token_ra": 1_000_000}, ty_gia=27000)
    assert money == {"usd": 0.536, "vnd": 14472}
    assert cp.parse_usage("không có dòng tổng") is None
