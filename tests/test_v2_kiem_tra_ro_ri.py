"""Test kiểm tra rò rỉ test ↔ train."""

import json

from src.v2.data import kiem_tra_ro_ri as k


def _row(i, text, nhan="LUA_DAO", kich_ban="LD1"):
    return {"id": i, "text": text, "nhan": nhan, "kich_ban": kich_ban}


TRAIN = [
    _row("v2_a", "ban nhan duoc 500 kim cuong mien phi, dang nhap ff-qua[.]com de nhan nha"),
    _row("v2_b", "mai ra sau truong gap tao, tao cho may biet tay", "DE_DOA", "DD1"),
    _row("v2_c", "Em oi vay 5 trieu chi can anh can cuoc giai ngan 5 phut", "LUA_DAO", "LD7"),
    _row("v2_d", "toi nay leo rank khong\nok 8h nha", "AN_TOAN", "LD1"),
]


def test_held_out_codes_read_from_yaml():
    assert "LD7" in k.held_out_codes()


def test_flags_id_scenario_exact_and_near():
    test = [
        _row("v2_a", "mot cau khac han khong lien quan gi ca"),                       # trùng id
        _row("t1", "Mai ra sau trường gặp tao, tao cho mày biết tay!!", "DE_DOA", "DD1"),  # trùng nguyên văn bỏ dấu
        _row("t2", "bạn nhận được 500 kim cương miễn phí, đăng nhập ff-qua[.]com để nhận nhé bạn"),  # gần
        _row("t3", "hom nay troi dep qua di da bong khong", "AN_TOAN", "moi"),         # sạch
    ]
    out = k.check(test, TRAIN, {"LD7"})
    rep = out["report"]
    joined = "\n".join(rep["loi"])
    assert "id test trung id train: v2_a" in joined
    assert "LD7" in joined and rep["so_cau_train_thuoc_kich_ban_giu_ngoai"] == 1
    assert "trung nguyen van" in joined
    assert {"v2_a", "v2_b"} <= set(out["drop"])
    assert rep["test_theo_muc"]["sach"] >= 1
    assert all(p["test_id"] != "t3" for p in out["pairs"])


def test_copied_line_inside_dialogue_is_caught():
    test = [_row("t9", "ê mày\nban nhan duoc 500 kim cuong mien phi, dang nhap ff-qua[.]com de nhan nha\nthật ko")]
    out = k.check(test, TRAIN, set())
    assert any(p["train_id"] == "v2_a" and p["muc"] == "trung_gan" for p in out["pairs"])


def test_cli_writes_outputs(tmp_path):
    te, tr = tmp_path / "test.jsonl", tmp_path / "train.jsonl"
    te.write_text(json.dumps(_row("t1", "mai ra sau truong gap tao, tao cho may biet tay", "DE_DOA")) + "\n",
                  encoding="utf-8")
    tr.write_text("".join(json.dumps(r) + "\n" for r in TRAIN if r["kich_ban"] != "LD7"), encoding="utf-8")
    out = tmp_path / "out"
    code = k.main(["--test", str(te), "--train", str(tr), "--out-dir", str(out)])
    assert code == 1
    assert (out / "id_train_can_bo.txt").read_text(encoding="utf-8").split() == ["v2_b"]
    clean = tmp_path / "train_sach.jsonl"
    code = k.main(["--test", str(te), "--train", str(tr), "--out-dir", str(out), "--ghi-train-sach", str(clean)])
    rows = [json.loads(l) for l in clean.read_text(encoding="utf-8").splitlines()]
    assert code == 1  # vẫn còn lỗi trùng nguyên văn được báo
    assert [r["id"] for r in rows] == ["v2_a", "v2_d"]
