"""Unit test cho CLI suy luận cục bộ src/v2/predict.py (thuần ONNX Runtime, không nạp PyTorch)."""

from __future__ import annotations

import json
import re

import pytest

from src.shared.normalization import normalizer as v1_normalizer
from src.v2.predict import (
    DEFAULT_ONNX_DIR,
    REPO_ROOT,
    SafeViChiPredictor,
    build_parser,
    highlight_text,
    legacy_normalize_v1_with_alignment,
    main,
    normalize_input_turns,
    render_pretty,
)


def test_default_paths_and_v1_deprecation_markers():
    assert DEFAULT_ONNX_DIR.is_absolute()
    assert v1_normalizer.DEPRECATED is True
    assert "lỗi thời" in v1_normalizer.DEPRECATION_NOTICE.lower()


def test_cli_parser_defaults_raw_and_exposes_deprecated_legacy_flag():
    parser = build_parser()
    assert parser.parse_args(["xin chào"]).legacy_normalize is False
    assert parser.parse_args(["--legacy-normalize", "xin chào"]).legacy_normalize is True

    help_text = parser.format_help()
    assert "--legacy-normalize" in help_text
    assert "Lỗi thời" in help_text or "Deprecated" in help_text
    norm_flags = [opt for action in parser._actions for opt in action.option_strings if "norm" in opt]
    assert norm_flags == ["--legacy-normalize"]


def test_render_pretty_raw_and_legacy_deprecated_card(tmp_path, capsys):
    base_sample = {
        "nhan": "LUA_DAO",
        "ten_nhan": "Lừa đảo / Dụ OTP & link",
        "hanh_dong": "Cảnh báo lừa đảo, vô hiệu hóa liên kết",
        "tin_cay": 0.94,
        "probs": {"AN_TOAN": 0.02, "XUC_PHAM": 0.01, "QUAY_ROI": 0.01, "DE_DOA": 0.02, "LUA_DAO": 0.94},
        "rui_ro": 85.0,
        "bang_chung": ["mã OTP"],
        "vi_tri": [[9, 15]],
        "text": "Đừng gửi mã OTP vào link lạ",
        "thoi_gian_ms": 12.5,
    }
    raw_card = render_pretty(base_sample, use_color=False)
    assert "[LUA_DAO]" in raw_card
    assert "[[mã OTP]]" in raw_card
    assert "Phân bố:" in raw_card
    assert "Chuẩn hóa" not in raw_card

    legacy_card = render_pretty({**base_sample, "legacy_normalize": True}, use_color=False)
    assert "[Chuẩn hóa v1 · Lỗi thời]" in legacy_card
    assert "--legacy-normalize" in legacy_card

    # Kiểm tra căn viền ANSI (67 cột) cho cả mặc định và legacy
    ansi_strip = re.compile(r"\x1b\[[0-9;]*m")
    for flag in (False, True):
        sample = {**base_sample, **({"legacy_normalize": True} if flag else {})}
        colored = render_pretty(sample, use_color=True)
        assert "\033[" in colored
        plain_lines = [ansi_strip.sub("", line) for line in colored.splitlines()]
        assert len(plain_lines[0]) == len(plain_lines[-1]) == 67
        assert ("[Chuẩn hóa v1 · Lỗi thời]" in plain_lines[0]) is flag

    rc = main(["--onnx-dir", str(tmp_path / "khong_ton_tai"), "--no-color", "xin chào"])
    assert rc == 1
    assert "Lỗi:" in capsys.readouterr().err


def test_normalize_input_turns_pipe_and_window():
    assert normalize_input_turns("Lượt 1 || Lượt 2 || Lượt 3") == "Lượt 1\nLượt 2\nLượt 3"
    ten_turns = " || ".join(f"turn {i}" for i in range(1, 11))
    out = normalize_input_turns(ten_turns).splitlines()
    assert len(out) == 8
    assert out[0] == "turn 3" and out[-1] == "turn 10"


def test_legacy_normalize_v1_with_alignment_preserves_lines():
    text = "Chào bạn\nMày là đ.ồ n.g.u quáaa"
    norm_text, mapping = legacy_normalize_v1_with_alignment(text)
    assert "\n" in norm_text
    assert "đồ" in norm_text and "ngu" in norm_text
    assert len(mapping) == len(norm_text.split())


def test_highlight_text_plain_and_ansi():
    text = "Đừng gửi mã OTP vào link lạ"
    assert highlight_text(text, [[9, 15]], use_color=False) == "Đừng gửi [[mã OTP]] vào link lạ"
    colored = highlight_text(text, [[9, 15]], use_color=True)
    assert "mã OTP" in colored and "\033[" in colored


@pytest.mark.skipif(
    not (DEFAULT_ONNX_DIR / "safevichi_v2_int8.onnx").exists(),
    reason="Chưa có tệp ONNX INT8 cục bộ để kiểm thử tích hợp",
)
def test_predictor_int8_raw_default_and_legacy_v1_opt_in():
    predictor = SafeViChiPredictor(onnx_dir=DEFAULT_ONNX_DIR, threads=4)

    # 1. Mặc định chạy thẳng văn bản gốc, khớp nhãn với real_test_int8_all_predictions.jsonl & 100% verbatim
    gold_path = REPO_ROOT / "data/v2/test_day_du.jsonl"
    ref_path = REPO_ROOT / "results/v2/official_eval/dualorc_mmbert_base/test/real_test_int8_all_predictions.jsonl"
    if gold_path.exists() and ref_path.exists():
        gold_rows = [
            json.loads(line)
            for line in gold_path.read_text(encoding="utf-8-sig").splitlines()[:8]
            if line.strip()
        ]
        ref_by_id = {
            rec["id"]: rec
            for rec in (
                json.loads(line)
                for line in ref_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            )
        }
        for row in gold_rows:
            got = predictor.predict(row["text"])
            ref = ref_by_id[row["id"]]
            assert "legacy_normalize" not in got
            assert got["nhan"] == ref["nhan"], f"Lệch nhãn ở {row['id']}: {got['nhan']} vs {ref['nhan']}"
            assert all(ph in row["text"] for ph in got["bang_chung"])
            if got["nhan"] != "AN_TOAN" and ref["bang_chung"]:
                assert len(got["bang_chung"]) >= 1

    # 2. Tùy chọn lỗi thời legacy_normalize=True vẫn bảo đảm bằng chứng 100% verbatim trên văn bản gốc
    noisy = "Chào bạn || Đúng là đ.ồ ó_c ch0, học n.g.u như bò"
    res_legacy = predictor.predict(noisy, legacy_normalize=True)
    assert res_legacy["legacy_normalize"] is True
    assert res_legacy["nhan"] != "AN_TOAN"
    assert all(ph in res_legacy["text"] for ph in res_legacy["bang_chung"])

    # 3. Đầu vào rỗng hoặc chỉ chứa ký tự tàng hình trả ValueError rõ ràng
    for empty_in in ("", "   ", "\u200b\ufeff\u200c"):
        with pytest.raises(ValueError):
            predictor.predict(empty_in)


@pytest.mark.skipif(
    not (DEFAULT_ONNX_DIR / "safevichi_v2_fp32.onnx").exists(),
    reason="Chưa có tệp ONNX FP32 cục bộ để kiểm thử khớp tuyệt đối",
)
def test_predictor_fp32_exact_parity_with_saved_eval():
    gold_slices = (
        (REPO_ROOT / "data/v2/test_day_du.jsonl", 8),
        (REPO_ROOT / "data/v2/test_perturb_C1.jsonl", 2),
        (REPO_ROOT / "data/v2/test_perturb_C2.jsonl", 2),
    )
    eval_dir = REPO_ROOT / "results/v2/official_eval/dualorc_mmbert_base/test"
    ref_path = eval_dir / "real_test_all_predictions.jsonl"
    conf_tol = 1e-4
    if not (gold_slices[0][0].exists() and ref_path.exists()):
        pytest.skip("Thiếu dữ liệu hoặc kết quả FP32 tham chiếu")

    predictor = SafeViChiPredictor(onnx_dir=DEFAULT_ONNX_DIR, use_fp32=True, threads=4)
    gold_rows = []
    for slice_path, limit in gold_slices:
        if slice_path.exists():
            gold_rows.extend(
                json.loads(line)
                for line in slice_path.read_text(encoding="utf-8-sig").splitlines()[:limit]
                if line.strip()
            )
    ref_by_id = {
        rec["id"]: rec
        for rec in (
            json.loads(line)
            for line in ref_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    }
    for row in gold_rows:
        got = predictor.predict(row["text"])
        ref = ref_by_id[row["id"]]
        assert "legacy_normalize" not in got
        assert got["nhan"] == ref["nhan"]
        assert got["bang_chung"] == ref["bang_chung"]
        assert abs(got["tin_cay"] - ref["tin_cay"]) < conf_tol


@pytest.mark.skipif(
    not (DEFAULT_ONNX_DIR / "safevichi_v2_int8.onnx").exists(),
    reason="Chưa có tệp ONNX INT8 cục bộ để kiểm thử CLI tương tác & hàng loạt",
)
def test_cli_deprecation_warning_batch_and_interactive(tmp_path, monkeypatch, capsys):
    # 1. --legacy-normalize + --json: cảnh báo lỗi thời in ra stderr, stdout vẫn là JSON hợp lệ
    rc_leg = main(["--legacy-normalize", "--json", "--no-color", "--threads", "4", "Đúng là đ.ồ n.g.u"])
    assert rc_leg == 0
    cap_leg = capsys.readouterr()
    assert "Lỗi thời" in cap_leg.err
    parsed = json.loads(cap_leg.out)
    assert parsed["legacy_normalize"] is True
    assert parsed["nhan"] != "AN_TOAN"

    # 2. Batch mặc định (--file và --jsonl): không bật chuẩn hóa, không có cảnh báo lỗi thời
    txt_in = tmp_path / "msgs.txt"
    txt_out = tmp_path / "out_txt.jsonl"
    txt_in.write_text("Hôm nay trời đẹp quá bạn ơi.\nTài khoản sẽ bị khóa, bấm link nhập OTP ngay\n", encoding="utf-8")
    rc = main(["--file", str(txt_in), "--out", str(txt_out), "--no-color", "--threads", "4"])
    assert rc == 0
    assert capsys.readouterr().err == ""
    rows_txt = [json.loads(line) for line in txt_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows_txt) == 2
    assert all("legacy_normalize" not in r for r in rows_txt)

    jsonl_in = tmp_path / "msgs.jsonl"
    jsonl_out = tmp_path / "out_jsonl.jsonl"
    jsonl_in.write_text(
        json.dumps({"id": "s1", "text": "Mày là đồ ngu chết đi", "nhan": "XUC_PHAM"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rc_j = main(["--jsonl", str(jsonl_in), "--out", str(jsonl_out), "--no-color", "--threads", "4"])
    assert rc_j == 0
    rows_j = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows_j) == 1
    assert rows_j[0]["id"] == "s1" and rows_j[0]["nhan_dung"] == "XUC_PHAM"

    # 3. Tương tác (interactive): /json, /chat, /history, /undo, /clear, /quit & đầu vào rỗng không crash
    capsys.readouterr()
    scripted_inputs = iter([
        "||",
        "\u200b",
        "/json",
        "Mày là đồ ngu",
        "/chat",
        "||",
        "Chào bạn",
        "Tao biết nhà mày ở đâu rồi",
        "/history",
        "/undo",
        "/clear",
        "/quit",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(scripted_inputs))
    rc_int = main(["--interactive", "--no-color", "--threads", "4"])
    assert rc_int == 0
    out_int = capsys.readouterr().out
    assert "Cảnh báo:" in out_int
    assert "Chuẩn hóa" not in out_int
    assert '"nhan":' in out_int
