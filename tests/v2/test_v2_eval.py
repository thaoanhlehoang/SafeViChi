"""Kiểm thử tự động cho mô-đun đánh giá SafeViChi v2 (src/v2/eval/).

Gồm toàn bộ các ví dụ chuẩn trong docs/v2/DO_DO_BANG_CHUNG.md, chỉ số phân loại,
lát cắt in-memory (thay thế 15 file cắt lát), thống kê Bootstrap/McNemar, và độ tin cậy đầu ra LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v2.eval import (
    LABELS,
    compare_systems,
    evaluate_bundle,
    evaluate_classification,
    evaluate_evidence,
    evaluate_output_reliability,
    score_evidence_item,
)
from src.v2.eval import evaluate as cli

ROOT = Path(__file__).resolve().parents[2]


def test_evidence_example_1_from_spec():
    """Ví dụ 1 trong docs/v2/DO_DO_BANG_CHUNG.md (mục 4)."""
    text = "sao m ngu như bò thế. t đã nói nhiều lần rồi mà"
    gold = ["[ngu] như bò"]
    cases = [
        (["ngu"], 1.00, 0.50),
        (["m ngu"], 1.00, 0.40),
        (["sao m ngu như bò thế"], 1.00, 0.67),
        (["ngu như bò thế. t đã"], 0.80, 0.67),
        ([text], 0.63, 0.38),
        (["như bò"], 0.00, 0.80),
    ]
    for pred, want_intent, want_word in cases:
        res = score_evidence_item(text, gold, pred, margin=2)
        assert round(res["f1_theo_y"]["f1"], 2) == want_intent, f"pred={pred}: {res}"
        assert round(res["f1_muc_tu"]["f1"], 2) == want_word, f"pred={pred}: {res}"


def test_evidence_example_2_from_spec():
    """Ví dụ 2 trong docs/v2/DO_DO_BANG_CHUNG.md (mục 4): nhiều từ lõi trong một cụm."""
    text = "mình biết bạn học trường nào rồi đấy"
    gold = ["mình [biết] bạn học [trường nào] rồi đấy"]
    cases = [
        (["biết bạn học trường nào"], 1.00, 0.77),
        (["trường nào"], 0.80, 0.40),
        (["rồi đấy"], 0.00, 0.40),
    ]
    for pred, want_intent, want_word in cases:
        res = score_evidence_item(text, gold, pred, margin=2)
        assert round(res["f1_theo_y"]["f1"], 2) == want_intent, f"pred={pred}: {res}"
        assert round(res["f1_muc_tu"]["f1"], 2) == want_word, f"pred={pred}: {res}"


def test_evidence_example_3_multiline_from_spec():
    """Ví dụ 3 trong docs/v2/DO_DO_BANG_CHUNG.md (hội thoại 8 dòng, 69 từ)."""
    text = "\n".join(
        [
            "ê m là đứa đăng bài nói xấu t trên group lớp à",
            "ừ thì sao, t nói đúng mà",
            "mày ngu như bò thế. t đã nhắn riêng bảo gỡ rồi mà ko nghe",
            "kệ t",
            "ok vậy mai tan học m ra cổng sau trường gặp t",
            "để làm gì",
            "ra thì biết. ko ra thì t đăng hết ảnh của m lên confession",
            "ừ để xem",
        ]
    )
    gold = ["[ngu] như bò", "ra thì biết", "t [đăng hết ảnh] của m lên [confession]"]

    # Gọn, đủ 3 ý -> 1.00
    r_full = score_evidence_item(text, gold, ["ngu như bò", "ra thì biết", "đăng hết ảnh của m lên confession"])
    assert round(r_full["f1_theo_y"]["f1"], 2) == 1.00

    # Đủ ý, lố nhẹ (chỉ "ko" vượt quá 2 từ) -> ~0.97
    r_slight = score_evidence_item(
        text,
        gold,
        ["mày ngu như bò thế", "ra thì biết", "ko ra thì t đăng hết ảnh của m lên confession"],
    )
    assert r_slight["f1_theo_y"]["f1"] >= 0.90

    # Tô cả đoạn 8 dòng -> bị phạt nặng (< 0.50)
    r_all = score_evidence_item(text, gold, [text])
    assert round(r_all["f1_theo_y"]["f1"], 2) == 0.41

    # Tô sai chỗ (dòng 1, dòng 5) -> 0
    r_wrong = score_evidence_item(text, gold, ["nói xấu t", "ra cổng sau trường"])
    assert r_wrong["f1_theo_y"]["f1"] == 0.0


def test_evidence_dataset_metrics_non_verbatim_and_safe_false_highlight():
    records = [
        {"id": "1", "text": "mày ngu như bò", "nhan": "XUC_PHAM", "bang_chung": ["ngu như bò"]},
        {"id": "2", "text": "t sẽ đăng ảnh của m", "nhan": "DE_DOA", "bang_chung": ["đăng ảnh của m"]},
        {"id": "3", "text": "mã OTP của bạn là 123456", "nhan": "AN_TOAN", "bang_chung": []},
        {"id": "4", "text": "hôm nay trời đẹp quá", "nhan": "AN_TOAN", "bang_chung": []},
    ]
    preds = [
        {"id": "1", "nhan": "XUC_PHAM", "bang_chung": ["ngu như bò", "cụm bịa không có trong câu"]},
        {"id": "2", "nhan": "DE_DOA", "bang_chung": []},
        {"id": "3", "nhan": "AN_TOAN", "bang_chung": ["mã OTP"]},  # tô oan AN_TOAN
        {"id": "4", "nhan": "AN_TOAN", "bang_chung": []},
    ]
    ev = evaluate_evidence(records, preds)
    assert ev["n_harmful_evaluated"] == 2
    assert ev["n_safe_evaluated"] == 2
    assert ev["safe_false_highlight_rate"] == 0.5
    assert ev["harmful_empty_evidence_rate"] == 0.5
    assert ev["total_pred_phrases"] == 3
    assert ev["total_non_verbatim_phrases"] == 1
    assert ev["non_verbatim_phrase_rate"] == round(1 / 3, 4)


def test_classification_llm_refusal_and_calibration():
    records = [
        {"id": "a", "text": "tốt lắm", "nhan": "AN_TOAN", "bang_chung": []},
        {"id": "b", "text": "mày ngu quá", "nhan": "XUC_PHAM", "bang_chung": ["ngu quá"]},
        {"id": "c", "text": "tao biết nhà mày", "nhan": "DE_DOA", "bang_chung": ["biết nhà mày"]},
        {"id": "d", "text": "chuyển tiền nhận quà", "nhan": "LUA_DAO", "bang_chung": ["chuyển tiền"]},
        {"id": "e", "text": "nhắn miết không tha", "nhan": "QUAY_ROI", "bang_chung": ["nhắn miết"]},
    ]
    # Mô hình classifier có probs hợp lệ
    preds_clf = [
        {"id": "a", "nhan": "AN_TOAN", "bang_chung": [], "probs": [0.9, 0.025, 0.025, 0.025, 0.025]},
        {"id": "b", "nhan": "XUC_PHAM", "bang_chung": ["ngu quá"], "probs": [0.05, 0.8, 0.05, 0.05, 0.05]},
        {"id": "c", "nhan": "DE_DOA", "bang_chung": ["biết nhà mày"], "probs": [0.05, 0.05, 0.05, 0.8, 0.05]},
        {"id": "d", "nhan": "LUA_DAO", "bang_chung": ["chuyển tiền"], "probs": [0.05, 0.05, 0.05, 0.05, 0.8]},
        {"id": "e", "nhan": "QUAY_ROI", "bang_chung": ["nhắn miết"], "probs": [0.05, 0.05, 0.8, 0.05, 0.05]},
    ]
    res_clf = evaluate_classification(records, preds_clf, n_bootstrap=50)
    assert res_clf["macro_f1"] == 1.0
    assert res_clf["binary_harmful_vs_safe"]["f1"] == 1.0
    assert res_clf["binary_harmful_vs_safe"]["fpr_an_toan"] == 0.0
    assert res_clf["calibration"] is not None and res_clf["calibration"]["ece"] < 0.25

    # Mô hình LLM không có probs + bị từ chối 1 câu DE_DOA + hỏng JSON 1 câu LUA_DAO
    preds_llm = [
        {"id": "a", "nhan": "AN_TOAN", "bang_chung": []},
        {"id": "b", "nhan": "XUC_PHAM", "bang_chung": ["ngu quá"]},
        {"id": "c", "nhan": "DE_DOA", "bang_chung": [], "refusal": True},
        {"id": "d", "nhan": "UNKNOWN", "bang_chung": [], "parse_error": True},
        {"id": "e", "nhan": "QUAY_ROI", "bang_chung": ["nhắn miết"]},
    ]
    rel = evaluate_output_reliability(preds_llm)
    assert rel["refusal_rate"] == 0.2
    assert rel["parse_error_rate"] == 0.2
    assert rel["invalid_label_rate"] == 0.2
    assert rel["valid_output_rate"] == 0.6

    res_llm = evaluate_classification(records, preds_llm, n_bootstrap=0)
    assert res_llm["calibration"] is None
    # 2 câu nguy hại bị refusal / parse_error phải bị tính thành FN (recall binary = 2/4 = 0.5)
    assert res_llm["binary_harmful_vs_safe"]["fn"] == 2
    assert res_llm["binary_harmful_vs_safe"]["recall"] == 0.5

    cmp = compare_systems(
        records,
        preds_clf,
        preds_llm,
        candidate_cost={"latency_p50_ms": 15.0, "latency_p95_ms": 25.0, "cost_per_1k_usd": 0.0},
        reference_cost={"latency_p50_ms": 450.0, "latency_p95_ms": 1200.0, "cost_per_1k_usd": 0.35},
    )
    assert cmp["delta_macro_f1"] > 0
    assert cmp["cost_latency_tradeoff"]["latency_p50_ms_speedup_x"] == 30.0
    assert cmp["cost_latency_tradeoff"]["latency_p95_ms_speedup_x"] == 48.0


def test_slices_and_cli_on_real_v2_test_suite(tmp_path):
    """Chạy CLI đánh giá trực tiếp trên tập test thật C0 + C1 + C2 (1.670 dòng) để kiểm tra cắt lát in-memory."""
    c0 = ROOT / "data" / "v2" / "test_day_du.jsonl"
    c1 = ROOT / "data" / "v2" / "test_perturb_C1.jsonl"
    c2 = ROOT / "data" / "v2" / "test_perturb_C2.jsonl"
    rows = [r for p in (c0, c1, c2) for r in cli.read_jsonl(p)]
    assert len(rows) == 334 + 668 + 668

    # Tạo dự đoán giả lập: C0 đúng 100%, C1 sai 10%, C2 sai 25%
    preds = []
    for i, r in enumerate(rows):
        bd = r.get("bien_dang") or "C0"
        wrong = (bd == "C1" and i % 10 == 0) or (bd == "C2" and i % 4 == 0)
        lbl = LABELS[(LABELS.index(r["nhan"]) + 1) % len(LABELS)] if wrong else r["nhan"]
        preds.append({"id": r["id"], "nhan": lbl, "bang_chung": list(r.get("bang_chung") or [])})

    pred_path = tmp_path / "pred.jsonl"
    out_path = tmp_path / "metrics.json"
    pred_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds), encoding="utf-8")

    rc = cli.main(
        [
            "--gold", str(c0), str(c1), str(c2),
            "--pred", str(pred_path),
            "--meta", str(c0),
            "--n-bootstrap", "50",
            "--out", str(out_path),
        ]
    )
    assert rc == 0
    rep = json.loads(out_path.read_text(encoding="utf-8"))
    assert rep["classification"]["n"] == 1670
    assert rep["slices"]["bien_dang"]["C0"]["macro_f1"] == 1.0
    assert rep["slices"]["bien_dang"]["delta_macro_f1"]["C1_minus_C0"] < 0
    assert rep["slices"]["bien_dang"]["delta_macro_f1"]["C2_minus_C0"] < rep["slices"]["bien_dang"]["delta_macro_f1"]["C1_minus_C0"]
    # Đủ cả 15 lát cắt in-memory (C0/C1/C2 x bac/trung/nam/da_thay/chua_thay) mà không cần 15 file trên đĩa
    combo = rep["slices"]["bien_dang_x_nhom"]
    assert len(combo) == 15
    assert combo["C0_bac"]["n"] == 112 and combo["C1_bac"]["n"] == 224 and combo["C2_chua_thay"]["n"] == 298
    assert "worst_group_macro_f1" in rep["slices"]["giong_vung"]
    assert "region_gap_max_min" in rep["slices"]["giong_vung"]
