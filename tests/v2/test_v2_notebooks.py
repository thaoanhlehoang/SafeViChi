"""Kiểm thử tự động cho hai notebook huấn luyện SafeViChi v2 (DualORC-CRF Schema v3 & GuardMerge Schema v6)."""

from __future__ import annotations

import ast
import builtins
import gc
import hashlib
import io
import json
import re
import symtable
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DUALORC_NB = ROOT / "notebooks" / "v2_dualorc_crf_colab.ipynb"
GUARDMERGE_NB = ROOT / "notebooks" / "v2_guardmerge_colab.ipynb"
DATA_DIR = ROOT / "data" / "v2"
EXPECTED_DATA_REV = "371a0f1caf3f7b62f62b87dbf557c66eda9c544a"
EXPECTED_TEENCODE_SHA256 = "c7f345574974668c2b319955c23df11b5d6ac3055e743cb1b4e40ec326ff1c14"
LABELS = ["AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO"]


def _load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


def _exec_defs(src: str, names: set[str], env: dict) -> dict:
    """Trích và thực thi các định nghĩa hàm/lớp/hằng cụ thể từ một ô notebook mà không chạy side-effects."""
    tree = ast.parse(src)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in names for t in targets):
                selected.append(node)
    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = dict(env)
    exec(compile(mod, "<nb_extract>", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def dualorc_nb() -> dict:
    return _load_nb(DUALORC_NB)


@pytest.fixture(scope="module")
def guardmerge_nb() -> dict:
    return _load_nb(GUARDMERGE_NB)


@pytest.fixture(scope="module")
def so_sanh_rows() -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / "so_sanh.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def test_day_du_rows() -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / "test_day_du.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def test_notebooks_valid_json_and_ast(dualorc_nb, guardmerge_nb):
    builtin_names = set(dir(builtins)) | {"__name__", "__file__"}
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        assert len(nb["cells"]) == 22
        defined = set(builtin_names)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                md_src = "".join(cell["source"])
                assert "`\n`" not in md_src, f"{label} cell {i} bị ngắt dòng trần bên trong backtick!"
                continue
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.lstrip().startswith("%pip"):
                continue
            ast.parse(src, filename=f"{label}_cell_{i}")
            st = symtable.symtable(src, f"{label}_cell_{i}", "exec")
            cell_defs = {
                sym.get_name()
                for sym in st.get_symbols()
                if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
            }
            missing = set()

            def _walk(table, is_top=True):
                for sym in table.get_symbols():
                    nm = sym.get_name()
                    if sym.is_referenced() and (
                        sym.is_global()
                        or (is_top and not sym.is_assigned() and not sym.is_imported() and not sym.is_namespace())
                    ):
                        if nm not in defined and nm not in cell_defs:
                            missing.add(nm)
                for child in table.get_children():
                    _walk(child, is_top=False)

            _walk(st, True)
            assert not missing, f"{label} cell {i} có biến toàn cục chưa định nghĩa: {sorted(missing)}"
            defined |= cell_defs


def test_config_defaults_and_locked_test_not_in_hash(dualorc_nb, guardmerge_nb):
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        c4 = _cell_source(nb, 4)
        assert f'DATA_REV = "{EXPECTED_DATA_REV}"' in c4, f"{label}: sai DATA_REV"
        assert 'RUN_MODE = "smoke"' in c4
        assert 'DATA_SCHEME = "dev_80_10_10"' in c4
        assert 'CKPT_SELECT_MODE = "real_aligned"' in c4
        assert 'CAL_WEIGHT_MODE = "real_aligned"' in c4
        assert "HUMAN_CAL_PATH = None" in c4
        assert "EVALUATE_REAL_TEST = False" in c4
        assert "EVALUATE_AUDIT = False" in c4
        assert "RUN_REAL_TEST_PERTURB = True" in c4
        assert "EVAL_WITH_NORMALIZER = True" in c4
        assert "QUANTIZE_EMBEDDINGS = True" in c4
        assert "CONSISTENCY_WEIGHT = 0.25" in c4
        assert "ACCENT_AUG = 0.30" in c4
        assert "SURFACE_AUG = 0.20" in c4
        full_nb = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "20/11" not in full_nb, f"{label}: vẫn còn ghi chú quyết định 20/11"

        # Đảm bảo cờ đánh giá test / normalizer không nằm trong config = dict(...) để giữ nguyên RUN hash khi mở khóa 1 lần cuối
        tree = ast.parse(c4)
        cfg_assign = next(
            n for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets)
        )
        cfg_kw_names = {kw.arg for kw in cfg_assign.value.keywords}
        assert "evaluate_real_test" not in cfg_kw_names
        assert "evaluate_audit" not in cfg_kw_names
        assert "eval_with_normalizer" not in cfg_kw_names
        assert "run_real_test_perturb" not in cfg_kw_names
        assert "consistency_weight" in cfg_kw_names
        assert "data_scheme" in cfg_kw_names
        assert "ckpt_select_mode" in cfg_kw_names
        assert "ckpt_select_weights" in cfg_kw_names
        assert "cal_weight_mode" in cfg_kw_names
        assert "human_cal_sha256" in cfg_kw_names

        # Đảm bảo ckpt_select_weights trong config phản ánh đúng hệ số mới và tách biệt RUN hash khỏi bản cũ (không trộn last.pt)
        w_kw = next(kw for kw in cfg_assign.value.keywords if kw.arg == "ckpt_select_weights")
        w_expr = compile(ast.Expression(w_kw.value), f"<{label}_ckpt_w>", "eval")
        w_span = eval(w_expr, {"SPAN_WEIGHT": 0.45, "CKPT_SELECT_MODE": "real_aligned"})
        w_nospan = eval(w_expr, {"SPAN_WEIGHT": 0.0, "CKPT_SELECT_MODE": "real_aligned"})
        w_macro = eval(w_expr, {"SPAN_WEIGHT": 0.45, "CKPT_SELECT_MODE": "macro_only"})
        assert w_span == [0.50, 0.15, 0.25, 0.10] and isinstance(w_span, list)
        assert w_nospan == [0.55, 0.20, 0.25, 0.0] and isinstance(w_nospan, list)
        assert w_macro is None
        old_cfg = {"ckpt_select_mode": "real_aligned", "span_weight": 0.45}
        new_cfg = {**old_cfg, "ckpt_select_weights": w_span}
        assert hashlib.sha256(json.dumps(old_cfg, sort_keys=True).encode()).hexdigest() != (
            hashlib.sha256(json.dumps(new_cfg, sort_keys=True).encode()).hexdigest()
        )


def test_data_sha256_matches_local_blobs_and_splits_identical(dualorc_nb, guardmerge_nb, so_sanh_rows, test_day_du_rows):
    base_env = {
        "Path": Path,
        "json": json,
        "np": np,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "digest": lambda b: hashlib.sha256(b).hexdigest(),
        "DATA_SCHEME": "dev_80_10_10",
    }
    ns_d = _exec_defs(_cell_source(dualorc_nb, 6), {"DATA_SHA256", "TEENCODE_DICT_SHA256", "_matches_blob_sha", "split_group", "make_splits"}, base_env)
    ns_g = _exec_defs(_cell_source(guardmerge_nb, 6), {"DATA_SHA256", "TEENCODE_DICT_SHA256", "_matches_blob_sha", "split_group", "make_splits"}, base_env)

    assert ns_d["DATA_SHA256"] == ns_g["DATA_SHA256"]
    assert set(ns_d["DATA_SHA256"]) == {
        "so_sanh.jsonl", "train.jsonl", "test.jsonl", "test_day_du.jsonl",
        "test_perturb_C1.jsonl", "test_perturb_C2.jsonl",
    }
    assert ns_d["TEENCODE_DICT_SHA256"] == ns_g["TEENCODE_DICT_SHA256"] == EXPECTED_TEENCODE_SHA256
    for fname, expected in ns_d["DATA_SHA256"].items():
        raw = (DATA_DIR / fname).read_bytes()
        assert ns_d["_matches_blob_sha"](raw, expected), f"Sai SHA-256 cho {fname}"
    tc_local = ROOT / "src" / "shared" / "normalization" / "teencode_dict.json"
    assert ns_d["_matches_blob_sha"](tc_local.read_bytes(), EXPECTED_TEENCODE_SHA256)

    # Kiểm tra cả 4 chế độ chia tập khớp 100% giữa hai notebook và không rò rỉ nhóm
    d_80 = ns_d["make_splits"](so_sanh_rows, scheme="dev_80_10_10", split_seed=42)
    g_80 = ns_g["make_splits"](so_sanh_rows, scheme="dev_80_10_10", split_seed=42)
    d_90 = ns_d["make_splits"](so_sanh_rows, scheme="final_90_10", split_seed=42)
    d_100 = ns_d["make_splits"](so_sanh_rows, scheme="final_100_human_cal", split_seed=42)
    g_100 = ns_g["make_splits"](so_sanh_rows, scheme="final_100_human_cal", split_seed=42)
    d_70 = ns_d["make_splits"](so_sanh_rows, scheme="legacy_70_10_10_10", split_seed=42)

    assert len(d_80["fit"]) == 10713 and len(d_80["dev"]) == 1340 and len(d_80["cal"]) == 1339
    assert len(d_90["fit"]) == 12053 and len(d_90["cal"]) == 1339 and "dev" not in d_90
    assert len(d_100["fit"]) == len(g_100["fit"]) == 13392 and set(d_100) == {"fit"}
    assert len(d_70["fit"]) == 9373 and len(d_70["dev"]) == 1340 and len(d_70["cal"]) == 1339 and len(d_70["audit"]) == 1340
    # Không có kịch bản giữ ngoài train trong so_sanh.jsonl
    assert not any((r.get("meta") or {}).get("kich_ban") in {"LD7", "LD12", "DD6", "QR6"} for r in so_sanh_rows)

    # dev và cal của dev_80_10_10 giữ nguyên 100% ID so với legacy_70_10_10_10 và khớp giữa 2 nhánh
    assert [r["id"] for r in d_80["dev"]] == [r["id"] for r in d_70["dev"]] == [r["id"] for r in g_80["dev"]]
    assert [r["id"] for r in d_80["cal"]] == [r["id"] for r in d_70["cal"]] == [r["id"] for r in g_80["cal"]] == [r["id"] for r in d_90["cal"]]

    # split_group trên test_day_du (nhom_trung == "" và không có meta) phải giữ đủ 334 nhóm độc lập
    test_groups = {ns_d["split_group"](r) for r in test_day_du_rows}
    assert len(test_groups) == len(test_day_du_rows) == 334
    assert not ({r["id"] for r in so_sanh_rows} & {r["id"] for r in test_day_du_rows})
    assert not ({r["text"].strip().casefold() for r in so_sanh_rows} & {r["text"].strip().casefold() for r in test_day_du_rows})


def test_depth_normalized_llrd(dualorc_nb):
    ns4 = _exec_defs(
        _cell_source(dualorc_nb, 4),
        {"BACKBONE_SPECS", "LLRD", "LLRD_BOTTOM_RATIO", "resolve_effective_llrd"},
        {},
    )
    assert "bamibert" in ns4["BACKBONE_SPECS"]
    bami = ns4["BACKBONE_SPECS"]["bamibert"]
    assert bami["repo"] == "Qualcomm-AI-Research/BamiBERT"
    assert bami["max_len"] == 1024 and bami["num_layers"] == 12
    assert bami["lr_backbone"] == 5e-5 and bami["lr_head"] == 3e-4
    assert ns4["BACKBONE_SPECS"]["mmbert_base"]["lr_backbone"] == 4.5e-5
    assert "lr_backbone=5e-5" in _cell_source(dualorc_nb, 0)
    assert '`"bamibert"` (`lr_backbone=5e-5`)' in _cell_source(dualorc_nb, 21)
    fn = ns4["resolve_effective_llrd"]
    r12 = fn(12, None, 0.92 ** 12)
    r22 = fn(22, None, 0.92 ** 12)
    assert abs(r12 - 0.92) < 1e-5
    assert abs(fn(bami["num_layers"], None, 0.92 ** 12) - 0.92) < 1e-5
    assert abs(r22 - (0.92 ** (12 / 22))) < 1e-5
    # Cả 12 lớp lẫn 22 lớp đều giữ cùng tỷ lệ LR tầng đáy / tầng đỉnh (~0.3677)
    assert abs((r12 ** 12) - (r22 ** 22)) < 1e-4
    # Cho phép ghi đè tay khi ablation
    assert fn(22, 0.92, 0.92 ** 12) == 0.92

    # Kiểm thử build_llrd_param_groups trên ViSoBERT (12 lớp, 3e-5), BamiBERT (12 lớp, 5e-5) và mmBERT-base (22 lớp, 4.5e-5)
    class FakeConfig:
        def __init__(self, n_layers):
            self.num_hidden_layers = n_layers

    class FakeBackbone(nn.Module):
        def __init__(self, n_layers, modern=False):
            super().__init__()
            self.config = FakeConfig(n_layers)
            self.embeddings = nn.Embedding(16, 8)
            if modern:
                self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(n_layers)])
            else:
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList([nn.Linear(8, 8) for _ in range(n_layers)])

    class FakeNet(nn.Module):
        def __init__(self, n_layers, modern=False):
            super().__init__()
            self.backbone = FakeBackbone(n_layers, modern=modern)
            self.head = nn.Linear(8, 5)

    for n_layers, modern, lr_bb, lr_hd in [
        (12, False, 3e-5, 3e-4),
        (bami["num_layers"], False, bami["lr_backbone"], bami["lr_head"]),
        (22, True, 4.5e-5, 3e-4),
    ]:
        env12 = {
            "re": re,
            "defaultdict": defaultdict,
            "SPEC": {"num_layers": n_layers},
            "LLRD": None,
            "LLRD_BOTTOM_RATIO": 0.92 ** 12,
            "LR_BACKBONE": lr_bb,
            "LR_HEAD": lr_hd,
            "resolve_effective_llrd": fn,
        }
        ns12 = _exec_defs(_cell_source(dualorc_nb, 12), {"build_llrd_param_groups"}, env12)
        groups = ns12["build_llrd_param_groups"](FakeNet(n_layers, modern=modern))
        lrs = sorted({g["lr"] for g in groups})
        assert len(lrs) == n_layers + 2  # embeddings + n_layers + head đều có LR tách biệt
        assert abs(lrs[-1] - lr_hd) < 1e-9  # head LR
        assert abs(lrs[-2] - lr_bb) < 1e-9  # top backbone layer
        assert abs(lrs[0] / lrs[-2] - (0.92 ** 12)) < 2e-3  # bottom/top chuẩn hóa theo chiều sâu!


def test_bio_crf_exactness_and_per_word_leg_locality(dualorc_nb):
    ns10 = _exec_defs(_cell_source(dualorc_nb, 10), {"TurnAwareBIOCRF"}, {"nn": nn, "torch": torch, "F": F})
    crf = ns10["TurnAwareBIOCRF"]()
    torch.manual_seed(42)
    W = 5
    em = torch.randn(1, W, 3)
    mk = torch.ones(1, W, dtype=torch.bool)
    ts = torch.tensor([[True, False, False, True, False]])

    # 1) NLL toàn -100 == 0.0
    nll_ign = crf.nll(em, torch.full((1, W), -100), mk, ts)
    assert abs(float(nll_ign)) < 1e-6

    # 2) Viterbi không bao giờ gán I-EV ở đầu lượt (vị trí 0 và vị trí 3)
    vit = crf.decode(em, mk, ts)[0]
    assert vit[0] != 2 and vit[3] != 2

    # 3) Per-word LEG locality: tăng polar_weights chỉ ở từ 2 phải chỉ đổi gradient ở từ 2, không rò sang từ 0!
    em1 = em.clone().requires_grad_(True)
    em2 = em.clone().requires_grad_(True)
    targets = torch.tensor([[0, 1, 2, 0, 0]])
    pw_base = torch.ones((1, W))
    pw_boost = torch.tensor([[1.0, 1.0, 1.6, 1.0, 1.0]])
    crf.nll(em1, targets, mk, ts, polar_weights=pw_base).backward()
    crf.nll(em2, targets, mk, ts, polar_weights=pw_boost).backward()
    assert float((em2.grad[0, 0] - em1.grad[0, 0]).abs().max()) < 1e-6
    assert float((em2.grad[0, 1] - em1.grad[0, 1]).abs().max()) < 1e-6
    assert float((em2.grad[0, 2] - em1.grad[0, 2]).abs().max()) > 1e-3

    # 4) Edge case: vị trí bị pad (mask=False) nhưng target >= 0 không được đóng góp loss hay NaN
    mk_pad = torch.tensor([[True, True, True, False, False]])
    loss_pad = crf.nll(em, torch.tensor([[0, 1, 2, 1, 2]]), mk_pad, ts, polar_weights=torch.full((1, W), 1.5))
    assert torch.isfinite(loss_pad) and float(loss_pad.detach()) >= 0.0


def test_f1_theo_y_matches_spec_and_evidence_fallback_and_bloat_guard(dualorc_nb, guardmerge_nb):
    env_d = {"re": re, "np": np, "defaultdict": defaultdict, "USE_BIO_CRF": True}
    ns_d = _exec_defs(
        _cell_source(dualorc_nb, 12),
        {
            "_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item",
            "_viterbi_single_np", "_decode_item_crf_tags", "_apply_marginal_postproc", "_apply_empty_fallback",
            "select_word_marks", "extract_evidence_runs", "evidence_stats",
        },
        env_d,
    )
    ns_g = _exec_defs(
        _cell_source(guardmerge_nb, 12),
        {"_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item", "_apply_empty_fallback", "select_word_marks", "evidence_stats"},
        {"re": re, "np": np},
    )

    # Kiểm thử khớp quy tắc không ngoặc (Bước 3 của docs/v2/DO_DO_BANG_CHUNG.md) trên cả hai notebook
    text = "sao m ngu như bò thế. t đã nói nhiều lần rồi mà"
    words = [(m.start(), m.end(), 0.1, 0) for m in re.finditer(r"\S+", text)]
    for fn in (ns_d["f1_theo_y_item"], ns_g["f1_theo_y_item"]):
        for idxs, expected in [
            ({2, 3, 4}, 1.0),
            ({0, 1, 2, 3, 4, 5}, 1.0),
            ({2, 3, 4, 5, 6, 7}, 0.8),
            (set(range(len(words))), 12 / 19),
            ({7, 8, 9}, 0.0),
            (set(), 0.0),
        ]:
            marks = [i in idxs for i in range(len(words))]
            assert abs(fn(text, words, marks, ["ngu như bò"]) - expected) < 1e-4

        # Dấu câu đứng trước ngoặc kép/ngoặc đơn đóng vẫn phải cắt đúng ranh giới mệnh đề!
        text_q = 'nó nói "m ngu như bò thế." t đã nói nhiều lần rồi'
        words_q = [(m.start(), m.end(), 0.1, 0) for m in re.finditer(r"\S+", text_q)]
        marks_q = [i in {3, 4, 5, 6, 7, 8} for i in range(len(words_q))]
        assert abs(fn(text_q, words_q, marks_q, ["ngu như bò"]) - 0.8) < 1e-4
        # Không tách nhầm khi ký tự sau dấu chấm là chữ thường (vd. "thế.u")
        assert not ns_d["_CLAUSE_END_RE"].search("thế.u")
        assert ns_d["_CLAUSE_END_RE"].search("câu:”") and ns_d["_CLAUSE_END_RE"].search("làm.:))")

    # Kiểm thử extract_evidence_runs thống nhất gate=True trên AN_TOAN ở cả mode="crf" lẫn mode="marginal"
    fake_crf_np = dict(
        start=np.array([0.0, 2.0, -5.0]),
        end=np.zeros(3),
        trans=np.array([[0.0, 1.0, -5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    ml_text = "dòng một bình thường\ndòng hai đe dọa nặng"
    ml_words = [(m.start(), m.end(), p, 0, 0, False) for m, p in zip(re.finditer(r"\S+", ml_text), [0.10, 0.28, 0.30, 0.32, 0.35, 0.34, 0.33, 0.31, 0.10])]
    item_harm = dict(
        id="t1", text=ml_text, gold=3, gold_phrases=["đe dọa nặng"],
        logits=[0.0, 0.1, 0.1, 3.0, 0.1], words=ml_words,
        emissions=np.tile(np.array([[-1.0, 2.0, 1.0]]), (len(ml_words), 1)),
    )
    # Ngưỡng chính 0.5 -> rỗng; fallback 0.25 -> chọn đỉnh dòng 2 (index 4, p=0.35) và mở rộng chỉ trong dòng 2!
    marks_fb = ns_d["select_word_marks"](item_harm, threshold=0.5, gate=True, mode="marginal", fallback_min_prob=0.25)
    assert any(marks_fb)
    # Không được tô bất kỳ từ nào ở dòng 1 (indices 0..3)!
    assert not any(marks_fb[:4])
    assert all(marks_fb[4:8])  # "dòng hai đe dọa" trong cùng dòng 2, dừng trước từ p=0.10
    # Khi peak (0.35) < fallback_min_prob (0.40) -> vẫn trả rỗng hợp lệ
    marks_strict = ns_d["select_word_marks"](item_harm, threshold=0.5, gate=True, mode="marginal", fallback_min_prob=0.40)
    assert not any(marks_strict)
    # Khi dự đoán AN_TOAN -> tuyệt đối không tô dù bật fallback hay giải mã CRF
    item_safe = dict(item_harm, logits=[4.0, 0.1, 0.1, 0.1, 0.1])
    assert not any(ns_d["select_word_marks"](item_safe, threshold=0.5, gate=True, mode="marginal", fallback_min_prob=0.10))
    assert ns_d["extract_evidence_runs"](item_safe, ml_text, crf_np=fake_crf_np, gate=True, mode="crf") == []
    assert ns_d["extract_evidence_runs"](item_safe, ml_text, threshold=0.1, gate=True, mode="marginal") == []
    assert ns_d["extract_evidence_runs"](item_harm, ml_text, crf_np=fake_crf_np, gate=True, mode="crf")

    # Kiểm thử _ev_rank_key phạt phình cụm và _fb_rank_key cân bằng giảm trắng bằng chứng vs không tăng tô oan AN_TOAN
    ns14_d = _exec_defs(_cell_source(dualorc_nb, 14), {"_ev_rank_key", "_fb_rank_key"}, {})
    rk = ns14_d["_ev_rank_key"]
    fb_rk = ns14_d["_fb_rank_key"]
    bloated_viterbi = dict(f1=0.6954, precision=0.6060, safe_mark_rate=0.052, word_length_ratio=1.33)
    clean_marginal = dict(f1=0.6949, precision=0.6563, safe_mark_rate=0.052, word_length_ratio=1.12)
    assert rk(clean_marginal) > rk(bloated_viterbi)

    no_fb_st = dict(f1=0.6950, f1_theo_y=0.7200, precision=0.656, safe_mark_rate=0.052, word_length_ratio=1.12, harm_empty_evidence_rate=0.077)
    good_fb_st = dict(f1=0.6945, f1_theo_y=0.7420, precision=0.651, safe_mark_rate=0.052, word_length_ratio=1.15, harm_empty_evidence_rate=0.015)
    noisy_fb_st = dict(f1=0.6952, f1_theo_y=0.7350, precision=0.630, safe_mark_rate=0.090, word_length_ratio=1.20, harm_empty_evidence_rate=0.005)
    assert fb_rk(good_fb_st, no_fb_st["safe_mark_rate"]) > fb_rk(no_fb_st, no_fb_st["safe_mark_rate"])
    assert fb_rk(good_fb_st, no_fb_st["safe_mark_rate"]) > fb_rk(noisy_fb_st, no_fb_st["safe_mark_rate"])


def test_real_aligned_calibration_weights_and_checkpoint_selection(dualorc_nb, guardmerge_nb, so_sanh_rows):
    # 1) Trọng số hiệu chuẩn 2 trục trên tập cal thật (fold 1, 1.339 mẫu)
    ns6 = _exec_defs(
        _cell_source(dualorc_nb, 6),
        {"split_group", "make_splits"},
        {"np": np, "StratifiedGroupKFold": StratifiedGroupKFold, "DATA_SCHEME": "dev_80_10_10"},
    )
    cal_rows = ns6["make_splits"](so_sanh_rows, scheme="dev_80_10_10", split_seed=42)["cal"]
    for nb in (dualorc_nb, guardmerge_nb):
        ns14 = _exec_defs(
            _cell_source(nb, 14),
            {"compute_cal_weights", "compute_temp_cal_weights"},
            {"np": np, "CAL_WEIGHT_MODE": "real_aligned"},
        )
        w = ns14["compute_cal_weights"](cal_rows, mode="real_aligned")
        assert len(w) == len(cal_rows)
        assert abs(float(w.mean()) - 1.0) < 1e-2
        assert float(w.max()) <= 4.0001
        # Cân bằng 5 lớp (~18–22%/lớp) và ưu tiên câu thật > ẩn ý > lộ liễu
        by_lbl = Counter()
        for r, wi in zip(cal_rows, w):
            by_lbl[r["nhan"]] += wi
        for lbl in LABELS:
            share = by_lbl[lbl] / w.sum()
            assert 0.17 <= share <= 0.23, f"{lbl}: share={share:.4f}"
        w_that_at = np.mean([wi for r, wi in zip(cal_rows, w) if r["nhan"] == "AN_TOAN" and r["meta"]["loai"] == "that"])
        w_sinh_at = np.mean([wi for r, wi in zip(cal_rows, w) if r["nhan"] == "AN_TOAN" and r["meta"]["loai"] == "sinh"])
        assert w_that_at > 2.0 * w_sinh_at

        # Trọng số hiệu chuẩn nhiệt độ T phải giữ ~70% tổng khối lượng cho câu thật (không bị 57% sinh kéo T về 1)
        wt = ns14["compute_temp_cal_weights"](cal_rows, mode="real_aligned", that_share=0.70)
        that_share = sum(wi for r, wi in zip(cal_rows, wt) if r["meta"]["loai"] == "that") / wt.sum()
        assert 0.65 <= that_share <= 0.75, f"that_share={that_share:.4f}"
        assert float(wt.max()) <= 3.0001
        # Vẫn đại diện đủ 5 lớp gồm LUA_DAO (sinh)
        lua_dao_mass = sum(wi for r, wi in zip(cal_rows, wt) if r["nhan"] == "LUA_DAO") / wt.sum()
        assert lua_dao_mass > 0.05

    # 2) Tiêu chí chọn checkpoint real_aligned giảm phương sai trên cả 6 quỹ đạo lịch sử thực tế
    #    (visobert v1, visobert v2, mmbert_base v1, mmbert_base v2, bamibert v2, guardmerge v5) + nhánh span_weight == 0 + biên
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        all_md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        assert "0.40 * macro_f1 + 0.30 * that_macro_f1" not in all_md, f"{label}: Markdown còn công thức checkpoint cũ"
        assert "0.45 * macro_f1 + 0.35 * that_macro_f1" not in all_md, f"{label}: Markdown còn công thức span=0 cũ"
        assert "0.50 * macro_f1 + 0.15 * that_macro_f1 + 0.25 * that_acc + 0.10 * ev_f1" in _cell_source(nb, 0)
        assert "0.50 * macro_f1 + 0.15 * that_macro_f1 + 0.25 * that_acc + 0.10 * ev_f1" in _cell_source(nb, 11)
        assert "0.55 * macro_f1 + 0.20 * that_macro_f1 + 0.25 * that_acc" in _cell_source(nb, 11)
        assert "0.50/0.15/0.25/0.10" in _cell_source(nb, 21) or "0.50 * macro_f1 + 0.15 * that_macro_f1 + 0.25 * that_acc + 0.10 * ev_f1" in _cell_source(nb, 21)

        ns12 = _exec_defs(
            _cell_source(nb, 12),
            {"checkpoint_select_key"},
            {"SPAN_WEIGHT": 0.45, "CKPT_SELECT_MODE": "real_aligned"},
        )
        sel = ns12["checkpoint_select_key"]

        # Kiểm định chính xác hệ số công thức cho span_weight > 0, span_weight == 0 (bất biến toàn bộ tuple), fallback rỗng/None, mode không hợp lệ & phá hòa
        k_span = sel(0.80, {"that_macro_f1": 0.60, "that_acc": 0.90}, 0.70, span_weight=0.45)
        assert abs(k_span[0] - round(0.50 * 0.80 + 0.15 * 0.60 + 0.25 * 0.90 + 0.10 * 0.70, 6)) < 1e-9
        k_nospan = sel(0.80, {"that_macro_f1": 0.60, "that_acc": 0.90}, 0.70, span_weight=0.0)
        assert abs(k_nospan[0] - round(0.55 * 0.80 + 0.20 * 0.60 + 0.25 * 0.90, 6)) < 1e-9
        k_nospan_ev_inv = sel(0.80, {"that_macro_f1": 0.60, "that_acc": 0.90}, 0.10, span_weight=0.0)
        assert k_nospan == k_nospan_ev_inv  # span_weight == 0 bất biến toàn bộ tuple trước ev_f1
        assert sel(0.80, {}, 0.70, span_weight=0.0, mode="macro_only") == (0.80, 0.0)
        assert sel(0.80, {}, 0.10, span_weight=0.0, mode="macro_only") == (0.80, 0.0)
        k_missing = sel(0.80, {}, 0.70, span_weight=0.45)
        assert abs(k_missing[0] - round(0.90 * 0.80 + 0.10 * 0.70, 6)) < 1e-9
        assert sel(0.80, None, 0.70, span_weight=0.45) == k_missing
        assert sel(0.80, {"that_macro_f1": None, "that_acc": None}, 0.70, span_weight=0.45) == k_missing
        assert sel(0.80, {"that_macro_f1": 0.10, "that_acc": 0.10}, 0.70, mode="macro_only") == (0.80, 0.70)
        with pytest.raises(ValueError, match="CKPT_SELECT_MODE"):
            sel(0.80, {}, 0.70, mode="invalid_mode")
        # Khi điểm số làm tròn bằng nhau tuyệt đối (0.800000), ưu tiên phá hòa theo macro_f1 > that_acc > that_f1 (giảm phương sai)
        tie_hi_macro = sel(0.90, {"that_macro_f1": 0.50, "that_acc": 0.82}, 0.70, span_weight=0.45)
        tie_hi_that = sel(0.87, {"that_macro_f1": 0.60, "that_acc": 0.82}, 0.70, span_weight=0.45)
        assert tie_hi_macro[0] == tie_hi_that[0] == 0.800000 and tie_hi_macro > tie_hi_that

        # (a) visobert v1: ep5 thắng ep4 (cả macro lẫn that_macro đều cao nhất)
        v1_ep4 = sel(0.9304, {"that_macro_f1": 0.5872, "that_acc": 0.8655}, 0.6818)
        v1_ep5 = sel(0.9348, {"that_macro_f1": 0.5883, "that_acc": 0.8670}, 0.6727)
        assert v1_ep5 > v1_ep4

        # (b) visobert v2 (dualorc_crf/21aabb95f3e6e0): ep4 thắng ep5 và đứng đầu cả 5 epoch
        v2_eps = [
            sel(0.886455, {"that_macro_f1": 0.557297, "that_acc": 0.843049}, 0.553555),
            sel(0.901352, {"that_macro_f1": 0.628488, "that_acc": 0.853513}, 0.627377),
            sel(0.915329, {"that_macro_f1": 0.618453, "that_acc": 0.860987}, 0.650293),
            sel(0.939783, {"that_macro_f1": 0.652384, "that_acc": 0.869955}, 0.691873),
            sel(0.938948, {"that_macro_f1": 0.641933, "that_acc": 0.862481}, 0.690606),
        ]
        assert v2_eps[3] > v2_eps[4] and max(v2_eps) == v2_eps[3]

        # (c) mmbert_base v1: ep3 thắng ep5 (that_macro +6.78 pp, ev_f1 tốt hơn)
        m1_ep3 = sel(0.9063, {"that_macro_f1": 0.6009, "that_acc": 0.8221}, 0.6543)
        m1_ep5 = sel(0.9111, {"that_macro_f1": 0.5331, "that_acc": 0.8341}, 0.6524)
        assert m1_ep3 > m1_ep5

        # (d) mmbert_base v2 (dualorc_crf/712ea025fd511e): ep5 thắng ep4 và đứng đầu cả 5 epoch
        m2_eps = [
            sel(0.812335, {"that_macro_f1": 0.440724, "that_acc": 0.751868}, 0.521620),
            sel(0.888509, {"that_macro_f1": 0.494851, "that_acc": 0.792227}, 0.613539),
            sel(0.917873, {"that_macro_f1": 0.549995, "that_acc": 0.846039}, 0.673401),
            sel(0.923837, {"that_macro_f1": 0.592015, "that_acc": 0.850523}, 0.643242),
            sel(0.924461, {"that_macro_f1": 0.610132, "that_acc": 0.853513}, 0.695562),
        ]
        assert m2_eps[4] > m2_eps[3] and max(m2_eps) == m2_eps[4]

        # (e) bamibert v2 (dualorc_crf/f61218ae88820b): ep5 thắng ep4 (ev_f1 +1.67 pp, safe_mark_rate giảm)
        b2_eps = [
            sel(0.779179, {"that_macro_f1": 0.478059, "that_acc": 0.690583}, 0.522603),
            sel(0.876182, {"that_macro_f1": 0.547778, "that_acc": 0.786248}, 0.635114),
            sel(0.906969, {"that_macro_f1": 0.567419, "that_acc": 0.847534}, 0.656296),
            sel(0.920188, {"that_macro_f1": 0.549696, "that_acc": 0.843049}, 0.679103),
            sel(0.919250, {"that_macro_f1": 0.545338, "that_acc": 0.843049}, 0.695765),
        ]
        assert b2_eps[4] > b2_eps[3] and max(b2_eps) == b2_eps[4]

        # (f) GuardMerge v5 (guardmerge/aa96881af096bd): ep3 > ep2 > ep4 (tránh chọn nhầm ep2 chưa chín)
        g5_eps = [
            sel(0.854548, {"that_macro_f1": 0.527470, "that_acc": 0.811659}, 0.610262),
            sel(0.886526, {"that_macro_f1": 0.660901, "that_acc": 0.840060}, 0.663085),
            sel(0.894777, {"that_macro_f1": 0.636987, "that_acc": 0.835575}, 0.673000),
            sel(0.898600, {"that_macro_f1": 0.600374, "that_acc": 0.835575}, 0.677815),
        ]
        assert g5_eps[2] > g5_eps[1] > g5_eps[3] and max(g5_eps) == g5_eps[2]


def test_per_row_int8_embedding_and_free_training_memory(dualorc_nb, guardmerge_nb):
    for nb in (dualorc_nb, guardmerge_nb):
        ns18 = _exec_defs(_cell_source(nb, 18), {"PerRowInt8Embedding", "quantize_embeddings_int8"}, {"nn": nn, "torch": torch, "F": F})
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_emb = nn.Embedding(20481, 64, padding_idx=1)  # đúng cỡ bảng từ vựng BamiBERT
                self.pos_emb = nn.Embedding(2050, 64, padding_idx=1)  # mô phỏng position_embeddings của BamiBERT (2050 hàng)
                self.turn_emb = nn.Embedding(8, 64)
                self.fc = nn.Linear(64, 16)
            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x) + 2
                return self.fc(self.vocab_emb(x) + self.pos_emb(pos) + self.turn_emb(torch.zeros_like(x)))

        torch.manual_seed(7)
        m = ToyModel().eval()
        x = torch.tensor([[1, 5, 100, 2048], [10, 20, 1, 1]])
        with torch.no_grad():
            y_fp = m(x)
        b_fp = io.BytesIO()
        torch.save(m.state_dict(), b_fp)

        replaced = ns18["quantize_embeddings_int8"](m)
        assert replaced == 1
        assert isinstance(m.vocab_emb, ns18["PerRowInt8Embedding"])
        assert m.vocab_emb.scale.dtype == torch.float32  # không underflow FP16 trên hàng có chuẩn nhỏ
        assert isinstance(m.pos_emb, nn.Embedding)  # position_embeddings (<=2050) giữ nguyên FP32
        assert isinstance(m.turn_emb, nn.Embedding)  # embedding nhỏ giữ nguyên FP32
        with torch.no_grad():
            y_q = m(x)
            pad_vec = m.vocab_emb(torch.tensor([[1]]))
        assert float(pad_vec.abs().max()) == 0.0
        assert float((y_fp - y_q).abs().max()) < 0.05
        # Kiểm thử hàng embedding cực nhỏ (1e-6) không bị underflow về 0
        tiny_emb = nn.Embedding(16, 8)
        with torch.no_grad():
            tiny_emb.weight.fill_(1e-6)
        q_tiny = ns18["PerRowInt8Embedding"](tiny_emb)
        assert float(q_tiny(torch.tensor([[3]])).abs().mean()) > 5e-7
        b_q = io.BytesIO()
        torch.save(m.state_dict(), b_q)
        assert len(b_q.getvalue()) < len(b_fp.getvalue()) * 0.35

        # Kiểm thử free_training_memory dọn sạch biến huấn luyện và grad
        ns12 = _exec_defs(_cell_source(nb, 12), {"free_training_memory"}, {"gc": gc, "torch": torch, "DEVICE": torch.device("cpu")})
        lin = nn.Linear(4, 2)
        lin(torch.randn(2, 4)).sum().backward()
        assert lin.weight.grad is not None
        extra_key = "g_grad" if nb is dualorc_nb else "named"
        fake_ns = {"opt": torch.optim.AdamW(lin.parameters()), extra_key: torch.randn(10, 10), "b": {"x": 1}, "keep_me": 123}
        ns12["free_training_memory"](lin, fake_ns)
        assert lin.weight.grad is None
        assert "opt" not in fake_ns and extra_key not in fake_ns and "b" not in fake_ns and fake_ns["keep_me"] == 123


def test_real_test_evaluate_all_required_slices_and_bootstrap(dualorc_nb, guardmerge_nb, test_day_du_rows):
    # Chạy thật hàm evaluate() của CẢ HAI notebook trên toàn bộ 334 bản ghi test_day_du.jsonl với dự đoán mô phỏng
    rng = np.random.default_rng(42)
    records = []
    logits_list = []
    for idx, r in enumerate(test_day_du_rows):
        rec = dict(r)
        if not rec.get("nhom_trung"):
            rec["nhom_trung"] = rec["id"]
        rec["meta"] = {"loai": "that", "phan_test": rec.get("phan_test", ""), "giong_vung": rec.get("giong_vung", "")}
        records.append(rec)
        g = LABELS.index(rec["nhan"])
        logits = rng.normal(0.0, 0.4, size=5)
        # 80% dự đoán đúng, 20% lệch để kiểm tra bootstrap CI không suy biến
        logits[g if idx % 5 != 0 else (g + 1) % 5] += 2.5
        logits_list.append(logits.tolist())

    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        env = {
            "re": re,
            "np": np,
            "defaultdict": defaultdict,
            "LABELS": LABELS,
            "SEED": 42,
            "USE_BIO_CRF": False,
            "SPAN_WEIGHT": 0.45,
            "f1_score": f1_score,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix,
            "log_loss": log_loss,
            "softmax": softmax,
            "crf_np": None,
            "DATA_SCHEME": "dev_80_10_10",
        }
        ns6 = _exec_defs(_cell_source(nb, 6), {"split_group"}, env)
        env.update(ns6)
        want12 = {
            "_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item",
            "_apply_empty_fallback", "select_word_marks", "evidence_stats", "macro",
        }
        if label == "dualorc":
            want12.add("_apply_marginal_postproc")
        ns12 = _exec_defs(_cell_source(nb, 12), want12, env)
        env.update(ns12)
        ns14 = _exec_defs(_cell_source(nb, 14), {"arrays", "calibration_stats", "evaluate"}, env)

        preds = []
        for idx, (rec, logits) in enumerate(zip(records, logits_list)):
            g = LABELS.index(rec["nhan"])
            # 1/4 mẫu nguy hại cho xác suất từ bằng chứng thấp hơn ngưỡng chính (0.35 < 0.50) nhưng >= fallback (0.25)
            ev_p = 0.35 if (g != 0 and idx % 4 == 0) else 0.80
            words = []
            for m in re.finditer(r"\S+", rec["text"]):
                a, b = m.span()
                in_ev = any(ph in rec["text"] and any(a < (s + len(ph)) and s < b for s in [rec["text"].find(ph)]) for ph in rec["bang_chung"])
                score = ev_p if in_ev else 0.10
                if label == "dualorc":
                    words.append((a, b, score, int(in_ev), 0, False))
                else:
                    words.append((a, b, score, int(in_ev)))
            item = dict(id=rec["id"], text=rec["text"], gold=g, gold_phrases=list(rec["bang_chung"]), logits=logits, words=words)
            if label == "dualorc":
                item.update(severity=0.5, emissions=None)
            preds.append(item)

        if label == "dualorc":
            cal_cfg = dict(
                temperature=1.15, o_bias=0.0, chosen_decoder="marginal",
                base_marginal_threshold=0.5, marginal_threshold=0.5,
                single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=0.25,
            )
            cal_no_fb = dict(cal_cfg, fallback_min_prob=1.01)
        else:
            cal_cfg = dict(
                temperature=1.15, base_span_threshold=0.5, span_threshold=0.5,
                single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=0.25,
            )
            cal_no_fb = dict(cal_cfg, fallback_min_prob=1.01)

        res = ns14["evaluate"](preds, records, cal_cfg=cal_cfg)
        res_no_fb = ns14["evaluate"](preds, records, cal_cfg=cal_no_fb)
        assert res["slices"]["phan:trong_phan_phoi"]["n"] == 185
        assert res["slices"]["phan:chua_thay"]["n"] == 149
        assert res["slices"]["vung:bac"]["n"] == 112
        assert res["slices"]["vung:trung"]["n"] == 111
        assert res["slices"]["vung:nam"]["n"] == 111
        assert res["slices"]["1 lượt"]["n"] == 108
        assert res["slices"]["nhiều lượt"]["n"] == 226
        assert res["slices"]["do_lo:lo_lieu"]["n"] == 134
        assert res["slices"]["do_lo:an_y"]["n"] == 104
        assert res["slices"]["cach_gan:nguoi"]["n"] == 325
        assert res["slices"]["cach_gan:llm_nguoi_duyet"]["n"] == 9
        assert 0.0 <= res["fpr_an_toan"] <= 1.0
        assert 0.0 < res["worst_group_region_macro_f1"] <= 1.0
        assert 0.0 <= res["region_gap_max_min"] <= 1.0
        assert 0.0 < res["macro_f1_4main"] <= 1.0
        assert res["macro_f1_bootstrap_95"][0] < res["macro_f1_bootstrap_95"][1]
        assert res["macro_f1_4main_bootstrap_95"][0] < res["macro_f1_4main_bootstrap_95"][1]
        assert res["binary_f1_bootstrap_95"][0] < res["binary_f1_bootstrap_95"][1]
        assert res["evidence"]["f1_theo_y"] > 0.7
        # Fallback làm giảm rõ rệt tỷ lệ trắng bằng chứng trên câu nguy hại nhưng vẫn giữ 0% tô oan câu AN_TOAN!
        assert res["evidence"]["harm_empty_evidence_rate"] < res_no_fb["evidence"]["harm_empty_evidence_rate"]
        assert res["evidence"]["safe_mark_rate"] == 0.0


def test_dev_meta_slices_and_normalized_dedup_and_ucci_router_non_collapse(dualorc_nb, guardmerge_nb, so_sanh_rows, test_day_du_rows):
    # 1) Kiểm thử evaluate() đọc đúng lát cắt vung/do_lo/kenh từ meta trên so_sanh.jsonl (dev)
    ns6 = _exec_defs(
        _cell_source(dualorc_nb, 6),
        {"split_group", "make_splits"},
        {"np": np, "StratifiedGroupKFold": StratifiedGroupKFold, "DATA_SCHEME": "dev_80_10_10"},
    )
    dev_rows = ns6["make_splits"](so_sanh_rows, scheme="dev_80_10_10", split_seed=42)["dev"][:300]
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        env = {
            "re": re, "np": np, "defaultdict": defaultdict, "LABELS": LABELS, "SEED": 42,
            "USE_BIO_CRF": False, "SPAN_WEIGHT": 0.45, "f1_score": f1_score,
            "classification_report": classification_report, "confusion_matrix": confusion_matrix,
            "log_loss": log_loss, "softmax": softmax, "crf_np": None, "DATA_SCHEME": "dev_80_10_10",
            "unicodedata": unicodedata,
        }
        env.update(_exec_defs(_cell_source(nb, 6), {"split_group"}, env))
        env.update(_exec_defs(_cell_source(nb, 8), {"strip_accents"}, env))
        want12 = {"_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item", "_apply_empty_fallback", "select_word_marks", "evidence_stats", "macro"}
        if label == "dualorc":
            want12.add("_apply_marginal_postproc")
        env.update(_exec_defs(_cell_source(nb, 12), want12, env))
        ns14 = _exec_defs(_cell_source(nb, 14), {"arrays", "calibration_stats", "evaluate", "_norm_dedup_text"}, env)

        # Không có trùng văn bản sau chuẩn hóa bỏ dấu/dấu câu giữa so_sanh và test_day_du
        norm_fn = ns14["_norm_dedup_text"]
        assert not ({norm_fn(r["text"]) for r in so_sanh_rows} & {norm_fn(r["text"]) for r in test_day_du_rows})

        preds = []
        for r in dev_rows:
            g = LABELS.index(r["nhan"])
            logits = [0.0] * 5
            logits[g] = 2.5
            words = [(m.start(), m.end(), 0.8, 0, 0, False) if label == "dualorc" else (m.start(), m.end(), 0.8, 0) for m in re.finditer(r"\S+", r["text"])]
            it = dict(id=r["id"], text=r["text"], gold=g, gold_phrases=list(r["bang_chung"]), logits=logits, words=words)
            if label == "dualorc":
                it.update(severity=0.5, emissions=None)
            preds.append(it)
        cal_cfg = (
            dict(temperature=1.1, o_bias=0.0, chosen_decoder="marginal", base_marginal_threshold=0.5,
                 marginal_threshold=0.5, single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=1.01)
            if label == "dualorc"
            else dict(temperature=1.1, base_span_threshold=0.5, span_threshold=0.5,
                      single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=1.01)
        )
        res = ns14["evaluate"](preds, dev_rows, cal_cfg=cal_cfg)
        for req_slice in ("do_lo:an_y", "do_lo:lo_lieu", "vung:bac", "vung:trung", "vung:nam", "kenh:messenger_zalo"):
            assert req_slice in res["slices"] and res["slices"][req_slice]["n"] > 0, f"{label}: thiếu lát cắt {req_slice}"

    # 2) Kiểm thử bộ định tuyến UCCI Cell 20 không suy biến thành chuyển tầng 100% khi mô hình chính xác cao (~92.5%)
    ns20 = _exec_defs(_cell_source(dualorc_nb, 20), {"_weighted_quantile", "_ucci_route_score"}, {"np": np})
    rng = np.random.default_rng(42)
    n_cal = 1339
    z_cal = rng.normal(0.0, 0.3, (n_cal, 5))
    y_cal = rng.integers(0, 5, n_cal)
    for i in range(n_cal):
        if rng.random() < 0.925:
            z_cal[i, y_cal[i]] += rng.uniform(2.5, 5.0)
        else:
            z_cal[i, (y_cal[i] + 1) % 5] += rng.uniform(1.0, 2.5)
    p_cal = softmax(z_cal, axis=1)
    top2 = np.sort(p_cal, axis=1)[:, -2:]
    margin_cal = top2[:, 1] - top2[:, 0]
    err_cal = (p_cal.argmax(1) != y_cal).astype(np.float64)
    w_cal = rng.uniform(0.5, 2.5, n_cal)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False, out_of_bounds="clip")
    iso.fit(margin_cal, err_cal, sample_weight=w_cal)
    scores = ns20["_ucci_route_score"](iso, margin_cal)
    for target_rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        thr = ns20["_weighted_quantile"](scores, w_cal, 1.0 - target_rate)
        weighted_esc = w_cal[scores >= thr].sum() / w_cal.sum()
        assert abs(weighted_esc - target_rate) < 0.03, f"target={target_rate}, got weighted_esc={weighted_esc:.4f}"


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_frozen_calibration_and_one_shot_test_lock_protection(dualorc_nb, guardmerge_nb, so_sanh_rows, test_day_du_rows, tmp_path):
    """Thực thi trực tiếp toàn bộ ô 14 của CẢ HAI notebook trên dữ liệu thật trong tmp_path để kiểm định:
    (1) Chạy lần 1 (EVALUATE_REAL_TEST=False): khớp và đóng băng calibration.json, không chạm test.
    (2) Mở khóa 1 lần cuối (EVALUATE_REAL_TEST=True): nạp nguyên vẹn calibration.json (không ghi đè SHA),
        chấm 334 câu test_day_du.jsonl và tạo REAL_TEST_LOCK.json + real_test_*.
    (3) Chạy lại lần 2 (EVALUATE_REAL_TEST=True): bị khóa từ chối chấm lại, không ghi đè calibration.json/real_test_metrics.json.
    (4) Sửa lén calibration.json sau khi đã khóa -> AssertionError.
    (5) Chạy EVALUATE_REAL_TEST=True ở RUN_MODE='smoke' -> RuntimeError.
    """
    import os
    import time
    import pandas as pd

    ns6 = _exec_defs(
        _cell_source(dualorc_nb, 6),
        {"split_group", "make_splits"},
        {"np": np, "StratifiedGroupKFold": StratifiedGroupKFold, "DATA_SCHEME": "dev_80_10_10"},
    )
    splits = ns6["make_splits"](so_sanh_rows, scheme="dev_80_10_10", split_seed=42)
    # Lấy tập nhỏ đủ 5 lớp và cả that/sinh cho tốc độ test nhanh (<1s) nhưng vẫn chạy đúng 334 câu test thật
    def _mini(rows_in, per_lbl=4):
        out = []
        for lbl in LABELS:
            sub = [r for r in rows_in if r["nhan"] == lbl]
            that_s = [r for r in sub if (r.get("meta") or {}).get("loai") == "that"][:per_lbl]
            sinh_s = [r for r in sub if (r.get("meta") or {}).get("loai") == "sinh"][:per_lbl]
            out.extend(that_s + sinh_s)
        return out

    mini_dev = _mini(splits["dev"], 3)
    mini_cal = _mini(splits["cal"], 3)
    test_slim_rows = [json.loads(l) for l in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_c1_rows = [json.loads(l) for l in (DATA_DIR / "test_perturb_C1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_c2_rows = [json.loads(l) for l in (DATA_DIR / "test_perturb_C2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    teencode_dict = json.loads((ROOT / "src" / "shared" / "normalization" / "teencode_dict.json").read_text(encoding="utf-8"))
    test_stats = json.loads((DATA_DIR / "test_stats.json").read_text(encoding="utf-8"))

    def _run_cell14(label, nb, run_dir, run_mode, eval_real_test):
        c14 = _cell_source(nb, 14)
        def _fake_prepare(r):
            return dict(row=r)

        def _fake_infer(_net, data):
            rng_i = np.random.default_rng(123)
            out = []
            for idx, e in enumerate(data):
                r = e["row"]
                g = LABELS.index(r["nhan"])
                logits = rng_i.normal(0.0, 0.3, 5)
                logits[g if idx % 4 != 0 else (g + 1) % 5] += 2.2
                words = []
                for m in re.finditer(r"\S+", r["text"]):
                    a, b = m.span()
                    in_ev = any(ph in r["text"] and a < (r["text"].find(ph) + len(ph)) and r["text"].find(ph) < b for ph in r["bang_chung"])
                    sc = 0.75 if (in_ev or (g != 0 and not r["bang_chung"] and m.start() < 15)) else 0.12
                    words.append((a, b, sc, int(in_ev), 0, False) if label == "dualorc" else (a, b, sc, int(in_ev)))
                it = dict(id=r["id"], text=r["text"], gold=g, gold_phrases=list(r["bang_chung"]), logits=logits.tolist(), words=words)
                if label == "dualorc":
                    it.update(severity=0.6, emissions=None)
                out.append(it)
            return out

        def _write_json(path, obj):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
            os.replace(tmp, path)

        def _fetch_data(name, expected=None):
            if name == "test_day_du.jsonl":
                return [dict(r) for r in test_day_du_rows]
            if name == "test.jsonl":
                return [dict(r) for r in test_slim_rows]
            if name == "test_perturb_C1.jsonl":
                return [dict(r) for r in test_c1_rows]
            if name == "test_perturb_C2.jsonl":
                return [dict(r) for r in test_c2_rows]
            raise ValueError(name)

        env = {
            "os": os, "re": re, "json": json, "time": time, "Path": Path, "np": np, "pd": pd,
            "defaultdict": defaultdict, "unicodedata": unicodedata,
            "LABELS": LABELS, "SEED": 42, "RUN": run_dir, "RUN_MODE": run_mode,
            "DATA_REV": EXPECTED_DATA_REV, "DATA_SCHEME": "dev_80_10_10",
            "CAL_WEIGHT_MODE": "real_aligned", "HUMAN_CAL_PATH": None,
            "EVALUATE_AUDIT": False, "EVALUATE_REAL_TEST": eval_real_test,
            "RUN_REAL_TEST_PERTURB": True, "EVAL_WITH_NORMALIZER": True,
            "ACCENT_AUG": 0.30, "SURFACE_AUG": 0.20,
            "SPAN_WEIGHT": 0.45, "SPAN_POSTPROC": True, "WORD_SCORE": "max",
            "USE_BIO_CRF": False, "EVIDENCE_DECODE_MODE": "marginal",
            "model": None, "rows": so_sanh_rows,
            "encoded": {"dev": [_fake_prepare(r) for r in mini_dev], "cal": [_fake_prepare(r) for r in mini_cal]},
            "prepare": _fake_prepare, "infer": _fake_infer,
            "get_crf_params_np": lambda _m: None,
            "digest": lambda b: hashlib.sha256(b).hexdigest(),
            "write_json": _write_json, "fetch_data": _fetch_data,
            "fetch_teencode_dict": lambda expected=None: dict(teencode_dict),
            "f1_score": f1_score, "classification_report": classification_report,
            "confusion_matrix": confusion_matrix, "log_loss": log_loss,
            "softmax": softmax, "minimize_scalar": minimize_scalar,
        }
        env.update(_exec_defs(_cell_source(nb, 6), {"DATA_SHA256", "_matches_blob_sha", "split_group"}, env))
        env.update(_exec_defs(_cell_source(nb, 8), {"strip_accents", "annotations", "overlaps"}, env))
        want12 = {
            "_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item", "_apply_empty_fallback",
            "select_word_marks", "evidence_stats", "macro",
            "_NORM_LOOKALIKE", "_NORM_SEP_RE", "_NORM_ELONG_RE", "_NORM_LEAD_RE", "_NORM_TRAIL_RE",
            "_norm_strip_base", "_normalize_v1_token", "normalize_with_alignment", "infer_normalized",
        }
        if label == "dualorc":
            want12.update({"_apply_marginal_postproc", "extract_evidence_runs", "_viterbi_single_np", "_decode_item_crf_tags"})
        env.update(_exec_defs(_cell_source(nb, 12), want12, env))

        import matplotlib
        matplotlib.use("Agg", force=True)
        exec(compile(c14, f"<{label}_cell14>", "exec"), env)
        return env

    from src.v2.eval import evaluate as official_eval

    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        rdir = tmp_path / label
        rdir.mkdir()
        # (1) Lần 1: đóng khóa test -> tạo calibration.json và biểu đồ dev_diagnostics.png, KHÔNG tạo real_test_*
        env1 = _run_cell14(label, nb, rdir, run_mode="full", eval_real_test=False)
        cal_bytes_1 = (rdir / "calibration.json").read_bytes()
        cal_sha_1 = hashlib.sha256(cal_bytes_1).hexdigest()
        assert (rdir / "dev_diagnostics.png").read_bytes()[:4] == b"\x89PNG"
        assert env1["real_test_pred"] is None
        assert not (rdir / "REAL_TEST_LOCK.json").exists()
        assert not (rdir / "real_test_metrics.json").exists()

        # (5) Chặn mở khóa ở RUN_MODE="smoke"
        with pytest.raises(RuntimeError, match="KHÓA AN TOÀN"):
            _run_cell14(label, nb, rdir, run_mode="smoke", eval_real_test=True)
        assert (rdir / "calibration.json").read_bytes() == cal_bytes_1

        # (2) Mở khóa đúng 1 lần cuối ở RUN_MODE="full" -> giữ nguyên calibration.json, xuất đủ C0+C1+C2 (#5 và #6) & khóa
        env2 = _run_cell14(label, nb, rdir, run_mode="full", eval_real_test=True)
        assert (rdir / "calibration.json").read_bytes() == cal_bytes_1, f"{label}: calibration.json bị ghi đè khi mở khóa test!"
        assert (rdir / "real_test_diagnostics.png").read_bytes()[:4] == b"\x89PNG"
        assert (rdir / "REAL_TEST_LOCK.json").exists()
        lock = json.loads((rdir / "REAL_TEST_LOCK.json").read_text(encoding="utf-8"))
        assert lock["frozen_calibration_sha256"] == cal_sha_1
        assert lock["n_test"] == 334 and lock["n_c1"] == 668 and lock["n_c2"] == 668
        assert len(env2["real_test_pred"]) == 334
        assert len(env2["real_test_c1_pred"]) == 668 and len(env2["real_test_c2_pred"]) == 668
        rt_metrics_bytes = (rdir / "real_test_metrics.json").read_bytes()
        assert len((rdir / "real_test_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 334
        assert len((rdir / "real_test_all_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1670
        assert len((rdir / "real_test_norm_all_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1670

        # Kiểm định ma trận 15 lát cắt và nhãn strictly_unseen trung thực (cả đối chiếu C0 train lẫn aug trực tuyến)
        raw_mat = json.loads((rdir / "real_test_official_matrix.json").read_text(encoding="utf-8"))
        norm_mat = json.loads((rdir / "real_test_norm_official_matrix.json").read_text(encoding="utf-8"))
        assert (rdir / "real_test_norm_metrics.json").exists()
        assert (rdir / "real_test_norm_c1_metrics.json").exists()
        assert (rdir / "real_test_norm_c2_metrics.json").exists()
        for sl_name, sl_meta in test_stats["lat_cat_C0_C1_C2"].items():
            assert raw_mat["slices_15"][sl_name]["n"] == sl_meta["so_dong"], f"{label}: lệch số lượng lát cắt {sl_name}"
        assert raw_mat["strictly_unseen_perturb"]["n"] == 764  # 668 C2 + 96 dinh_tu (go_sai là da_thay theo perturb.py!)
        assert raw_mat["by_kieu_bien_dang"]["go_sai"]["seen_in_train_aug"] is False
        assert raw_mat["by_kieu_bien_dang"]["go_sai"]["strictly_unseen"] is False
        for u_kieu in ("cyrillic_homoglyph", "zero_width", "exotic_separator", "space_injection", "double_elongation", "fullwidth", "dinh_tu"):
            assert raw_mat["by_kieu_bien_dang"][u_kieu]["seen_in_train_aug"] is False
            assert raw_mat["by_kieu_bien_dang"][u_kieu]["strictly_unseen"] is True
        assert "gain_vs_raw" in norm_mat and set(norm_mat["gain_vs_raw"]) >= {"C0", "C1", "C2", "strictly_unseen"}

        # Kiểm định trường hợp ablation tắt tăng cường (ACCENT_AUG=0, SURFACE_AUG=0): strictly_unseen phải đúng 938 (668 C2 + 270 C1 chua_thay),
        # KHÔNG gom nhầm teencode/bo_dau/keo_dai/go_sai (vốn da_thay trong train C0)!
        env_noaug = dict(env2, ACCENT_AUG=0.0, SURFACE_AUG=0.0)
        ns_noaug = _exec_defs(_cell_source(nb, 14), {"_kieu_seen_in_train_aug", "build_official_test_matrix"}, env_noaug)
        noaug_mat, _ = ns_noaug["build_official_test_matrix"](
            env2["real_test_records"], env2["real_test_pred"],
            env2["real_test_c1_records"], env2["real_test_c1_pred"],
            env2["real_test_c2_records"], env2["real_test_c2_pred"],
            env2["cal_payload"],
        )
        assert noaug_mat["strictly_unseen_perturb"]["n"] == 938  # 668 C2 + 81 chen_ky_tu + 93 so_thay_chu + 96 dinh_tu
        for seen_c0 in ("teencode", "bo_dau", "keo_dai", "go_sai"):
            assert noaug_mat["by_kieu_bien_dang"][seen_c0]["strictly_unseen"] is False
        for unseen_noaug in ("chen_ky_tu", "so_thay_chu", "dinh_tu", "fullwidth"):
            assert noaug_mat["by_kieu_bien_dang"][unseen_noaug]["strictly_unseen"] is True

        # Kiểm định tích hợp đầu-cuối trực tiếp với CLI chuẩn src.v2.eval.evaluate cho cả #5 thô và #6 chuẩn hóa!
        for pred_name in ("real_test_all_predictions.jsonl", "real_test_norm_all_predictions.jsonl"):
            eval_out = rdir / f"official_eval_{pred_name}.json"
            rc = official_eval.main([
                "--gold", str(DATA_DIR / "test_day_du.jsonl"), str(DATA_DIR / "test_perturb_C1.jsonl"), str(DATA_DIR / "test_perturb_C2.jsonl"),
                "--pred", str(rdir / pred_name),
                "--meta", str(DATA_DIR / "test_day_du.jsonl"),
                "--n-bootstrap", "0",
                "--out", str(eval_out),
            ])
            assert rc == 0
            rep = json.loads(eval_out.read_text(encoding="utf-8"))
            assert rep["classification"]["n"] == 1670
            assert rep["classification"]["calibration"] is not None, f"{label}/{pred_name}: thiếu probs nên calibration=None!"
            assert rep["evidence"]["non_verbatim_phrase_rate"] == 0.0, f"{label}/{pred_name}: sinh cụm không verbatim!"
            assert rep["slices"]["bien_dang"]["C0"]["n"] == 334
            assert rep["slices"]["bien_dang"]["C1"]["n"] == 668
            assert rep["slices"]["bien_dang"]["C2"]["n"] == 668
            for sl_name, sl_meta in test_stats["lat_cat_C0_C1_C2"].items():
                assert rep["slices"]["bien_dang_x_nhom"][sl_name]["n"] == sl_meta["so_dong"]

        # (3) Thử chấm lại lần 2 -> bị chặn bởi REAL_TEST_LOCK.json, không ghi đè bất kỳ tệp nào
        env3 = _run_cell14(label, nb, rdir, run_mode="full", eval_real_test=True)
        assert env3["real_test_pred"] is None
        assert (rdir / "calibration.json").read_bytes() == cal_bytes_1
        assert (rdir / "real_test_metrics.json").read_bytes() == rt_metrics_bytes

        # (4) Sửa lén calibration.json sau khi đã khóa -> phát hiện sai SHA-256!
        tampered = json.loads(cal_bytes_1.decode("utf-8"))
        tampered["temperature"] = 9.99
        (rdir / "calibration.json").write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(AssertionError, match="SHA-256"):
            _run_cell14(label, nb, rdir, run_mode="full", eval_real_test=True)

    # Đảm bảo GuardMerge Cell 19 không còn hướng dẫn lỗi thời về EVALUATE_AUDIT hay "chưa có test người gán"
    c19_g = _cell_source(guardmerge_nb, 19)
    assert "EVALUATE_AUDIT = True" not in c19_g
    assert "EVALUATE_REAL_TEST = True" in c19_g
    assert "final_100_human_cal" in c19_g


def test_read_human_anti_leakage_and_round2_fixes(dualorc_nb, guardmerge_nb, so_sanh_rows, tmp_path):
    """Kiểm định 4 điểm vá Vòng 2 trên cả hai notebook:
    (1) read_human() chặn test.jsonl / test_day_du.jsonl (kể cả bản đổi tên / CRLF) qua SHA-256 ghim,
        chặn trùng sau bỏ dấu (_norm_dedup_text) với so_sanh.jsonl, và chấp nhận tệp validation người gán sạch.
    (2) Cell 4 chuẩn hóa CRLF -> LF khi tính human_cal_sha256 để RUN hash đồng nhất Windows/Linux.
    (3) Không còn chuỗi ghi cứng 'fixed_epoch_final_90_10' trong cả hai notebook; dùng mode=f'fixed_epoch_{DATA_SCHEME}'.
    (4) DualORC cell 17 ghi đúng 'scale FP32' (không còn 'scale FP16' lệch với PerRowInt8Embedding).
    """
    # Tạo câu trùng sau bỏ dấu + đổi hoa/thường + thêm dấu câu từ một câu thật có dấu trong so_sanh.jsonl
    accented_src = next(
        r for r in so_sanh_rows
        if r["nhan"] != "AN_TOAN"
        and any(ord(ch) > 127 for ch in r["text"])
        and "\n" not in r["text"]
        and len(r["text"]) >= 25
    )

    clean_records = [
        {
            "id": "hval_clean_001",
            "text": "Chiều nay trời mát ghê, nhóm mình đi ăn bún bò Huế không?\nĐi luôn bạn ơi, hẹn 6 giờ trước cổng thư viện nhé.",
            "nhan": "AN_TOAN",
            "bang_chung": [],
            "nguon": "nguoi_viet",
            "cach_gan": "nguoi",
            "meta": {"giong_vung": "trung", "do_lo": "lo_lieu", "kenh": "messenger_zalo"},
        },
        {
            "id": "hval_clean_002",
            "text": "🎉🚀✨",
            "nhan": "AN_TOAN",
            "bang_chung": [],
            "nguon": "nguoi_viet",
            "cach_gan": "nguoi",
        },
        {
            "id": "hval_clean_003",
            "text": "Gửi bài tập muộn chút thôi mà làm quá lên vậy?\nCái đồ vô dụng óc bã đậu, cút khỏi nhóm cho sạch chỗ.",
            "nhan": "XUC_PHAM",
            "bang_chung": ["đồ vô dụng óc bã đậu"],
            "nguon": "nguoi_viet",
            "cach_gan": "nguoi",
        },
        {
            "id": "hval_clean_004",
            "text": "Mình đã nói rõ không muốn gặp nữa rồi.\nKhông trả lời tin nhắn thì tối nay tao đăng hết ảnh riêng tư lên nhóm trường cho biết mặt.",
            "nhan": "DE_DOA",
            "bang_chung": ["đăng hết ảnh riêng tư lên nhóm trường"],
            "nguon": "nguoi_viet",
            "cach_gan": "nguoi",
        },
        {
            "id": "hval_clean_005",
            "text": "Hệ thống ngân hàng thông báo tài khoản quý khách sắp bị tạm khóa, vui lòng nhập mã OTP tại trang-xac-minh-nhanh.example.",
            "nhan": "LUA_DAO",
            "bang_chung": ["nhập mã OTP tại trang-xac-minh-nhanh.example"],
            "nguon": "nguoi_viet",
            "cach_gan": "nguoi",
        },
    ]
    clean_lf = tmp_path / "clean_human_val_lf.jsonl"
    clean_crlf = tmp_path / "clean_human_val_crlf.jsonl"
    clean_bom_crlf = tmp_path / "clean_human_val_bom_crlf.jsonl"
    clean_payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in clean_records) + "\n"
    clean_lf.write_bytes(clean_payload.encode("utf-8"))
    clean_crlf.write_bytes(clean_payload.replace("\n", "\r\n").encode("utf-8"))
    clean_bom_crlf.write_bytes(b"\xef\xbb\xbf" + clean_payload.replace("\n", "\r\n").encode("utf-8"))

    # Bản sao đổi tên + chuyển CRLF + gắn UTF-8 BOM của các tệp test và train (chống rò rỉ trên cả Windows/Linux)
    renamed_test_crlf = tmp_path / "renamed_val_copy.jsonl"
    renamed_test_crlf.write_bytes((DATA_DIR / "test.jsonl").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    renamed_full_crlf = tmp_path / "renamed_full_copy.jsonl"
    renamed_full_crlf.write_bytes((DATA_DIR / "test_day_du.jsonl").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    renamed_test_bom = tmp_path / "renamed_test_bom.jsonl"
    renamed_test_bom.write_bytes(b"\xef\xbb\xbf" + (DATA_DIR / "test.jsonl").read_bytes())
    renamed_full_bom = tmp_path / "renamed_full_bom.jsonl"
    renamed_full_bom.write_bytes(b"\xef\xbb\xbf" + (DATA_DIR / "test_day_du.jsonl").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    renamed_c1_bom = tmp_path / "renamed_c1_bom.jsonl"
    renamed_c1_bom.write_bytes(b"\xef\xbb\xbf" + (DATA_DIR / "test_perturb_C1.jsonl").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    renamed_c2_crlf = tmp_path / "renamed_c2_crlf.jsonl"
    renamed_c2_crlf.write_bytes((DATA_DIR / "test_perturb_C2.jsonl").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        all_src = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "fixed_epoch_final_90_10" not in all_src, f"{label}: vẫn còn chuỗi ghi cứng fixed_epoch_final_90_10"
        assert 'mode=f"fixed_epoch_{DATA_SCHEME}"' in _cell_source(nb, 12), f"{label}: thiếu mode=f'fixed_epoch_{{DATA_SCHEME}}'"
        assert "scale FP16" not in all_src, f"{label}: vẫn còn ghi sai scale FP16"

        # Kiểm tra biểu thức human_cal_sha256 trong cell 4 chuẩn hóa BOM + CRLF -> LF cho ra hash giống hệt nhau
        c4 = _cell_source(nb, 4)
        tree4 = ast.parse(c4)
        cfg_assign = next(
            n for n in tree4.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets)
        )
        hsha_kw = next(kw for kw in cfg_assign.value.keywords if kw.arg == "human_cal_sha256")
        hsha_code = compile(ast.Expression(hsha_kw.value), f"<{label}_hsha>", "eval")
        sha_lf = eval(hsha_code, {"hashlib": hashlib, "Path": Path, "HUMAN_CAL_PATH": str(clean_lf)})
        sha_crlf = eval(hsha_code, {"hashlib": hashlib, "Path": Path, "HUMAN_CAL_PATH": str(clean_crlf)})
        sha_bom = eval(hsha_code, {"hashlib": hashlib, "Path": Path, "HUMAN_CAL_PATH": str(clean_bom_crlf)})
        sha_none = eval(hsha_code, {"hashlib": hashlib, "Path": Path, "HUMAN_CAL_PATH": None})
        assert sha_lf == sha_crlf == sha_bom and len(sha_lf) == 64 and sha_none is None

        # Kiểm tra _norm_dedup_text được định nghĩa trước read_human trong cell 14
        c14 = _cell_source(nb, 14)
        assert c14.index("def _norm_dedup_text(") < c14.index("def read_human(")
        assert c14.count("def _norm_dedup_text(") == 1

        env = {
            "re": re,
            "json": json,
            "Path": Path,
            "unicodedata": unicodedata,
            "LABELS": LABELS,
            "rows": so_sanh_rows,
            "digest": lambda b: hashlib.sha256(b).hexdigest(),
        }
        env.update(_exec_defs(_cell_source(nb, 6), {"DATA_SHA256", "LOCKED_DATA_FILES", "_matches_blob_sha"}, env))
        env.update(_exec_defs(_cell_source(nb, 8), {"strip_accents"}, env))
        ns14 = _exec_defs(c14, {"_norm_dedup_text", "read_human"}, env)
        read_human = ns14["read_human"]
        strip_accents = env["strip_accents"]

        # 1) Từ chối ngay test.jsonl, test_day_du.jsonl, test_perturb_C1/C2.jsonl, so_sanh.jsonl, train.jsonl và các bản sao đổi tên/CRLF/BOM
        c6_tree = ast.parse(_cell_source(nb, 6))
        c6_hcal_if = next(
            n for n in c6_tree.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "HUMAN_CAL_PATH"
        )
        c6_guard_mod = compile(ast.Module(body=[c6_hcal_if], type_ignores=[]), f"<{label}_c6_hcal_guard>", "exec")
        for bad_blob_path in (
            DATA_DIR / "test.jsonl",
            DATA_DIR / "test_day_du.jsonl",
            DATA_DIR / "test_perturb_C1.jsonl",
            DATA_DIR / "test_perturb_C2.jsonl",
            DATA_DIR / "so_sanh.jsonl",
            DATA_DIR / "train.jsonl",
            renamed_test_crlf,
            renamed_full_crlf,
            renamed_test_bom,
            renamed_full_bom,
            renamed_c1_bom,
            renamed_c2_crlf,
        ):
            with pytest.raises(ValueError, match="KHÓA AN TOÀN"):
                exec(c6_guard_mod, {**env, "HUMAN_CAL_PATH": str(bad_blob_path)})
            with pytest.raises(ValueError, match="KHÓA AN TOÀN"):
                read_human(bad_blob_path)
        exec(c6_guard_mod, {**env, "HUMAN_CAL_PATH": str(clean_lf)})
        exec(c6_guard_mod, {**env, "HUMAN_CAL_PATH": str(clean_bom_crlf)})
        exec(c6_guard_mod, {**env, "HUMAN_CAL_PATH": None})

        # 2) Từ chối câu trùng sau bỏ dấu + đổi hoa/thường + chèn dấu câu hoặc dấu gạch dưới `_` với so_sanh.jsonl
        near_dup_text = strip_accents(accented_src["text"]).upper() + "!!!"
        assert near_dup_text.strip().casefold() != accented_src["text"].strip().casefold()
        underscore_dup_text = re.sub(r"\s+", "_", strip_accents(accented_src["text"]).strip())
        for idx_nd, bad_text in enumerate((near_dup_text, underscore_dup_text)):
            near_dup_file = tmp_path / f"{label}_near_dup_{idx_nd}.jsonl"
            near_dup_file.write_text(
                json.dumps({
                    "id": f"hval_near_dup_{idx_nd}",
                    "text": bad_text,
                    "nhan": accented_src["nhan"],
                    "bang_chung": [],
                    "nguon": "nguoi_viet",
                    "cach_gan": "nguoi",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(AssertionError, match="Rò rỉ dữ liệu train"):
                read_human(near_dup_file)

        # 3) Từ chối nguồn LLM sinh, AN_TOAN có bằng chứng, bang_chung kiểu str/thiếu, hoặc trùng nội dung nội bộ
        llm_file = tmp_path / f"{label}_llm_bad.jsonl"
        llm_file.write_text(
            json.dumps({
                "id": "hval_llm_001",
                "text": "Câu hoàn toàn mới nhưng do LLM sinh ra.",
                "nhan": "AN_TOAN",
                "bang_chung": [],
                "nguon": "sinh_llm",
                "cach_gan": "llm",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError):
            read_human(llm_file)

        bad_safe_ev = tmp_path / f"{label}_safe_ev_bad.jsonl"
        bad_safe_ev.write_text(
            json.dumps({
                "id": "hval_safe_ev_001",
                "text": "Hôm nay trời rất đẹp và trong xanh.",
                "nhan": "AN_TOAN",
                "bang_chung": ["trời rất đẹp"],
                "nguon": "nguoi_viet",
                "cach_gan": "nguoi",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError):
            read_human(bad_safe_ev)

        bad_str_ev = tmp_path / f"{label}_str_ev_bad.jsonl"
        bad_str_ev.write_text(
            json.dumps({
                "id": "hval_str_ev_001",
                "text": "Cái đồ ngu ngốc đáng ghét lắm luôn.",
                "nhan": "XUC_PHAM",
                "bang_chung": "đồ ngu",
                "nguon": "nguoi_viet",
                "cach_gan": "nguoi",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="bang_chung"):
            read_human(bad_str_ev)

        bad_missing_ev = tmp_path / f"{label}_missing_ev_bad.jsonl"
        bad_missing_ev.write_text(
            json.dumps({
                "id": "hval_missing_ev_001",
                "text": "Hôm nay nắng ấm dễ chịu ghê.",
                "nhan": "AN_TOAN",
                "nguon": "nguoi_viet",
                "cach_gan": "nguoi",
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="bang_chung"):
            read_human(bad_missing_ev)

        dup_internal = tmp_path / f"{label}_dup_internal.jsonl"
        dup_internal.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in [
                {"id": "hval_di_1", "text": "Chiều nay đi học thêm lúc mấy giờ vậy bạn?", "nhan": "AN_TOAN", "bang_chung": [], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
                {"id": "hval_di_2", "text": "  CHIỀU NAY ĐI HỌC THÊM LÚC MẤY GIỜ VẬY BẠN? ", "nhan": "AN_TOAN", "bang_chung": [], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
            ]) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(AssertionError, match="Trùng lặp nội dung"):
            read_human(dup_internal)

        # 4) Chấp nhận tệp validation người gán sạch giả lập (LF, CRLF, UTF-8 BOM+CRLF; bảo toàn meta và câu emoji riêng biệt)
        loaded_lf = read_human(clean_lf)
        loaded_crlf = read_human(clean_crlf)
        loaded_bom = read_human(clean_bom_crlf)
        assert len(loaded_lf) == len(loaded_crlf) == len(loaded_bom) == len(clean_records)
        assert loaded_lf[0]["so_dong"] == 2
        assert loaded_lf[0]["meta"]["giong_vung"] == "trung"
        assert loaded_lf[0]["meta"]["kenh"] == "messenger_zalo"
        assert loaded_lf[1]["text"] == "🎉🚀✨"
        assert all(r["nhom_trung"] == r["id"] and r["meta"]["loai"] == "that" for r in loaded_bom)

    # Xác nhận Markdown DualORC cell 17 khớp chính xác scale FP32
    assert "scale FP32" in _cell_source(dualorc_nb, 17)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_cell14_blocks_test_as_human_cal_and_supports_final_100_human_cal(
    dualorc_nb, guardmerge_nb, so_sanh_rows, test_day_du_rows, tmp_path
):
    """Kiểm định tích hợp ô 14 cho kịch bản HUMAN_CAL_PATH & mở khóa test trên final_100_human_cal:
    (1) Đặt nhầm HUMAN_CAL_PATH = data/v2/test.jsonl khi EVALUATE_REAL_TEST=False -> lập tức ValueError, không tạo calibration.json.
    (2) Chạy DATA_SCHEME='final_100_human_cal' với HUMAN_CAL_PATH sạch (encoded chỉ có 'fit') -> đóng băng calibration.json,
        mở khóa EVALUATE_REAL_TEST=True chấm 334 câu thật mà không ghi đè calibration.json, và từ chối nếu calibration.json bị xóa sau khi khóa.
    (3) Nếu HUMAN_CAL_PATH bị trộn 1 câu trích từ test_day_du.jsonl sau bỏ dấu -> load_locked_real_test() chặn ngay khi mở khóa.
    """
    import os
    import time
    import pandas as pd

    test_slim_rows = [json.loads(l) for l in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    clean_rows = [
        {"id": "hc_01", "text": "Sáng nay cả nhà đi chợ mua trái cây tươi ngon quá.", "nhan": "AN_TOAN", "bang_chung": [], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_02", "text": "Chiều nay cậu nhớ mang theo sổ tay học nhóm nhé.", "nhan": "AN_TOAN", "bang_chung": [], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_03", "text": "Thứ đồ ngu ngốc kém cỏi làm hỏng hết kế hoạch của nhóm.", "nhan": "XUC_PHAM", "bang_chung": ["đồ ngu ngốc kém cỏi"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_04", "text": "Đồ hạ đẳng bẩn thỉu đừng có mở miệng ở đây nữa.", "nhan": "XUC_PHAM", "bang_chung": ["Đồ hạ đẳng bẩn thỉu"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_05", "text": "Em gửi ảnh riêng tư cho anh đi, đừng giả bộ ngoan hiền nữa.", "nhan": "QUAY_ROI", "bang_chung": ["gửi ảnh riêng tư cho anh đi"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_06", "text": "Tôi đã chặn rồi mà sao cứ bám riết nhắn gạ gẫm mãi vậy.", "nhan": "QUAY_ROI", "bang_chung": ["bám riết nhắn gạ gẫm"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_07", "text": "Không trả đủ năm triệu chiều nay thì tao tới tận nhà đập phá.", "nhan": "DE_DOA", "bang_chung": ["tới tận nhà đập phá"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_08", "text": "Mày mà báo công an tao sẽ cho người xử lý cả nhà mày.", "nhan": "DE_DOA", "bang_chung": ["cho người xử lý cả nhà mày"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_09", "text": "Chúc mừng trúng xe máy, chuyển phí hồ sơ và gửi mã OTP ngay.", "nhan": "LUA_DAO", "bang_chung": ["chuyển phí hồ sơ và gửi mã OTP"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
        {"id": "hc_10", "text": "Tài khoản ví điện tử bị khóa khẩn cấp, đăng nhập link-gia-mao.example để mở lại.", "nhan": "LUA_DAO", "bang_chung": ["đăng nhập link-gia-mao.example"], "nguon": "nguoi_viet", "cach_gan": "nguoi"},
    ]
    clean_path = tmp_path / "clean_hcal.jsonl"
    clean_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean_rows) + "\n", encoding="utf-8")

    # Tệp HUMAN_CAL_PATH bị trộn 1 câu từ test_day_du.jsonl sau bỏ dấu + đổi dấu gạch dưới (khác SHA-256 nguyên tệp)
    strip_fn = _exec_defs(_cell_source(dualorc_nb, 8), {"strip_accents"}, {"unicodedata": unicodedata})["strip_accents"]
    leaked_t0 = test_day_du_rows[0]
    poison_rows = clean_rows + [{
        "id": "hc_poison_test_01",
        "text": re.sub(r"\s+", "_", strip_fn(leaked_t0["text"]).strip()).upper(),
        "nhan": leaked_t0["nhan"],
        "bang_chung": [],
        "nguon": "nguoi_viet",
        "cach_gan": "nguoi",
    }]
    poison_path = tmp_path / "poison_hcal.jsonl"
    poison_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in poison_rows) + "\n", encoding="utf-8")

    def _run_c14_hcal(label, nb, run_dir, hcal_path, eval_real_test=False):
        def _fake_prepare(r):
            return dict(row=r)

        def _fake_infer(_net, data):
            rng_i = np.random.default_rng(7)
            out = []
            for idx, e in enumerate(data):
                r = e["row"]
                g = LABELS.index(r["nhan"])
                logits = rng_i.normal(0.0, 0.25, 5)
                logits[g if idx % 5 != 0 else (g + 1) % 5] += 2.4
                words = []
                for m in re.finditer(r"\S+", r["text"]):
                    a, b = m.span()
                    in_ev = any(ph in r["text"] and a < (r["text"].find(ph) + len(ph)) and r["text"].find(ph) < b for ph in r["bang_chung"])
                    sc = 0.78 if in_ev else 0.11
                    words.append((a, b, sc, int(in_ev), 0, False) if label == "dualorc" else (a, b, sc, int(in_ev)))
                it = dict(id=r["id"], text=r["text"], gold=g, gold_phrases=list(r["bang_chung"]), logits=logits.tolist(), words=words)
                if label == "dualorc":
                    it.update(severity=0.5, emissions=None)
                out.append(it)
            return out

        def _write_json(path, obj):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
            os.replace(tmp, path)

        def _fetch_data(name, expected=None):
            if not eval_real_test:
                raise AssertionError("Không được gọi fetch_data khi EVALUATE_REAL_TEST=False")
            if name == "test_day_du.jsonl":
                return [dict(r) for r in test_day_du_rows]
            if name == "test.jsonl":
                return [dict(r) for r in test_slim_rows]
            raise ValueError(name)

        env = {
            "os": os, "re": re, "json": json, "time": time, "Path": Path, "np": np, "pd": pd,
            "defaultdict": defaultdict, "unicodedata": unicodedata,
            "LABELS": LABELS, "SEED": 42, "RUN": run_dir, "RUN_MODE": "full",
            "DATA_REV": EXPECTED_DATA_REV, "DATA_SCHEME": "final_100_human_cal",
            "CAL_WEIGHT_MODE": "real_aligned", "HUMAN_CAL_PATH": str(hcal_path),
            "EVALUATE_AUDIT": False, "EVALUATE_REAL_TEST": eval_real_test,
            "RUN_REAL_TEST_PERTURB": False, "EVAL_WITH_NORMALIZER": False,
            "ACCENT_AUG": 0.30, "SURFACE_AUG": 0.20,
            "SPAN_WEIGHT": 0.45, "SPAN_POSTPROC": True, "WORD_SCORE": "max",
            "USE_BIO_CRF": False, "EVIDENCE_DECODE_MODE": "marginal",
            "model": None, "rows": so_sanh_rows,
            "encoded": {"fit": []},
            "prepare": _fake_prepare, "infer": _fake_infer,
            "get_crf_params_np": lambda _m: None,
            "digest": lambda b: hashlib.sha256(b).hexdigest(),
            "write_json": _write_json,
            "fetch_data": _fetch_data,
            "fetch_teencode_dict": lambda expected=None: {},
            "infer_normalized": lambda *_a, **_kw: [],
            "f1_score": f1_score, "classification_report": classification_report,
            "confusion_matrix": confusion_matrix, "log_loss": log_loss,
            "softmax": softmax, "minimize_scalar": minimize_scalar,
        }
        env.update(_exec_defs(_cell_source(nb, 6), {"DATA_SHA256", "_matches_blob_sha", "split_group"}, env))
        env.update(_exec_defs(_cell_source(nb, 8), {"strip_accents"}, env))
        want12 = {"_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item", "_apply_empty_fallback", "select_word_marks", "evidence_stats", "macro"}
        if label == "dualorc":
            want12.update({"_apply_marginal_postproc", "extract_evidence_runs", "_viterbi_single_np", "_decode_item_crf_tags"})
        env.update(_exec_defs(_cell_source(nb, 12), want12, env))

        import matplotlib
        matplotlib.use("Agg", force=True)
        exec(compile(_cell_source(nb, 14), f"<{label}_cell14_hcal>", "exec"), env)
        return env

    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        # (1) Từ chối đặt nhầm test.jsonl / test_perturb_C1.jsonl / test_perturb_C2.jsonl làm HUMAN_CAL_PATH
        for bad_name in ("test.jsonl", "test_perturb_C1.jsonl", "test_perturb_C2.jsonl"):
            bad_dir = tmp_path / f"{label}_bad_hcal_{bad_name}"
            bad_dir.mkdir()
            with pytest.raises(ValueError, match="KHÓA AN TOÀN"):
                _run_c14_hcal(label, nb, bad_dir, DATA_DIR / bad_name, eval_real_test=False)
            assert not (bad_dir / "calibration.json").exists()

        # (2) Chạy clean final_100_human_cal -> đóng băng -> mở khóa real_test 1 lần -> xóa calibration.json phải bị chặn
        ok_dir = tmp_path / f"{label}_ok_hcal"
        ok_dir.mkdir()
        env_ok = _run_c14_hcal(label, nb, ok_dir, clean_path, eval_real_test=False)
        assert env_ok["cal_source"] == "human"
        assert env_ok["report_key"] == "human_cal"
        cal_bytes = (ok_dir / "calibration.json").read_bytes()
        assert (ok_dir / "human_cal_metrics.json").exists()
        assert (ok_dir / "human_cal_diagnostics.png").read_bytes()[:4] == b"\x89PNG"
        assert env_ok["real_test_pred"] is None

        env_rt = _run_c14_hcal(label, nb, ok_dir, clean_path, eval_real_test=True)
        assert (ok_dir / "calibration.json").read_bytes() == cal_bytes
        assert (ok_dir / "REAL_TEST_LOCK.json").exists()
        assert len(env_rt["real_test_pred"]) == 334

        (ok_dir / "calibration.json").unlink()
        with pytest.raises(AssertionError, match="thiếu calibration.json"):
            _run_c14_hcal(label, nb, ok_dir, clean_path, eval_real_test=True)
        assert not (ok_dir / "calibration.json").exists()

        # (3) HUMAN_CAL_PATH chứa 1 câu lấy từ test_day_du.jsonl sau bỏ dấu + gạch dưới -> chặn ngay khi mở khóa
        poison_dir = tmp_path / f"{label}_poison_hcal"
        poison_dir.mkdir()
        _run_c14_hcal(label, nb, poison_dir, poison_path, eval_real_test=False)
        with pytest.raises(AssertionError):
            _run_c14_hcal(label, nb, poison_dir, poison_path, eval_real_test=True)
        assert not (poison_dir / "REAL_TEST_LOCK.json").exists()
        assert not (poison_dir / "real_test_metrics.json").exists()


def test_cell12_fixed_epoch_history_mode_for_final_schemes(dualorc_nb, guardmerge_nb, tmp_path):
    """Thực thi thật vòng lặp huấn luyện nhánh `else:` (not has_dev) và nhánh resume `last.pt` của ô 12
    trên mô hình đồ chơi CPU cho CẢ HAI notebook với `final_100_human_cal` và `final_90_10`."""
    import math
    import os
    import random
    import time
    from contextlib import nullcontext
    from itertools import islice
    import pandas as pd

    class _TinyDualORC(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.config = type("Cfg", (), {"num_hidden_layers": 2})()
            self.backbone.embeddings = nn.Embedding(32, 8)
            self.backbone.get_input_embeddings = lambda: self.backbone.embeddings
            self.backbone.encoder = nn.Module()
            self.backbone.encoder.layer = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
            self.cls_head = nn.Linear(8, 5)
            self.sev_head = nn.Linear(8, 1)
            self.harm_head = nn.Linear(8, 1)
            self.orc_blend_logit = nn.Parameter(torch.zeros(()))

        def forward(self, b, with_aux=False):
            h = self.backbone.embeddings(b["ids"]).mean(dim=1)
            for lyr in self.backbone.encoder.layer:
                h = F.relu(lyr(h))
            return dict(
                logits=self.cls_head(h),
                harm_logit=self.harm_head(h).squeeze(-1),
                severity=self.sev_head(h).squeeze(-1).sigmoid(),
                proto=None,
                emissions=None,
                ev_prob=None,
            )

    class _TinyGuardMerge(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.head = nn.Linear(8, 5)

        def forward(self, b, with_aux=False):
            x = torch.ones((b["labels"].shape[0], 8))
            logits = self.head(F.relu(self.backbone(x)))
            spans = torch.zeros_like(b["targets"], dtype=torch.float32)
            return (logits, spans, None) if with_aux else (logits, spans)

    fit_rows = []
    for i, lbl in enumerate(LABELS):
        fit_rows.append({"id": f"r{i}", "text": f"mẫu thử thứ {i} tiếng việt", "nhan": lbl, "bang_chung": [], "meta": {"loai": "that"}})
        fit_rows.append({"id": f"s{i}", "text": f"câu sinh thứ {i} tiếng việt", "nhan": lbl, "bang_chung": [], "meta": {"loai": "sinh"}})

    def _write_json(p, o):
        p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf8")

    def _fake_save_file(state, path):
        torch.save(state, path)

    for scheme in ("final_100_human_cal", "final_90_10"):
        # DualORC cell 12
        rdir_d = tmp_path / f"dualorc_{scheme}"
        rdir_d.mkdir()
        _prep_d = lambda r: dict(row=r, label=LABELS.index(r["nhan"]), words_meta=[(0, 3, 0, 0, True, 0, 0, [1], 1.0)])

        def _batches_d(items, shuffle=False, epoch=0, batch_size=4):
            out = []
            for i in range(0, len(items), batch_size):
                exs = [it[0] if isinstance(it, tuple) else it for it in items[i:i + batch_size]]
                out.append(dict(ids=torch.ones((len(exs), 4), dtype=torch.long), labels=torch.tensor([e["label"] for e in exs]), examples=exs, pairs=[]))
            return out

        env_d = {
            "os": os, "re": re, "gc": gc, "math": math, "time": time, "random": random,
            "unicodedata": unicodedata,
            "Counter": Counter, "defaultdict": defaultdict, "islice": islice, "nullcontext": nullcontext,
            "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "save_file": _fake_save_file, "f1_score": f1_score,
            "LABELS": LABELS, "SEED": 42, "RUN": rdir_d, "DEVICE": torch.device("cpu"), "DTYPE": torch.float32,
            "DATA_SCHEME": scheme, "CKPT_SELECT_MODE": "real_aligned",
            "STRATUM_WEIGHT": True, "REAL_RARE_CAP": 2.5, "SEVERITY_TARGETS": [0.0, 0.45, 0.65, 1.0, 0.85],
            "LABEL_SMOOTHING": 0.02, "FOCAL_GAMMA": 1.0, "MUTEX_WEIGHT": 0.1, "ORC_AUX_WEIGHT": 0.2,
            "PROTO_WEIGHT": 0.0, "SPAN_WEIGHT": 0.0, "SPAN_DICE_WEIGHT": 0.0, "LEG_POLAR_WEIGHT": 0.0,
            "CONSISTENCY_WEIGHT": 0.0, "USE_BIO_CRF": False, "USE_ORC": True,
            "ACCENT_AUG": 0.0, "SURFACE_AUG": 0.0, "FGM_EPS": 0.5,
            "SPEC": {"num_layers": 2}, "LLRD": 0.92, "LLRD_BOTTOM_RATIO": 0.92 ** 12,
            "LR_BACKBONE": 3e-5, "LR_HEAD": 3e-4, "BATCH": 4, "ACCUM": 1, "EVAL_BATCH": 4, "EPOCHS": 2,
            "config": {"scheme": scheme},
            "model": _TinyDualORC(),
            "parts": {"fit": fit_rows},
            "encoded": {"fit": [_prep_d(r) for r in fit_rows]},
            "prepare": _prep_d, "annotations": lambda r: ([], []), "overlaps": lambda a, b: False,
            "augment_accents": lambda r, rng: r, "augment_surface": lambda r, rng: r, "strip_accents": lambda s: s,
            "batches": _batches_d, "split_microbatch": lambda b: [b], "to_device": lambda b, dev: b,
            "amp": nullcontext, "seed_all": lambda s: None, "resolve_effective_llrd": lambda n, l, br: 0.92,
            "write_json": _write_json,
        }
        exec(compile(_cell_source(dualorc_nb, 12), f"<dualorc_c12_{scheme}>", "exec"), env_d)
        hist_d = json.loads((rdir_d / "history.json").read_text(encoding="utf-8"))
        assert len(hist_d) == 2 and all(h["mode"] == f"fixed_epoch_{scheme}" for h in hist_d)
        assert (rdir_d / "model_best.safetensors").exists() and (rdir_d / "last.pt").exists()
        # Kiểm tra nhánh khôi phục trực tiếp khi đã đủ EPOCHS
        exec(compile(_cell_source(dualorc_nb, 12), f"<dualorc_c12_resume_{scheme}>", "exec"), dict(env_d, model=_TinyDualORC()))

        # GuardMerge cell 12
        rdir_g = tmp_path / f"guardmerge_{scheme}"
        rdir_g.mkdir()
        _prep_g = lambda r: dict(row=r, label=LABELS.index(r["nhan"]), chunks=[(0, [1, 2], [(0, 2), (2, 4)], [0, 0])])

        def _batches_g(items, shuffle=False, epoch=0, batch_size=4):
            out = []
            for i in range(0, len(items), batch_size):
                exs = [it[0] if isinstance(it, tuple) else it for it in items[i:i + batch_size]]
                out.append(dict(targets=torch.zeros((len(exs), 3), dtype=torch.long), labels=torch.tensor([e["label"] for e in exs]), examples=exs, pairs=[]))
            return out

        env_g = {
            "os": os, "re": re, "gc": gc, "math": math, "time": time, "random": random,
            "unicodedata": unicodedata,
            "Counter": Counter, "defaultdict": defaultdict, "islice": islice, "nullcontext": nullcontext,
            "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "save_file": _fake_save_file, "f1_score": f1_score,
            "LABELS": LABELS, "SEED": 42, "RUN": rdir_g, "DEVICE": torch.device("cpu"), "DTYPE": torch.float32,
            "DATA_SCHEME": scheme, "CKPT_SELECT_MODE": "real_aligned",
            "STRATUM_WEIGHT": True, "REAL_RARE_CAP": 2.5, "HEAD_ARCH": "v4",
            "LABEL_SMOOTHING": 0.02, "POS_WEIGHT_CAP": 3.0, "CONSISTENCY_WEIGHT": 0.0,
            "SPAN_WEIGHT": 0.0, "SPAN_DICE_WEIGHT": 0.0, "PROTOTYPE_WEIGHT": 0.0, "WORD_SCORE": "max",
            "ACCENT_AUG": 0.0, "SURFACE_AUG": 0.0,
            "LR_ADAPTER": 1e-4, "LR_HEAD": 4e-4, "BATCH": 4, "ACCUM": 1, "EVAL_BATCH": 4, "EPOCHS": 2,
            "config": {"scheme": scheme},
            "model": _TinyGuardMerge(),
            "parts": {"fit": fit_rows},
            "encoded": {"fit": [_prep_g(r) for r in fit_rows]},
            "prepare": _prep_g, "annotations": lambda r: ([], []), "overlaps": lambda a, b: False,
            "augment_accents": lambda r, rng: r, "augment_surface": lambda r, rng: r, "strip_accents": lambda s: s,
            "batches": _batches_g, "split_microbatch": lambda b: [b], "to_device": lambda b, dev: b,
            "amp": nullcontext, "seed_all": lambda s: None, "write_json": _write_json,
        }
        exec(compile(_cell_source(guardmerge_nb, 12), f"<guardmerge_c12_{scheme}>", "exec"), env_g)
        hist_g = json.loads((rdir_g / "history.json").read_text(encoding="utf-8"))
        assert len(hist_g) == 2 and all(h["mode"] == f"fixed_epoch_{scheme}" for h in hist_g)
        assert (rdir_g / "adapter.safetensors").exists() and (rdir_g / "last.pt").exists()
        exec(compile(_cell_source(guardmerge_nb, 12), f"<guardmerge_c12_resume_{scheme}>", "exec"), dict(env_g, model=_TinyGuardMerge()))


def test_cell12_dev_checkpoint_selection_and_resume_config_guard(dualorc_nb, guardmerge_nb, tmp_path):
    """Thực thi thật nhánh `has_dev=True` (dev_80_10_10) của ô 12 trên cả hai notebook để kiểm định:
    (1) Vòng lặp ghi đúng `select_score` theo công thức giảm phương sai mới và khôi phục đúng `last.pt` cùng `config`.
    (2) Từ chối ngay `last.pt` cũ từ công thức trước (thiếu hoặc khác `ckpt_select_weights`), chặn lỗi trộn thang điểm ~0.78 vs ~0.82."""
    import math
    import os
    import random
    import time
    from contextlib import nullcontext
    from itertools import islice
    import pandas as pd

    class _TinyDualORC(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.config = type("Cfg", (), {"num_hidden_layers": 2})()
            self.backbone.embeddings = nn.Embedding(32, 8)
            self.backbone.get_input_embeddings = lambda: self.backbone.embeddings
            self.backbone.encoder = nn.Module()
            self.backbone.encoder.layer = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
            self.cls_head = nn.Linear(8, 5)
            self.sev_head = nn.Linear(8, 1)
            self.harm_head = nn.Linear(8, 1)
            self.orc_blend_logit = nn.Parameter(torch.zeros(()))

        def forward(self, b, with_aux=False):
            h = self.backbone.embeddings(b["ids"]).mean(dim=1)
            for lyr in self.backbone.encoder.layer:
                h = F.relu(lyr(h))
            return dict(
                logits=self.cls_head(h),
                harm_logit=self.harm_head(h).squeeze(-1),
                severity=self.sev_head(h).squeeze(-1).sigmoid(),
                proto=None,
                emissions=torch.zeros((b["labels"].shape[0], 1)),
                ev_prob=torch.zeros((b["labels"].shape[0], 1)),
            )

    class _TinyGuardMerge(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.head = nn.Linear(8, 5)

        def forward(self, b, with_aux=False):
            x = torch.ones((b["labels"].shape[0], 8))
            logits = self.head(F.relu(self.backbone(x)))
            spans = torch.zeros_like(b["targets"], dtype=torch.float32)
            return (logits, spans, None) if with_aux else (logits, spans)

    rows_toy = []
    for i, lbl in enumerate(LABELS):
        rows_toy.append({"id": f"t{i}", "text": f"mẫu thật thứ {i}", "nhan": lbl, "bang_chung": [], "meta": {"loai": "that"}})
        rows_toy.append({"id": f"s{i}", "text": f"mẫu sinh thứ {i}", "nhan": lbl, "bang_chung": [], "meta": {"loai": "sinh"}})

    def _write_json(p, o):
        p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf8")

    def _fake_save_file(state, path):
        torch.save(state, path)

    # DualORC dev_80_10_10 (SPAN_WEIGHT=0.0 -> công thức 0.55/0.20/0.25)
    rdir_d = tmp_path / "dualorc_dev"
    rdir_d.mkdir()
    _prep_d = lambda r: dict(row=r, label=LABELS.index(r["nhan"]), n_words=1, words_meta=[(0, 3, 0, 0, True, 0, 0, [1], 1.0)])

    def _batches_d(items, shuffle=False, epoch=0, batch_size=4):
        out = []
        for i in range(0, len(items), batch_size):
            exs = [it[0] if isinstance(it, tuple) else it for it in items[i:i + batch_size]]
            out.append(dict(ids=torch.ones((len(exs), 4), dtype=torch.long), labels=torch.tensor([e["label"] for e in exs]), examples=exs, pairs=[]))
        return out

    cfg_d = {"ckpt_select_mode": "real_aligned", "ckpt_select_weights": [0.55, 0.20, 0.25, 0.0]}
    env_d = {
        "os": os, "re": re, "gc": gc, "math": math, "time": time, "random": random, "unicodedata": unicodedata,
        "Counter": Counter, "defaultdict": defaultdict, "islice": islice, "nullcontext": nullcontext,
        "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "save_file": _fake_save_file, "f1_score": f1_score,
        "LABELS": LABELS, "SEED": 42, "RUN": rdir_d, "DEVICE": torch.device("cpu"), "DTYPE": torch.float32,
        "DATA_SCHEME": "dev_80_10_10", "CKPT_SELECT_MODE": "real_aligned",
        "STRATUM_WEIGHT": True, "REAL_RARE_CAP": 2.5, "SEVERITY_TARGETS": [0.0, 0.45, 0.65, 1.0, 0.85],
        "LABEL_SMOOTHING": 0.02, "FOCAL_GAMMA": 1.0, "MUTEX_WEIGHT": 0.1, "ORC_AUX_WEIGHT": 0.2,
        "PROTO_WEIGHT": 0.0, "SPAN_WEIGHT": 0.0, "SPAN_DICE_WEIGHT": 0.0, "LEG_POLAR_WEIGHT": 0.0,
        "CONSISTENCY_WEIGHT": 0.0, "USE_BIO_CRF": False, "USE_ORC": True,
        "ACCENT_AUG": 0.0, "SURFACE_AUG": 0.0, "FGM_EPS": 0.0,
        "SPEC": {"num_layers": 2}, "LLRD": 0.92, "LLRD_BOTTOM_RATIO": 0.92 ** 12,
        "LR_BACKBONE": 5e-5, "LR_HEAD": 3e-4, "BATCH": 4, "ACCUM": 1, "EVAL_BATCH": 4, "EPOCHS": 2,
        "config": cfg_d, "model": _TinyDualORC(),
        "parts": {"fit": rows_toy, "dev": rows_toy},
        "encoded": {"fit": [_prep_d(r) for r in rows_toy], "dev": [_prep_d(r) for r in rows_toy]},
        "prepare": _prep_d, "annotations": lambda r: ([], []), "overlaps": lambda a, b: False,
        "augment_accents": lambda r, rng: r, "augment_surface": lambda r, rng: r, "strip_accents": lambda s: s,
        "batches": _batches_d, "split_microbatch": lambda b: [b], "to_device": lambda b, dev: b,
        "amp": nullcontext, "seed_all": lambda s: None, "resolve_effective_llrd": lambda n, l, br: 0.92,
        "get_crf_params_np": lambda _m: None,
        "write_json": _write_json,
    }
    exec(compile(_cell_source(dualorc_nb, 12), "<dualorc_c12_dev>", "exec"), env_d)
    hist_d = json.loads((rdir_d / "history.json").read_text(encoding="utf-8"))
    assert len(hist_d) == 2
    for row in hist_d:
        exp_sc = round(0.55 * row["macro_f1"] + 0.20 * row["that_macro_f1"] + 0.25 * row["that_acc"], 6)
        assert abs(row["select_score"] - exp_sc) < 1e-6
    # Từ chối resume khi last.pt mang config cũ (thiếu hoặc khác ckpt_select_weights)
    stale_d = dict(env_d, model=_TinyDualORC(), config={"ckpt_select_mode": "real_aligned"})
    with pytest.raises(AssertionError):
        exec(compile(_cell_source(dualorc_nb, 12), "<dualorc_c12_stale>", "exec"), stale_d)

    # GuardMerge dev_80_10_10 (SPAN_WEIGHT=0.35 -> công thức 0.50/0.15/0.25/0.10)
    rdir_g = tmp_path / "guardmerge_dev"
    rdir_g.mkdir()
    _prep_g = lambda r: dict(row=r, label=LABELS.index(r["nhan"]), chunks=[(0, [1, 2], [(0, 3), (4, 8)], [0, 0])])

    def _batches_g(items, shuffle=False, epoch=0, batch_size=4):
        out = []
        for i in range(0, len(items), batch_size):
            exs = [it[0] if isinstance(it, tuple) else it for it in items[i:i + batch_size]]
            out.append(dict(
                targets=torch.zeros((len(exs), 3), dtype=torch.long),
                labels=torch.tensor([e["label"] for e in exs]),
                owner=torch.arange(len(exs), dtype=torch.long),
                offsets=[[(0, 3), (4, 8)] for _ in exs],
                examples=exs, pairs=[],
            ))
        return out

    cfg_g = {"ckpt_select_mode": "real_aligned", "ckpt_select_weights": [0.50, 0.15, 0.25, 0.10]}
    env_g = {
        "os": os, "re": re, "gc": gc, "math": math, "time": time, "random": random, "unicodedata": unicodedata,
        "Counter": Counter, "defaultdict": defaultdict, "islice": islice, "nullcontext": nullcontext,
        "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "save_file": _fake_save_file, "f1_score": f1_score,
        "LABELS": LABELS, "SEED": 42, "RUN": rdir_g, "DEVICE": torch.device("cpu"), "DTYPE": torch.float32,
        "DATA_SCHEME": "dev_80_10_10", "CKPT_SELECT_MODE": "real_aligned",
        "STRATUM_WEIGHT": True, "REAL_RARE_CAP": 2.5, "HEAD_ARCH": "v4",
        "LABEL_SMOOTHING": 0.02, "POS_WEIGHT_CAP": 3.0, "CONSISTENCY_WEIGHT": 0.0,
        "SPAN_WEIGHT": 0.35, "SPAN_DICE_WEIGHT": 0.0, "PROTOTYPE_WEIGHT": 0.0, "WORD_SCORE": "max",
        "ACCENT_AUG": 0.0, "SURFACE_AUG": 0.0,
        "LR_ADAPTER": 1e-4, "LR_HEAD": 4e-4, "BATCH": 4, "ACCUM": 1, "EVAL_BATCH": 4, "EPOCHS": 2,
        "config": cfg_g, "model": _TinyGuardMerge(),
        "parts": {"fit": rows_toy, "dev": rows_toy},
        "encoded": {"fit": [_prep_g(r) for r in rows_toy], "dev": [_prep_g(r) for r in rows_toy]},
        "prepare": _prep_g, "annotations": lambda r: ([], []), "overlaps": lambda a, b: False,
        "augment_accents": lambda r, rng: r, "augment_surface": lambda r, rng: r, "strip_accents": lambda s: s,
        "batches": _batches_g, "split_microbatch": lambda b: [b], "to_device": lambda b, dev: b,
        "amp": nullcontext, "seed_all": lambda s: None, "write_json": _write_json,
    }
    exec(compile(_cell_source(guardmerge_nb, 12), "<guardmerge_c12_dev>", "exec"), env_g)
    hist_g = json.loads((rdir_g / "history.json").read_text(encoding="utf-8"))
    assert len(hist_g) == 2
    for row in hist_g:
        exp_sc = round(
            0.50 * row["macro_f1"] + 0.15 * row["that_macro_f1"] + 0.25 * row["that_acc"] + 0.10 * row["evidence_f1"],
            6,
        )
        assert abs(row["select_score"] - exp_sc) < 1e-6
    stale_g = dict(env_g, model=_TinyGuardMerge(), config={"ckpt_select_mode": "real_aligned", "ckpt_select_weights": [0.40, 0.30, 0.20, 0.10]})
    with pytest.raises(AssertionError):
        exec(compile(_cell_source(guardmerge_nb, 12), "<guardmerge_c12_stale>", "exec"), stale_g)


def test_augment_surface_strict_phi_diem_mu_and_guardmerge_dual_view_js(dualorc_nb, guardmerge_nb):
    """Kiểm định tăng cường trực tuyến phi điểm mù (hỗ trợ từ >=3 chữ, tuyệt đối không rò C2/C1 held-out)
    và Dual-View JS Consistency + ghép cặp microbatch của GuardMerge Schema v6."""
    import random

    probe = {
        "id": "aug_len3",
        "text": "Không được nói mày ngu cút đi nhé!\nTao sẽ không tha cho đồ bẩn thỉu đâu.",
        "nhan": "XUC_PHAM",
        "bang_chung": ["mày ngu cút đi", "đồ bẩn thỉu"],
        "so_dong": 2,
    }
    forbidden_re = re.compile(r"[\u200b\u200c\u200d\ufeffЀ-ӿ！-～·*/|+^589]")
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        ns8 = _exec_defs(
            _cell_source(nb, 8),
            {
                "strip_accents", "C1_TEENCODE_MAP", "C1_LOOKALIKE", "C1_SEPS", "_ALPHA_CORE_RE",
                "_split_affix", "_map_pos", "transform_line_with_spans", "_apply_line_transforms",
                "augment_accents", "_c1_lookalike_word", "_c1_sep_word", "_c1_elongate_word", "augment_surface",
            },
            {"re": re, "unicodedata": unicodedata},
        )
        rng = random.Random(42)
        # Xác nhận từ 3 chữ cái ("ngu", "mày", "cút") nhận được lookalike/sep hợp lệ (không bắt đầu bằng số, alpha > digit)
        seen_len3_sep = False
        seen_len3_look = False
        for _ in range(80):
            s_ngu = ns8["_c1_sep_word"]("ngu", rng)
            if s_ngu != "ngu" and any(c in s_ngu for c in ".-_"):
                seen_len3_sep = True
            l_may = ns8["_c1_lookalike_word"]("may", rng)
            if l_may != "may":
                seen_len3_look = True
                assert not l_may[0].isdigit()
                assert sum(c.isalpha() for c in l_may) > sum(c.isdigit() or c == "@" for c in l_may)
        assert seen_len3_sep and seen_len3_look, f"{label}: từ 3 chữ chưa được tăng cường C1 đơn nghĩa"

        # 200 lượt augment_surface: tuyệt đối không sinh C2 / held-out (5,8,9,·,cyrillic,zw,fullwidth), giữ nguyên số từ & bằng chứng verbatim
        orig_words = len(probe["text"].split())
        for _ in range(200):
            out_r = ns8["augment_surface"](probe, rng)
            assert not forbidden_re.search(out_r["text"]), f"{label}: rò ký tự điểm mù/held-out: {out_r['text']}"
            assert len(out_r["text"].split()) == orig_words
            assert len(out_r["bang_chung"]) == len(probe["bang_chung"])
            assert all(any(ph in ln for ln in out_r["text"].split("\n")) for ph in out_r["bang_chung"])

    # Kiểm định GuardMerge Dual-View JS Consistency: collate ghép cặp, split_microbatch không cắt đôi cặp, JS loss > 0 khi lệch
    ns8_g = _exec_defs(
        _cell_source(guardmerge_nb, 8),
        {"collate", "_item_sort_key", "split_microbatch"},
        {"torch": torch, "BOS": 1, "PAD": 0, "MAX_MICROBATCH_TOKENS": 20},
    )
    e0 = dict(row={"id": "a", "meta": {"loai": "that"}}, label=1, chunks=[(0, [10, 11, 12], [(0, 1), (1, 2), (2, 3)], [0, 1, 0])], max_len=4, n_chunks=1)
    e1 = dict(row={"id": "a", "meta": {"loai": "that"}}, label=1, chunks=[(0, [20, 21, 22, 23], [(0, 1), (1, 2), (2, 3), (3, 4)], [0, 1, 0, 0])], max_len=5, n_chunks=1)
    e2 = dict(row={"id": "b", "meta": {"loai": "sinh"}}, label=0, chunks=[(0, [30, 31, 32], [(0, 1), (1, 2), (2, 3)], [0, 0, 0])], max_len=4, n_chunks=1)
    b_all = ns8_g["collate"]([(e0, e1), e2])
    assert b_all["pairs"] == [(0, 1)] and len(b_all["examples"]) == 3
    subs = ns8_g["split_microbatch"](b_all, max_tokens=10)
    assert len(subs) == 2
    assert subs[0]["pairs"] == [(0, 1)] and len(subs[0]["examples"]) == 2  # cặp (e0, e1) không bị cắt đôi!
    assert subs[1]["pairs"] == [] and len(subs[1]["examples"]) == 1

    base_env_obj = {
        "torch": torch, "F": F, "DEVICE": torch.device("cpu"), "STRATUM_WEIGHT": False,
        "stratum_weights": {0: 1.0, 1: 1.0}, "LABEL_SMOOTHING": 0.0, "SPAN_WEIGHT": 0.0,
        "SPAN_DICE_WEIGHT": 0.0, "PROTOTYPE_WEIGHT": 0.0, "pos_weight": torch.tensor(1.0),
    }
    ns12_on = _exec_defs(_cell_source(guardmerge_nb, 12), {"sample_weights", "js_divergence_probs", "objective"}, {**base_env_obj, "CONSISTENCY_WEIGHT": 0.25})
    ns12_off = _exec_defs(_cell_source(guardmerge_nb, 12), {"sample_weights", "js_divergence_probs", "objective"}, {**base_env_obj, "CONSISTENCY_WEIGHT": 0.0})
    logits_same = torch.tensor([[0.1, 2.5, -0.5, 0.0, 0.0], [0.1, 2.5, -0.5, 0.0, 0.0], [2.5, 0.0, 0.0, 0.0, 0.0]])
    logits_diff = torch.tensor([[0.1, 2.5, -0.5, 0.0, 0.0], [2.5, 0.1, -0.5, 0.0, 0.0], [2.5, 0.0, 0.0, 0.0, 0.0]])
    spans_z = torch.zeros_like(b_all["targets"], dtype=torch.float32)
    mass = torch.tensor(3.0)
    l_same_on = float(ns12_on["objective"](logits_same, spans_z, None, b_all, mass, 1))
    l_same_off = float(ns12_off["objective"](logits_same, spans_z, None, b_all, mass, 1))
    l_diff_on = float(ns12_on["objective"](logits_diff, spans_z, None, b_all, mass, 1))
    l_diff_off = float(ns12_off["objective"](logits_diff, spans_z, None, b_all, mass, 1))
    assert abs(l_same_on - l_same_off) < 1e-6
    assert l_diff_on > l_diff_off + 0.02


def test_normalize_with_alignment_and_infer_normalized_verbatim_across_c0_c1_c2(dualorc_nb, guardmerge_nb):
    """Kiểm định chuẩn hóa 5 tầng giữ dòng (\\n) + ánh xạ ngược span verbatim trên các ca biên
    (teencode nở nhiều từ đm/vcl, lookalike, phân cách, kéo dài, giữ 400k/3.14, C2 zero_width/fullwidth/space_injection)."""
    teencode_map = json.loads((ROOT / "src" / "shared" / "normalization" / "teencode_dict.json").read_text(encoding="utf-8"))
    edge_rows = [
        {
            "id": "e1",
            "text": "Chào bạn, giá 400k hay 3.14 vậy?\nĐm m4y n.g.u quáaaaa vcl luôn!",
            "nhan": "XUC_PHAM",
            "bang_chung": ["Đm m4y n.g.u quáaaaa vcl"],
        },
        {
            "id": "e2",
            "text": "kbt nói gì nữa\nn\u200bgu ｈếｔ phần thiên hạ",
            "nhan": "XUC_PHAM",
            "bang_chung": ["n\u200bgu ｈếｔ phần thiên hạ"],
        },
    ]
    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        env = {
            "re": re, "np": np, "unicodedata": unicodedata, "defaultdict": defaultdict,
            "LABELS": LABELS, "WORD_SCORE": "max", "USE_BIO_CRF": (label == "dualorc"),
            "SPAN_WEIGHT": 0.45,
        }
        env.update(_exec_defs(_cell_source(nb, 8), {"annotations", "overlaps"}, env))
        want12 = {
            "_NORM_LOOKALIKE", "_NORM_SEP_RE", "_NORM_ELONG_RE", "_NORM_LEAD_RE", "_NORM_TRAIL_RE",
            "_norm_strip_base", "_normalize_v1_token", "normalize_with_alignment", "infer_normalized",
            "_apply_empty_fallback", "select_word_marks",
        }
        if label == "dualorc":
            want12.update({"_apply_marginal_postproc", "extract_evidence_runs", "_viterbi_single_np", "_decode_item_crf_tags"})
        ns12 = _exec_defs(_cell_source(nb, 12), want12, env)

        # Kiểm tra bảo toàn \n, giữ 400k & 3.14, nở đm -> địt mẹ vẫn ánh xạ đầy đủ về từ gốc
        nt0, mp0 = ns12["normalize_with_alignment"](edge_rows[0]["text"], teencode_map)
        assert nt0.count("\n") == 1
        assert "400k" in nt0 and "3.14" in nt0
        assert "ngu" in nt0 and "quá" in nt0 and "địt mẹ" in nt0.lower()
        assert set(mp0) == set(range(len(list(re.finditer(r"\S+", edge_rows[0]["text"])))))

        # Giả lập infer trên văn bản đã chuẩn hóa và kiểm tra chiếu ngược về văn bản gốc 100% verbatim
        def _stub_prep(r):
            return dict(row=r)

        def _stub_infer(_net, data):
            out = []
            for e in data:
                r = e["row"]
                ws = []
                ems = []
                for m in re.finditer(r"\S+", r["text"]):
                    w = m.group(0).lower()
                    is_bad = any(k in w for k in ("địt", "mẹ", "ngu", "vãi", "lồn", "n\u200bgu", "ｈếｔ"))
                    sc = 0.90 if is_bad else 0.05
                    if label == "dualorc":
                        ws.append((m.start(), m.end(), sc, 0, 0, False))
                        if "mẹ" in w:
                            # Mô phỏng từ thứ 2 của cụm nở "Địt mẹ" mang I-EV cao hơn B-EV
                            ems.append(np.array([-0.5, -1.5, 3.5], dtype=np.float32))
                            ws[-1] = (m.start(), m.end(), 0.96, 0, 0, False)
                        else:
                            ems.append(np.array([-2.0, 2.5, 1.5] if is_bad else [2.5, -1.0, -1.0], dtype=np.float32))
                    else:
                        ws.append((m.start(), m.end(), sc, 0))
                it = dict(id=r["id"], text=r["text"], gold=1, gold_phrases=[], logits=[-1.0, 3.0, -1.0, -1.0, -1.0], words=ws)
                if label == "dualorc":
                    it.update(severity=0.45, emissions=np.asarray(ems, dtype=np.float32))
                out.append(it)
            return out

        proj = ns12["infer_normalized"](None, edge_rows, teencode_map, infer_fn=_stub_infer, prep_fn=_stub_prep)
        assert len(proj) == len(edge_rows)
        fake_crf_np = dict(
            start=np.array([0.2, 0.0, -5.0]),
            end=np.zeros(3),
            trans=np.array([[0.5, 0.0, -5.0], [-0.2, -0.3, 0.6], [0.0, -0.4, 0.6]]),
        )
        for r, item in zip(edge_rows, proj):
            assert item["text"] == r["text"]
            assert len(item["words"]) == len(list(re.finditer(r"\S+", r["text"])))
            if label == "dualorc":
                assert item["emissions"].shape == (len(item["words"]), 3)
                runs = ns12["extract_evidence_runs"](item, r["text"], threshold=0.5, gate=True, mode="marginal")
                runs_crf = ns12["extract_evidence_runs"](item, r["text"], crf_np=fake_crf_np, gate=True, mode="crf")
                assert runs_crf, f"dualorc CRF: không trích được bằng chứng sau ánh xạ ngược cho {r['id']}"
                for a, b in runs_crf:
                    phrase = r["text"][a:b]
                    assert phrase and "\n" not in phrase and any(phrase in ln for ln in r["text"].split("\n"))
            else:
                marks = ns12["select_word_marks"](item, threshold=0.5, gate=True)
                runs = [[w[0], w[1]] for w, mk in zip(item["words"], marks) if mk]
            assert runs, f"{label}: không trích được bằng chứng sau ánh xạ ngược cho {r['id']}"
            for a, b in runs:
                phrase = r["text"][a:b]
                assert phrase and "\n" not in phrase and any(phrase in ln for ln in r["text"].split("\n"))
        # Đảm bảo từ "Đm" ở e1 được tô (cả marginal lẫn CRF Viterbi ở đầu lượt, không rơi vì I-EV bị chặn đầu lượt!)
        e1_marked_texts = [edge_rows[0]["text"][w[0]:w[1]] for w in proj[0]["words"] if w[2] >= 0.5]
        assert "Đm" in e1_marked_texts and "n.g.u" in e1_marked_texts and "vcl" in e1_marked_texts
        if label == "dualorc":
            crf_phrases_e1 = [edge_rows[0]["text"][a:b] for a, b in ns12["extract_evidence_runs"](proj[0], edge_rows[0]["text"], crf_np=fake_crf_np, gate=True, mode="crf")]
            assert any("Đm" in ph for ph in crf_phrases_e1), f"Viterbi làm rơi Đm đầu lượt: {crf_phrases_e1}"


def test_dual_view_js_microbatch_invariance_both_notebooks(dualorc_nb, guardmerge_nb):
    """Đảm bảo hàm mất mát Dual-View JS Consistency bất biến tuyệt đối trước split_microbatch & ACCUM trên cả 2 nhánh."""
    # 1) GuardMerge: 1 cửa sổ gồm 2 cặp + 2 mẫu đơn tách thành 2 microbatch lệch số cặp (2 cặp vs 0 cặp, và 1 cặp + 2 đơn vs 1 cặp)
    ns8_g = _exec_defs(
        _cell_source(guardmerge_nb, 8),
        {"collate"},
        {"torch": torch, "BOS": 1, "PAD": 0},
    )
    ns12_g = _exec_defs(
        _cell_source(guardmerge_nb, 12),
        {"sample_weights", "loss_denominators", "js_divergence_probs", "objective"},
        {
            "torch": torch, "F": F, "DEVICE": torch.device("cpu"), "STRATUM_WEIGHT": False,
            "stratum_weights": {k: 1.0 for k in range(5)}, "LABEL_SMOOTHING": 0.02,
            "SPAN_WEIGHT": 0.0, "SPAN_DICE_WEIGHT": 0.0, "PROTOTYPE_WEIGHT": 0.0,
            "CONSISTENCY_WEIGHT": 0.25, "pos_weight": torch.tensor(1.0),
        },
    )

    def _gm_ex(idx, lbl):
        return dict(row={"id": f"g{idx}", "meta": {"loai": "that"}}, label=lbl, chunks=[(0, [10 + idx, 11 + idx], [(0, 1), (1, 2)], [0, 0])], max_len=3, n_chunks=1)

    items_g = [(_gm_ex(0, 1), _gm_ex(1, 1)), _gm_ex(2, 0), _gm_ex(3, 2), (_gm_ex(4, 3), _gm_ex(5, 3))]
    full_g = ns8_g["collate"](items_g)
    subs_g = [ns8_g["collate"](items_g[:3]), ns8_g["collate"](items_g[3:])]  # sub0: 1 pair + 2 singles; sub1: 1 pair

    torch.manual_seed(11)
    z_full_g = torch.randn((len(full_g["examples"]), 5), requires_grad=True)
    sp_full_g = torch.zeros_like(full_g["targets"], dtype=torch.float32)
    cm_f, kc_f, pc_f = ns12_g["loss_denominators"]([full_g])
    loss_full_g = ns12_g["objective"](z_full_g, sp_full_g, None, full_g, cm_f, kc_f, pc_f)
    loss_full_g.backward()

    z_split_g = z_full_g.detach().clone().requires_grad_(True)
    cm_s, kc_s, pc_s = ns12_g["loss_denominators"](subs_g)
    assert pc_f == pc_s == 2
    loss_split_g = torch.zeros(())
    offset = 0
    for sb in subs_g:
        n_ex = len(sb["examples"])
        sp_s = torch.zeros_like(sb["targets"], dtype=torch.float32)
        loss_split_g = loss_split_g + ns12_g["objective"](z_split_g[offset:offset + n_ex], sp_s, None, sb, cm_s, kc_s, pc_s)
        offset += n_ex
    loss_split_g.backward()
    assert abs(float(loss_full_g.detach()) - float(loss_split_g.detach())) < 1e-5
    torch.testing.assert_close(z_full_g.grad, z_split_g.grad, atol=1e-5, rtol=1e-5)

    # 2) DualORC: kiểm định bất biến cho cả JS phân loại lẫn ev_prob MSE mức từ
    ns8_d = _exec_defs(
        _cell_source(dualorc_nb, 8),
        {"collate"},
        {"torch": torch, "PAD_ID": 0, "MAX_WORD_BYTES": 8},
    )
    ns12_d = _exec_defs(
        _cell_source(dualorc_nb, 12),
        {"sample_weights", "js_divergence_probs", "objective"},
        {
            "torch": torch, "F": F, "DEVICE": torch.device("cpu"), "STRATUM_WEIGHT": False,
            "stratum_weights": {k: 1.0 for k in range(5)}, "LABEL_SMOOTHING": 0.02, "FOCAL_GAMMA": 1.0,
            "MUTEX_WEIGHT": 0.0, "ORC_AUX_WEIGHT": 0.0, "PROTO_WEIGHT": 0.0,
            "SPAN_WEIGHT": 0.45, "SPAN_DICE_WEIGHT": 0.0, "LEG_POLAR_WEIGHT": 0.0,
            "USE_BIO_CRF": False, "bin_pos_weight": torch.tensor(1.0),
            "CONSISTENCY_WEIGHT": 0.25, "SEVERITY_TENSOR": torch.tensor([0.0, 0.45, 0.65, 1.0, 0.85]),
        },
    )

    def _d_ex(idx, lbl, n_w=3):
        wm = [(i * 2, i * 2 + 1, 0, 0, i == 0, -100, -1, [65], 1.0) for i in range(n_w)]
        return dict(row={"id": f"d{idx}", "meta": {"loai": "that"}}, label=lbl, ids=[1, 2, 3], n_tokens=3, n_words=n_w, words_meta=wm, word_sub_idxs=[[1]] * n_w)

    items_d = [(_d_ex(0, 1, 3), _d_ex(1, 1, 3)), _d_ex(2, 0, 4), _d_ex(3, 2, 2), (_d_ex(4, 4, 3), _d_ex(5, 4, 3))]
    full_d = ns8_d["collate"](items_d)
    subs_d = [ns8_d["collate"](items_d[:3]), ns8_d["collate"](items_d[3:])]
    max_w_full = full_d["word_mask"].size(1)

    torch.manual_seed(22)
    z_fd = torch.randn((len(full_d["examples"]), 5), requires_grad=True)
    ev_fd = torch.rand((len(full_d["examples"]), max_w_full), requires_grad=True)
    em_fd = torch.zeros((len(full_d["examples"]), max_w_full))
    cm_fd = ns12_d["sample_weights"](full_d).sum().clamp_min(1e-6)
    pc_fd = max(len(full_d["pairs"]), 1)
    out_fd = dict(logits=z_fd, emissions=em_fd, ev_prob=ev_fd, proto=None, harm_logit=None, severity=None)
    loss_fd = ns12_d["objective"](None, out_fd, full_d, cm_fd, use_consistency=True, pair_count=pc_fd)
    loss_fd.backward()

    z_sd = z_fd.detach().clone().requires_grad_(True)
    ev_sd = ev_fd.detach().clone().requires_grad_(True)
    cm_sd = sum(ns12_d["sample_weights"](sb).sum() for sb in subs_d).clamp_min(1e-6)
    pc_sd = max(sum(len(sb["pairs"]) for sb in subs_d), 1)
    loss_sd = torch.zeros(())
    offset = 0
    for sb in subs_d:
        n_ex = len(sb["examples"])
        w_sb = sb["word_mask"].size(1)
        out_sb = dict(
            logits=z_sd[offset:offset + n_ex],
            emissions=torch.zeros((n_ex, w_sb)),
            ev_prob=ev_sd[offset:offset + n_ex, :w_sb],
            proto=None, harm_logit=None, severity=None,
        )
        loss_sd = loss_sd + ns12_d["objective"](None, out_sb, sb, cm_sd, use_consistency=True, pair_count=pc_sd)
        offset += n_ex
    loss_sd.backward()
    assert abs(float(loss_fd.detach()) - float(loss_sd.detach())) < 1e-5
    torch.testing.assert_close(z_fd.grad, z_sd.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ev_fd.grad, ev_sd.grad, atol=1e-5, rtol=1e-5)


def test_dualorc_infer_normalized_crf_viterbi_exact_projection(dualorc_nb):
    """Kiểm định giải mã Viterbi trên chuỗi đã chuẩn hóa rồi chiếu ngược về từ gốc (khắc phục 3 lỗi gộp emission heuristic):
    (1) Từ an toàn đầu lượt nở nhiều từ ('ae' -> 'anh em') có I-EV không huấn luyện cực lớn (+5.0) KHÔNG tạo báo động giả.
    (2) Từ nguy hại đầu lượt nở nhiều từ ('đm mày' -> 'địt mẹ mày') với o_bias=+0.6 vẫn giữ trọn span nhờ trans[B,I].
    (3) Từ nở giữa cụm ('đồ đm mày' -> 'đồ địt mẹ mày') giữ nguyên 1 cụm liền mạch không bị tách đôi và đồng nhất marks/runs.
    """
    teencode_map = json.loads((ROOT / "src" / "shared" / "normalization" / "teencode_dict.json").read_text(encoding="utf-8"))
    env = {
        "re": re, "np": np, "unicodedata": unicodedata, "defaultdict": defaultdict,
        "LABELS": LABELS, "USE_BIO_CRF": True, "SPAN_WEIGHT": 0.45,
    }
    env.update(_exec_defs(_cell_source(dualorc_nb, 8), {"annotations", "overlaps"}, env))
    ns12 = _exec_defs(
        _cell_source(dualorc_nb, 12),
        {
            "_NORM_LOOKALIKE", "_NORM_SEP_RE", "_NORM_ELONG_RE", "_NORM_LEAD_RE", "_NORM_TRAIL_RE",
            "_norm_strip_base", "_normalize_v1_token", "normalize_with_alignment", "infer_normalized",
            "_viterbi_single_np", "_decode_item_crf_tags", "_apply_marginal_postproc",
            "_apply_empty_fallback", "select_word_marks", "extract_evidence_runs",
        },
        env,
    )
    crf_np = dict(
        start=np.array([0.2, 0.1, -1e4]),
        end=np.array([0.1, 0.0, 0.1]),
        trans=np.array([
            [0.8, 0.2, -1e4],
            [-0.1, -0.8, 1.2],
            [0.1, -0.8, 1.0],
        ]),
    )

    def _run_case(text, norm_emissions_fn, o_bias=0.0):
        rec = {"id": "c", "text": text, "nhan": "XUC_PHAM", "bang_chung": []}
        def _stub_infer(_net, data):
            r = data[0]["row"]
            words, ems = [], []
            cur = 0
            for t_idx, ln in enumerate(r["text"].split("\n")):
                first = True
                for m in re.finditer(r"\S+", ln):
                    words.append((cur + m.start(), cur + m.end(), 0.5, 0, t_idx, first))
                    ems.append(norm_emissions_fn(m.group(0).lower(), len(words) - 1, first))
                    first = False
                cur += len(ln) + 1
            return [dict(id="c", text=r["text"], gold=1, gold_phrases=[], logits=[-1.0, 3.0, -1.0, -1.0, -1.0], severity=0.5, words=words, emissions=np.asarray(ems, dtype=np.float32))]
        proj = ns12["infer_normalized"](None, [rec], teencode_map, infer_fn=_stub_infer, prep_fn=lambda r: dict(row=r))[0]
        marks = ns12["select_word_marks"](proj, crf_np=crf_np, o_bias=o_bias, gate=True, mode="crf")
        runs = ns12["extract_evidence_runs"](proj, text, crf_np=crf_np, o_bias=o_bias, gate=True, mode="crf")
        phrases = [text[a:b] for a, b in runs]
        # Đảm bảo select_word_marks và extract_evidence_runs tô cùng tập từ
        run_word_idxs = {i for i, w in enumerate(proj["words"]) if any(w[0] >= a and w[1] <= b for a, b in runs)}
        mark_word_idxs = {i for i, mk in enumerate(marks) if mk}
        assert run_word_idxs == mark_word_idxs
        return marks, phrases

    # Case 1: 'ae ơi' -> 'anh em ơi', pos 0 có I-EV=+5.0 không huấn luyện nhưng thực tế là O
    marks1, phrases1 = _run_case(
        "ae ơi",
        lambda w, idx, is_ts: np.array([2.0, -3.0, 5.0] if idx == 0 else [2.0, -3.0, -2.0], dtype=np.float32),
    )
    assert marks1 == [False, False] and phrases1 == []

    # Case 2: 'đm mày' -> 'địt mẹ mày' với o_bias=+0.6, 'địt' B yếu (0.4) nhưng 'mẹ' I mạnh (4.0)
    marks2, phrases2 = _run_case(
        "đm mày",
        lambda w, idx, is_ts: np.array([0.5, 0.4, -2.0] if idx == 0 else ([-2.0, -1.0, 4.0] if idx == 1 else [-2.0, -1.0, 3.0]), dtype=np.float32),
        o_bias=0.6,
    )
    assert marks2 == [True, True] and phrases2 == ["đm mày"]

    # Case 3: 'đồ đm mày' -> 'đồ địt mẹ mày' giữ 1 cụm nguyên vẹn
    marks3, phrases3 = _run_case(
        "đồ đm mày",
        lambda w, idx, is_ts: np.array([-2.0, 3.0, -2.0] if idx == 0 else [-2.0, 0.1, 3.5], dtype=np.float32),
    )
    assert marks3 == [True, True, True] and phrases3 == ["đồ đm mày"]


def test_cell16_norm_clean_c0_baseline_and_predict_text_normalizer(dualorc_nb, guardmerge_nb, tmp_path):
    """Kiểm định ô 16 của cả hai notebook: xuất clean_c0 (kèm bằng chứng F1 theo ý) trong norm_stress_tests.json
    và hỗ trợ predict_text(..., use_normalizer=True) trả về cụm nguyên văn trên văn bản biến dạng."""
    import random
    import pandas as pd

    teencode_map = json.loads((ROOT / "src" / "shared" / "normalization" / "teencode_dict.json").read_text(encoding="utf-8"))
    sample_rows = [
        {"id": "s1", "text": "Hôm nay trời đẹp quá bạn ơi", "nhan": "AN_TOAN", "bang_chung": [], "so_dong": 1, "nguon": "vihsd", "meta": {"loai": "that"}},
        {"id": "s2", "text": "Đm mày ngu lắm luôn ấy", "nhan": "XUC_PHAM", "bang_chung": ["Đm mày ngu"], "so_dong": 1, "nguon": "vihsd", "meta": {"loai": "that"}},
        {"id": "s3", "text": "Chuyển 500k vào STK để nhận quà ngay", "nhan": "LUA_DAO", "bang_chung": ["Chuyển 500k vào STK"], "so_dong": 1, "nguon": "sinh_llm", "nguon_id": "ld1:0", "meta": {"loai": "sinh"}},
    ]

    for label, nb in [("dualorc", dualorc_nb), ("guardmerge", guardmerge_nb)]:
        rdir = tmp_path / f"c16_{label}"
        rdir.mkdir()

        def _fake_prep(r):
            return dict(row=r)

        def _fake_infer(_net, data, device=None):
            out = []
            for e in data:
                r = e["row"]
                g = LABELS.index(r["nhan"])
                is_bad_txt = any(k in r["text"].lower() for k in ("địt", "mẹ", "ngu", "đm", "chuyển"))
                logits = [-1.0, 3.0, -1.0, -1.0, -1.0] if ("ngu" in r["text"].lower() or "địt" in r["text"].lower()) else (
                    [-1.0, -1.0, -1.0, -1.0, 3.0] if "chuyển" in r["text"].lower() else [3.0, -1.0, -1.0, -1.0, -1.0]
                )
                ws = []
                for m in re.finditer(r"\S+", r["text"]):
                    sc = 0.85 if (is_bad_txt and any(k in m.group(0).lower() for k in ("địt", "mẹ", "ngu", "đm", "chuyển"))) else 0.05
                    ws.append((m.start(), m.end(), sc, int(sc > 0.5), 0, m.start() == 0) if label == "dualorc" else (m.start(), m.end(), sc, int(sc > 0.5)))
                it = dict(id=r["id"], text=r["text"], gold=g, gold_phrases=list(r["bang_chung"]), logits=logits, words=ws)
                if label == "dualorc":
                    it.update(severity=0.5, emissions=None)
                out.append(it)
            return out

        env = {
            "re": re, "np": np, "pd": pd, "random": random, "json": json, "unicodedata": unicodedata,
            "defaultdict": defaultdict, "LABELS": LABELS, "SEED": 42, "RUN": rdir, "DEVICE": torch.device("cpu"),
            "ACCENT_AUG": 0.30, "SURFACE_AUG": 0.20, "EVAL_WITH_NORMALIZER": True, "RUN_REAL_TEST_PERTURB": False,
            "SPAN_WEIGHT": 0.45, "USE_BIO_CRF": False, "WORD_SCORE": "max",
            "DATA_SCHEME": "dev_80_10_10", "crf_np": None,
            "model": None, "report_key": "dev", "report_encoded": [_fake_prep(r) for r in sample_rows],
            "TEENCODE_DICT_MAP": teencode_map, "fetch_teencode_dict": lambda: teencode_map,
            "prepare": _fake_prep, "infer": _fake_infer,
            "f1_score": f1_score, "classification_report": classification_report,
            "confusion_matrix": confusion_matrix, "log_loss": log_loss, "softmax": softmax,
            "write_json": lambda p, o: p.write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8"),
        }
        env.update(_exec_defs(_cell_source(nb, 6), {"split_group"}, env))
        env.update(_exec_defs(_cell_source(nb, 8), {
            "strip_accents", "C1_TEENCODE_MAP", "C1_LOOKALIKE", "C1_SEPS", "_ALPHA_CORE_RE",
            "_split_affix", "_map_pos", "transform_line_with_spans", "_apply_line_transforms",
            "_c1_lookalike_word", "_c1_sep_word", "_c1_elongate_word", "augment_surface",
            "annotations", "overlaps",
        }, env))
        want12 = {
            "_NORM_LOOKALIKE", "_NORM_SEP_RE", "_NORM_ELONG_RE", "_NORM_LEAD_RE", "_NORM_TRAIL_RE",
            "_norm_strip_base", "_normalize_v1_token", "normalize_with_alignment", "infer_normalized",
            "_CLAUSE_END_RE", "_word_line_clauses", "f1_theo_y_item", "_apply_empty_fallback",
            "select_word_marks", "evidence_stats", "macro",
        }
        if label == "dualorc":
            want12.update({"_apply_marginal_postproc", "extract_evidence_runs", "_viterbi_single_np", "_decode_item_crf_tags"})
        env.update(_exec_defs(_cell_source(nb, 12), want12, env))
        env["cal_payload"] = (
            dict(temperature=1.0, o_bias=0.0, chosen_decoder="marginal", base_marginal_threshold=0.5,
                 marginal_threshold=0.5, single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=1.01)
            if label == "dualorc"
            else dict(temperature=1.0, base_span_threshold=0.5, span_threshold=0.5,
                      single_word_threshold=0.5, bridge_ratio=0.0, fallback_min_prob=1.01)
        )
        env.update(_exec_defs(_cell_source(nb, 14), {"arrays", "calibration_stats", "format_output", "evaluate"}, env))
        env["final_pred"] = _fake_infer(None, env["report_encoded"])
        env["real_test_encoded"] = env["real_test_pred"] = None

        # Thực thi phần stress + norm_stress + predict_text của ô 16
        c16 = _cell_source(nb, 16)
        block_stress = c16[:c16.index("if real_test_encoded is not None")]
        exec(compile(block_stress, f"<{label}_c16_stress>", "exec"), env)
        ns_pred = _exec_defs(c16, {"predict_text"}, env)

        norm_st = json.loads((rdir / "dev_norm_stress_tests.json").read_text(encoding="utf-8"))
        assert "clean_c0" in norm_st
        assert set(norm_st["clean_c0"]) >= {"perturbed_raw_f1", "perturbed_norm_f1", "gain_vs_raw_f1", "evidence_f1", "evidence_f1_theo_y", "gain_vs_raw_ev_theo_y"}
        assert "c1_in_dist" in norm_st and "perturbed_raw_f1" in norm_st["c1_in_dist"]

        # predict_text(..., use_normalizer=True) phải trả về cụm bằng chứng verbatim trên văn bản biến dạng gốc!
        obf = "Đm m4y n.g.u quáaaaa"
        out_norm = ns_pred["predict_text"](None, obf, device=torch.device("cpu"), use_normalizer=True)
        assert out_norm["nhan"] == "XUC_PHAM"
        assert out_norm["bang_chung"] and all(ph in obf for ph in out_norm["bang_chung"])
        assert any("n.g.u" in ph or "Đm" in ph for ph in out_norm["bang_chung"])
