"""Unit test cho mô-đun lượng tử hóa embedding và giải mã BIO-CRF thuần NumPy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.v2.bench.dualorc_model import (
    PerRowInt8Embedding,
    TurnAwareBIOCRF,
    _mha_forward_onnx_safe,
    crf_marginals_single_np,
    decode_prediction,
    quantize_embeddings_int8,
    viterbi_single_np,
)


def test_mha_onnx_safe_matches_pytorch_mha():
    torch.manual_seed(42)
    mha = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.1).eval()
    q = torch.randn(2, 8, 64)
    k = torch.randn(2, 19, 64)
    kpm = torch.zeros(2, 19, dtype=torch.bool)
    kpm[0, 12:] = True
    with torch.inference_mode():
        ref = mha(q, k, k, key_padding_mask=kpm, need_weights=False)[0]
        got = _mha_forward_onnx_safe(mha, q, k, k, key_padding_mask=kpm)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)


def test_crf_marginals_numpy_matches_torch():
    torch.manual_seed(42)
    crf = TurnAwareBIOCRF()
    w_len = 7
    em = torch.randn(1, w_len, 3)
    mask = torch.ones(1, w_len, dtype=torch.bool)
    ts = torch.tensor([[True, False, False, True, False, True, False]], dtype=torch.bool)

    pt_marg, _ = crf.marginals(em, mask, ts)
    np_marg = crf_marginals_single_np(
        em[0].numpy(),
        ts[0].tolist(),
        crf.start_trans.detach().numpy(),
        crf.end_trans.detach().numpy(),
        crf.trans.detach().numpy(),
    )
    assert np.allclose(pt_marg[0].detach().numpy(), np_marg, atol=1e-5, rtol=1e-5)
    # Ràng buộc đầu lượt: P(I-EV) tại đầu lượt phải gần bằng 0
    for idx in (0, 3, 5):
        assert np_marg[idx, 2] < 1e-6


def test_viterbi_respects_turn_boundaries():
    crf = TurnAwareBIOCRF()
    em = np.array(
        [
            [-2.0, 3.0, 10.0],
            [-2.0, 0.0, 4.0],
            [-2.0, 0.0, 10.0],  # đầu lượt mới -> cấm I-EV (2)
            [-2.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    seq = viterbi_single_np(
        em,
        [True, False, True, False],
        crf.start_trans.detach().numpy(),
        crf.end_trans.detach().numpy(),
        crf.trans.detach().numpy(),
    )
    assert seq[0] != 2
    assert seq[2] != 2


def test_per_row_int8_embedding_accuracy_and_padding():
    torch.manual_seed(42)
    emb = nn.Embedding(5000, 64, padding_idx=0)
    with torch.no_grad():
        emb.weight[0].zero_()

    container = nn.Sequential(emb)
    replaced = quantize_embeddings_int8(container, min_num_embeddings=4096)
    assert replaced == 1
    q_emb = container[0]
    assert isinstance(q_emb, PerRowInt8Embedding)
    assert q_emb.qweight.dtype == torch.int8

    ids = torch.tensor([[0, 5, 123, 4999]])
    ref = emb(ids)
    approx = q_emb(ids)
    assert torch.allclose(approx[0, 0], torch.zeros(64))
    max_err = float((ref - approx).abs().max().item())
    assert max_err < 0.05


def test_decode_prediction_verbatim_and_safe_gate():
    text = "Chào bạn\nMày là đồ ngu quá"
    # Từ: Chào(0,4), bạn(5,8), Mày(9,12), là(13,15), đồ(16,18), ngu(19,22), quá(23,26)
    words_meta = [
        (0, 4, 0, 1, True, [1]),
        (5, 8, 0, 1, False, [1]),
        (9, 12, 1, 0, True, [1]),
        (13, 15, 1, 0, False, [1]),
        (16, 18, 1, 0, False, [1]),
        (19, 22, 1, 0, False, [1]),
        (23, 26, 1, 0, False, [1]),
    ]
    crf_meta = {
        "labels": ["AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO"],
        "temperature": 1.0,
        "chosen_decoder": "marginal",
        "marginal_threshold": 0.5,
        "single_word_threshold": 0.6,
        "bridge_ratio": 0.0,
        "fallback_min_prob": 0.2,
        "o_bias": 0.0,
        "start": [0.2, 0.0, -5.0],
        "end": [0.0, 0.0, 0.0],
        "trans": [[0.5, 0.0, -5.0], [-0.2, -0.3, 0.6], [0.0, -0.4, 0.6]],
    }
    emissions = np.array(
        [
            [4.0, -2.0, -2.0],
            [4.0, -2.0, -2.0],
            [4.0, -2.0, -2.0],
            [4.0, -2.0, -2.0],
            [-3.0, 4.0, -1.0],
            [-3.0, 0.0, 4.0],
            [4.0, -2.0, -2.0],
        ],
        dtype=np.float32,
    )
    row = {"id": "t1", "text": text, "nhan": "XUC_PHAM"}

    harm_out = decode_prediction(row, words_meta, np.array([-2.0, 4.0, -1.0, -1.0, -1.0]), emissions, 0.5, crf_meta)
    assert harm_out["nhan"] == "XUC_PHAM"
    assert harm_out["bang_chung"] == ["đồ ngu"]
    assert all(p in text for p in harm_out["bang_chung"])

    safe_out = decode_prediction(row, words_meta, np.array([4.0, -2.0, -1.0, -1.0, -1.0]), emissions, 0.05, crf_meta)
    assert safe_out["nhan"] == "AN_TOAN"
    assert safe_out["bang_chung"] == []


def test_compress_fp32_matmul_weights_w8a32_roundtrip(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort
    from src.v2.bench.export_onnx import compress_fp32_matmul_weights_w8a32

    rng = np.random.default_rng(42)
    w = rng.normal(0.0, 0.5, size=(16, 8)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(2, 16)).astype(np.float32)

    w_init = numpy_helper.from_array(w, name="w_fp32")
    mm = helper.make_node("MatMul", ["x", "w_fp32"], ["y"], name="/net/backbone/layers.0/mlp/Wo/MatMul")
    graph = helper.make_graph(
        [mm],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    src = tmp_path / "src.onnx"
    dst = tmp_path / "dst.onnx"
    onnx.save(model, str(src))

    n_comp = compress_fp32_matmul_weights_w8a32(src, dst, ["/net/backbone/layers.0/mlp/Wo/MatMul"])
    assert n_comp == 1

    sess_src = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    sess_dst = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    y_ref = sess_src.run(["y"], {"x": x})[0]
    y_q = sess_dst.run(["y"], {"x": x})[0]
    assert np.max(np.abs(y_ref - y_q)) < 0.05


def test_export_onnx_default_model_name():
    from src.v2.bench.export_onnx import DEFAULT_MODEL_NAME

    assert DEFAULT_MODEL_NAME == "safevichi_v2"

