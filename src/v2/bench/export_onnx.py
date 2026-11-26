"""Xuất mô hình SafeVi-DualORC-CRF (mmBERT-base) sang ONNX FP32 và ONNX INT8.

Quy trình nén INT8 tối ưu cho kiến trúc từ vựng lớn (256k token, ModernBERT):
1. Xuất bản ONNX FP32 gốc (`safevichi_v2_fp32.onnx`, ~1,15 GiB) để đối chiếu chính xác.
2. Thay bảng từ vựng `tok_embeddings` (~750 MiB FP32) bằng `PerRowInt8Embedding`
   (INT8 đối xứng theo hàng + scale FP32, nén 4× còn ~188,5 MiB, dùng toán tử Gather/Cast/Mul chuẩn opset 17).
3. Lượng tử hóa động (`quantize_dynamic`, `QUInt8`) toàn bộ lớp tuyến tính `MatMul`/`Gemm`
   của 22 tầng backbone ModernBERT (~420 MiB → ~106 MiB), đồng thời giữ nguyên đầu DualORC-CRF (~6,5 MiB)
   ở FP32 để bảo toàn độ hiệu chuẩn xác suất và điểm bằng chứng.
4. Xuất `crf_params.json` và `tokenizer/`, rồi đối chiếu sai số trực tiếp với PyTorch FP32
   và dự đoán đã khóa trong `real_test_predictions.jsonl`.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.v2.bench.dualorc_model import (
    DualORCPreprocessor,
    MAX_SUBWORDS_PER_WORD_DEFAULT,
    MAX_WORD_BYTES_DEFAULT,
    SafeViDualORCCRF,
    crf_marginals_single_np,
    decode_prediction,
    load_dualorc_pytorch,
    quantize_embeddings_int8,
)


class _OnnxExportWrapper(nn.Module):
    """Lớp bọc đưa 8 tensor đầu vào phẳng vào SafeViDualORCCRF và trả (logits, emissions, severity)."""

    def __init__(self, net: SafeViDualORCCRF) -> None:
        super().__init__()
        self.net = net

    def forward(
        self,
        ids: torch.Tensor,
        attn_mask: torch.Tensor,
        word_sub_idx: torch.Tensor,
        word_sub_mask: torch.Tensor,
        word_mask: torch.Tensor,
        word_turn: torch.Tensor,
        word_recency: torch.Tensor,
        word_bytes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {
            "ids": ids,
            "attn_mask": attn_mask,
            "word_sub_idx": word_sub_idx,
            "word_sub_mask": word_sub_mask,
            "word_mask": word_mask,
            "word_turn": word_turn,
            "word_recency": word_recency,
            "word_bytes": word_bytes,
        }
        out = self.net(batch, with_aux=False)
        return out["logits"], out["emissions"], out["severity"]


INPUT_NAMES = [
    "ids",
    "attn_mask",
    "word_sub_idx",
    "word_sub_mask",
    "word_mask",
    "word_turn",
    "word_recency",
    "word_bytes",
]
OUTPUT_NAMES = ["logits", "emissions", "severity"]
DEFAULT_MODEL_NAME = "safevichi_v2"
DYNAMIC_AXES = {
    "ids": {0: "B", 1: "L"},
    "attn_mask": {0: "B", 1: "L"},
    "word_sub_idx": {0: "B", 1: "W", 2: "S"},
    "word_sub_mask": {0: "B", 1: "W", 2: "S"},
    "word_mask": {0: "B", 1: "W"},
    "word_turn": {0: "B", 1: "W"},
    "word_recency": {0: "B", 1: "W"},
    "word_bytes": {0: "B", 1: "W"},
    "logits": {0: "B"},
    "emissions": {0: "B", 1: "W"},
    "severity": {0: "B"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_torch_to_onnx(
    net: SafeViDualORCCRF,
    sample_torch: dict[str, torch.Tensor],
    out_path: Path,
    opset: int = 17,
) -> None:
    wrapper = _OnnxExportWrapper(net).eval()
    args = tuple(sample_torch[name] for name in INPUT_NAMES)
    export_kwargs: dict[str, Any] = dict(
        opset_version=opset,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC_AXES,
        do_constant_folding=True,
    )
    # Ưu tiên exporter TorchScript cổ điển (ổn định với opset 17 & giữ tên node backbone rõ ràng)
    try:
        torch.onnx.export(wrapper, args, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapper, args, str(out_path), **export_kwargs)


def _classify_matmul_nodes(onnx_path: Path) -> tuple[list[str], list[str], int]:
    """Phân loại node MatMul/Gemm thành: đầu DualORC (giữ FP32), GeGLU mlp.Wo (W8A32), và backbone thường (W8A8)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    head_nodes: list[str] = []
    mlp_wo_nodes: list[str] = []
    backbone_total = 0
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"}:
            continue
        name_l = node.name.lower()
        if "backbone" not in name_l:
            head_nodes.append(node.name)
        else:
            backbone_total += 1
            if "/mlp/wo/" in name_l:
                mlp_wo_nodes.append(node.name)
    return head_nodes, mlp_wo_nodes, backbone_total


def compress_fp32_matmul_weights_w8a32(src_path: Path, dst_path: Path, target_node_names: list[str]) -> int:
    """Nén trọng số hằng FP32 của các node MatMul ngoại lai (GeGLU mlp.Wo) sang INT8 per-channel + DequantizeLinear.

    Giữ activation ở FP32 (W8A32) để triệt tiêu sai số cắt ngọn ngoại lai của ModernBERT GeGLU
    trong khi vẫn nén trọng số 4× trên đĩa và tương thích 100% ONNX opset 17.
    """
    model = onnx.load(str(src_path))
    init_map = {init.name: init for init in model.graph.initializer}
    target_set = set(target_node_names)
    new_nodes = []
    replaced_inits: set[str] = set()
    new_inits = []
    count = 0
    for node in model.graph.node:
        if node.name in target_set and node.op_type == "MatMul" and len(node.input) == 2 and node.input[1] in init_map:
            w_name = node.input[1]
            w = numpy_helper.to_array(init_map[w_name]).astype(np.float32)
            scale = np.maximum(np.max(np.abs(w), axis=0), 1e-8).astype(np.float32) / 127.0
            qw = np.clip(np.round(w / scale[None, :]), -127, 127).astype(np.int8)
            zp = np.zeros_like(scale, dtype=np.int8)
            qw_name = f"{w_name}_w8"
            sc_name = f"{w_name}_w8_scale"
            zp_name = f"{w_name}_w8_zp"
            dq_out = f"{w_name}_w8_deq"
            new_inits.extend(
                [
                    numpy_helper.from_array(qw, name=qw_name),
                    numpy_helper.from_array(scale, name=sc_name),
                    numpy_helper.from_array(zp, name=zp_name),
                ]
            )
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[qw_name, sc_name, zp_name],
                outputs=[dq_out],
                name=f"{node.name}_deq",
                axis=1,
            )
            node.input[1] = dq_out
            new_nodes.extend([dq_node, node])
            replaced_inits.add(w_name)
            count += 1
        else:
            new_nodes.append(node)
    kept_inits = [i for i in model.graph.initializer if i.name not in replaced_inits]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits + new_inits)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save(model, str(dst_path))
    return count


def build_crf_params(
    net: SafeViDualORCCRF,
    cal_cfg: dict[str, Any],
    cfg: dict[str, Any],
    prep: DualORCPreprocessor,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "arch": cfg["arch"],
        "schema": cfg["schema"],
        "backbone": cfg["backbone"],
        "backbone_rev": cfg.get("backbone_rev"),
        "labels": list(cfg["labels"]),
        "max_tokens": int(cfg["max_tokens"]),
        "max_word_bytes": MAX_WORD_BYTES_DEFAULT,
        "max_subwords_per_word": MAX_SUBWORDS_PER_WORD_DEFAULT,
        "bos_id": int(prep.bos_id),
        "sep_id": int(prep.sep_id),
        "pad_id": int(prep.pad_id),
        "chosen_decoder": cal_cfg["chosen_decoder"],
        "temperature": float(cal_cfg["temperature"]),
        "o_bias": float(cal_cfg["o_bias"]),
        "marginal_threshold": float(cal_cfg["marginal_threshold"]),
        "single_word_threshold": float(cal_cfg["single_word_threshold"]),
        "bridge_ratio": float(cal_cfg["bridge_ratio"]),
        "fallback_min_prob": float(cal_cfg["fallback_min_prob"]),
    }
    if net.use_bio_crf:
        meta["start"] = net.crf.start_trans.detach().cpu().numpy().tolist()
        meta["end"] = net.crf.end_trans.detach().cpu().numpy().tolist()
        meta["trans"] = net.crf.trans.detach().cpu().numpy().tolist()
    return meta


def verify_parity(
    net_fp32: SafeViDualORCCRF,
    prep: DualORCPreprocessor,
    crf_meta: dict[str, Any],
    fp32_onnx: Path,
    int8_onnx: Path,
    test_records: list[dict[str, Any]],
    ref_preds_by_id: dict[str, dict[str, Any]],
    n_probe: int = 16,
) -> dict[str, Any]:
    """Kiểm chứng sai số số học giữa PyTorch FP32, ONNX FP32, ONNX INT8 và dự đoán đã khóa."""
    # Chọn các mẫu đa dạng độ dài (kể cả mẫu dài > 128 token để kiểm tra sliding-window attention của ModernBERT)
    prepared_all = [prep.prepare(r) for r in test_records]
    short_samples = [e for e in prepared_all if e["n_tokens"] < 64][: max(1, n_probe // 3)]
    mid_samples = [e for e in prepared_all if 64 <= e["n_tokens"] <= 128][: max(1, n_probe // 3)]
    long_samples = [e for e in prepared_all if e["n_tokens"] > 128][: max(1, n_probe - len(short_samples) - len(mid_samples))]
    probe = short_samples + mid_samples + long_samples
    if len(probe) < n_probe:
        seen = {e["row"]["id"] for e in probe}
        for e in prepared_all:
            if e["row"]["id"] not in seen:
                probe.append(e)
                seen.add(e["row"]["id"])
            if len(probe) >= n_probe:
                break

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(str(fp32_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_onnx), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    # Kiểm chứng thêm tính chính xác của crf_marginals_single_np so với PyTorch TurnAwareBIOCRF.marginals
    max_crf_marg_diff = 0.0
    max_fp32_logit_diff = 0.0
    max_fp32_emit_diff = 0.0
    max_fp32_sev_diff = 0.0
    max_int8_logit_diff = 0.0
    max_int8_emit_diff = 0.0

    pt_vs_ref_label_ok = 0
    pt_vs_ref_ev_ok = 0
    fp32_vs_pt_label_ok = 0
    int8_vs_pt_label_ok = 0
    token_lengths = [e["n_tokens"] for e in probe]

    with torch.inference_mode():
        for e in probe:
            tb = prep.collate_torch([e])
            nb = prep.collate_numpy([e])
            nw = e["n_words"]

            pt_out = net_fp32(tb, with_aux=False)
            pt_logits = pt_out["logits"][0].numpy()
            pt_emit = pt_out["emissions"][0, :nw].numpy()
            pt_sev = float(pt_out["severity"][0].item())

            # Kiểm chứng CRF marginals NumPy vs PyTorch
            ts_t = torch.tensor([[bool(m[4]) for m in e["words_meta"]]], dtype=torch.bool)
            pt_marg, _ = net_fp32.crf.marginals(pt_out["emissions"], tb["word_mask"], ts_t)
            np_marg = crf_marginals_single_np(
                pt_emit,
                [bool(m[4]) for m in e["words_meta"]],
                np.asarray(crf_meta["start"]),
                np.asarray(crf_meta["end"]),
                np.asarray(crf_meta["trans"]),
            )
            max_crf_marg_diff = max(max_crf_marg_diff, float(np.max(np.abs(pt_marg[0, :nw].numpy() - np_marg))))

            o32_logits, o32_emit, o32_sev = sess_fp32.run(OUTPUT_NAMES, nb)
            o8_logits, o8_emit, o8_sev = sess_int8.run(OUTPUT_NAMES, nb)

            max_fp32_logit_diff = max(max_fp32_logit_diff, float(np.max(np.abs(pt_logits - o32_logits[0]))))
            max_fp32_emit_diff = max(max_fp32_emit_diff, float(np.max(np.abs(pt_emit - o32_emit[0, :nw]))))
            max_fp32_sev_diff = max(max_fp32_sev_diff, abs(pt_sev - float(o32_sev[0])))

            max_int8_logit_diff = max(max_int8_logit_diff, float(np.max(np.abs(pt_logits - o8_logits[0]))))
            max_int8_emit_diff = max(max_int8_emit_diff, float(np.max(np.abs(pt_emit - o8_emit[0, :nw]))))

            dec_pt = decode_prediction(e["row"], e["words_meta"], pt_logits, pt_emit, pt_sev, crf_meta)
            dec_o32 = decode_prediction(e["row"], e["words_meta"], o32_logits[0], o32_emit[0, :nw], float(o32_sev[0]), crf_meta)
            dec_o8 = decode_prediction(e["row"], e["words_meta"], o8_logits[0], o8_emit[0, :nw], float(o8_sev[0]), crf_meta)

            ref = ref_preds_by_id.get(e["row"]["id"])
            if ref is not None:
                pt_vs_ref_label_ok += int(dec_pt["nhan"] == ref["nhan"])
                pt_vs_ref_ev_ok += int(dec_pt["bang_chung"] == ref["bang_chung"])
            fp32_vs_pt_label_ok += int(dec_o32["nhan"] == dec_pt["nhan"])
            int8_vs_pt_label_ok += int(dec_o8["nhan"] == dec_pt["nhan"])

        # Kiểm chứng thêm batch động B=4 trên ONNX FP32 và ONNX INT8
        batch4 = probe[:4]
        nb4 = prep.collate_numpy(batch4)
        tb4 = prep.collate_torch(batch4)
        pt4 = net_fp32(tb4, with_aux=False)["logits"].numpy()
        o32_4 = sess_fp32.run(OUTPUT_NAMES, nb4)[0]
        o8_4 = sess_int8.run(OUTPUT_NAMES, nb4)[0]
        batch4_fp32_diff = float(np.max(np.abs(pt4 - o32_4)))
        batch4_int8_diff = float(np.max(np.abs(pt4 - o8_4)))

    return {
        "n_probe": len(probe),
        "token_len_min_max": [min(token_lengths), max(token_lengths)],
        "crf_numpy_vs_torch_max_abs_diff": round(max_crf_marg_diff, 8),
        "onnx_fp32_vs_torch": {
            "batch1_max_abs_diff_logits": round(max_fp32_logit_diff, 6),
            "batch1_max_abs_diff_emissions": round(max_fp32_emit_diff, 6),
            "batch1_max_abs_diff_severity": round(max_fp32_sev_diff, 6),
            "batch4_max_abs_diff_logits": round(batch4_fp32_diff, 6),
            "label_agreement": f"{fp32_vs_pt_label_ok}/{len(probe)}",
        },
        "onnx_int8_vs_torch": {
            "batch1_max_abs_diff_logits": round(max_int8_logit_diff, 6),
            "batch1_max_abs_diff_emissions": round(max_int8_emit_diff, 6),
            "batch4_max_abs_diff_logits": round(batch4_int8_diff, 6),
            "label_agreement": f"{int8_vs_pt_label_ok}/{len(probe)}",
        },
        "torch_local_vs_locked_colab_ref": {
            "label_agreement": f"{pt_vs_ref_label_ok}/{len(probe)}",
            "evidence_exact_agreement": f"{pt_vs_ref_ev_ok}/{len(probe)}",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/dualorc_mmbert_base/model_best.safetensors"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/config.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/calibration.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/dualorc_mmbert_base/onnx"),
    )
    parser.add_argument(
        "--probe-gold",
        type=Path,
        default=Path("data/v2/test_day_du.jsonl"),
    )
    parser.add_argument(
        "--probe-ref-pred",
        type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/test/real_test_predictions.jsonl"),
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--keep-head-fp32", action="store_true", default=True)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    t0 = time.perf_counter()
    out_dir = args.out_dir
    tok_dir = out_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1/5. Nạp kiến trúc SafeVi-DualORC-CRF & trọng số model_best.safetensors...")
    net, tokenizer, cfg = load_dualorc_pytorch(args.weights, args.config, tokenizer_cache_dir=tok_dir)
    cal_cfg = json.loads(args.calibration.read_text(encoding="utf-8"))
    prep = DualORCPreprocessor(tokenizer, max_tokens=int(cfg["max_tokens"]), labels=tuple(cfg["labels"]))
    crf_meta = build_crf_params(net, cal_cfg, cfg, prep)
    (out_dir / "crf_params.json").write_text(json.dumps(crf_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Tạo batch mẫu B=2 (1 câu ngắn + 1 hội thoại dài >140 token có padding) để trace đúng shape động & sliding-window
    test_records = [json.loads(line) for line in args.probe_gold.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    long_dummy = (
        "Chào anh, hôm nay anh rảnh không mình trao đổi chút việc nhóm.\n"
        + " ".join(["Cảm ơn bạn rất nhiều vì đã hỗ trợ gửi tài liệu học tập đúng hạn trong tuần này."] * 10)
        + "\nOk để tối mình xem kỹ lại rồi nhắn tin phản hồi đầy đủ cho cả nhóm nhé."
    )
    dummy_rows = [
        {"id": "dummy_short", "text": "Cảm ơn bạn nhiều nhé"},
        {"id": "dummy_long", "text": long_dummy},
    ]
    dummy_ex = [prep.prepare(r) for r in dummy_rows]
    dummy_torch = prep.collate_torch(dummy_ex)
    print(f"    Dummy trace shapes: ids={tuple(dummy_torch['ids'].shape)}, words={tuple(dummy_torch['word_mask'].shape)}")

    fp32_onnx = out_dir / f"{args.model_name}_fp32.onnx"
    print(f"2/5. Xuất đồ thị ONNX FP32 (opset {args.opset}) -> {fp32_onnx}...")
    _export_torch_to_onnx(net, dummy_torch, fp32_onnx, opset=args.opset)
    fp32_mib = fp32_onnx.stat().st_size / (1024 ** 2)
    print(f"    Xong ONNX FP32: {fp32_mib:.2f} MiB")

    print("3/5. Lượng tử hóa bảng từ vựng PerRowInt8Embedding (nén 4×) & xuất đồ thị trung gian...")
    net_emb_q = copy.deepcopy(net).eval()
    emb_replaced = quantize_embeddings_int8(net_emb_q.backbone, min_num_embeddings=4096)
    tmp_emb_onnx = out_dir / "_tmp_emb_int8.onnx"
    _export_torch_to_onnx(net_emb_q, dummy_torch, tmp_emb_onnx, opset=args.opset)
    del net_emb_q
    gc.collect()
    emb_tmp_mib = tmp_emb_onnx.stat().st_size / (1024 ** 2)
    print(f"    Đã nén {emb_replaced} bảng embedding -> đồ thị trung gian {emb_tmp_mib:.2f} MiB")

    int8_onnx = out_dir / f"{args.model_name}_int8.onnx"
    tmp_w8a8_onnx = out_dir / "_tmp_w8a8.onnx"
    print(f"4/5. Lượng tử hóa lai ONNX INT8 (QInt8 per-channel W8A8 + W8A32 cho GeGLU mlp.Wo) -> {int8_onnx}...")
    head_nodes, mlp_wo_nodes, n_bb_mm = _classify_matmul_nodes(tmp_emb_onnx)
    exclude_from_w8a8 = (head_nodes if args.keep_head_fp32 else []) + mlp_wo_nodes
    print(
        f"    Backbone có {n_bb_mm} node MatMul/Gemm: W8A8 QInt8 per-channel cho attn/mlp.Wi, "
        f"W8A32 DequantizeLinear cho {len(mlp_wo_nodes)} tầng GeGLU mlp.Wo, giữ {len(head_nodes)} node đầu DualORC ở FP32."
    )
    quantize_dynamic(
        model_input=str(tmp_emb_onnx),
        model_output=str(tmp_w8a8_onnx),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_exclude=exclude_from_w8a8,
    )
    n_wo_w8a32 = compress_fp32_matmul_weights_w8a32(tmp_w8a8_onnx, int8_onnx, mlp_wo_nodes)
    tmp_emb_onnx.unlink(missing_ok=True)
    tmp_w8a8_onnx.unlink(missing_ok=True)
    int8_mib = int8_onnx.stat().st_size / (1024 ** 2)
    print(f"    Xong ONNX INT8: {int8_mib:.2f} MiB (W8A32 mlp.Wo={n_wo_w8a32}, tỉ lệ nén {fp32_mib / int8_mib:.2f}× so với FP32)")

    print("5/5. Kiểm chứng sai số số học (PyTorch FP32 vs ONNX FP32 vs ONNX INT8 vs Colab locked)...")
    ref_preds_by_id: dict[str, dict[str, Any]] = {}
    if args.probe_ref_pred.exists():
        for line in args.probe_ref_pred.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rec = json.loads(line)
                ref_preds_by_id[str(rec["id"])] = rec

    parity = verify_parity(net, prep, crf_meta, fp32_onnx, int8_onnx, test_records, ref_preds_by_id, n_probe=16)
    elapsed = time.perf_counter() - t0

    manifest = {
        "model_name": args.model_name,
        "files": {
            "onnx_fp32": fp32_onnx.name,
            "onnx_int8": int8_onnx.name,
            "crf_params": "crf_params.json",
        },
        "opset": args.opset,
        "quantization_method": (
            "Hybrid_INT8: PerRowInt8Embedding(int8_sym_fp32_scale) "
            "+ ORT_quantize_dynamic(QInt8_per_channel_W8A8, attn+mlp.Wi) "
            "+ DequantizeLinear(int8_per_channel_W8A32, 22_GeGLU_mlp.Wo) "
            "+ DualORC_head_FP32"
        ),
        "embeddings_quantized": emb_replaced,
        "backbone_matmul_nodes": n_bb_mm,
        "mlp_wo_w8a32_nodes": n_wo_w8a32,
        "head_matmul_nodes_kept_fp32": len(head_nodes) if args.keep_head_fp32 else 0,
        "sizes_MiB": {
            "safetensors_fp32": round(args.weights.stat().st_size / (1024 ** 2), 2),
            "onnx_fp32": round(fp32_mib, 2),
            "onnx_int8": round(int8_mib, 2),
            "tokenizer_json": round((tok_dir / "tokenizer.json").stat().st_size / (1024 ** 2), 2),
            "compression_ratio_vs_safetensors": round((args.weights.stat().st_size / (1024 ** 2)) / int8_mib, 2),
            "compression_ratio_vs_onnx_fp32": round(fp32_mib / int8_mib, 2),
        },
        "sha256": {
            "onnx_fp32": sha256_file(fp32_onnx),
            "onnx_int8": sha256_file(int8_onnx),
            "crf_params": sha256_file(out_dir / "crf_params.json"),
        },
        "parity": parity,
        "export_seconds": round(elapsed, 2),
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
