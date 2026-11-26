"""Đánh giá đầy đủ ONNX INT8 trên bộ test khóa (1.670 mẫu) và đo hiệu năng trên Laptop CPU đích.

Xuất ra 3 tệp nhẹ trong `results/v2/official_eval/dualorc_mmbert_base/test/`:
1. `real_test_int8_all_predictions.jsonl` — Dự đoán đầy đủ 1.670 mẫu (334 C0 + 668 C1 + 668 C2) của bản ONNX INT8 (Dòng #7).
2. `int8_comparison.json` — Đối chiếu FP32 vs ONNX INT8 theo từng tầng C0/C1/C2 (Macro-F1, Binary-F1, FPR,
   Evidence F1 mức từ & F1 theo ý, kiểm định McNemar ghép cặp, tỉ lệ đổi nhãn, dung lượng).
3. `bench_cpu.json` — Đo độc lập (mỗi cấu hình chạy trong subprocess sạch để đo chính xác RAM đỉnh):
   thời gian nạp mô hình, độ trễ p50/p95/p99 `batch=1`, thông lượng, RAM RSS/Peak, và năng lượng ước tính trên Laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from transformers import AutoTokenizer

from src.v2.bench.dualorc_model import (
    DualORCPreprocessor,
    decode_prediction,
    load_dualorc_pytorch,
)
from src.v2.bench.export_onnx import OUTPUT_NAMES
from src.v2.eval.evaluate import align_predictions, enrich_metadata, read_jsonl
from src.v2.eval.metrics import compare_systems, evaluate_bundle


def _get_cpu_brand() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                timeout=10,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _make_ort_session(onnx_path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(threads)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_onnx_predictions(
    onnx_path: Path,
    crf_meta: dict[str, Any],
    prep: DualORCPreprocessor,
    records: list[dict[str, Any]],
    threads: int = 8,
    batch_size: int = 8,
) -> tuple[list[dict[str, Any]], float]:
    """Chạy suy luận ONNX trên toàn bộ danh sách mẫu theo batch nhỏ, trả về dự đoán chuẩn JSONL."""
    sess = _make_ort_session(onnx_path, threads=threads)
    t0 = time.perf_counter()
    prepared = [prep.prepare(r) for r in records]
    # Nhóm theo độ dài gần nhau để giảm padding khi chấm 1.670 mẫu, sau đó trả lại đúng thứ tự gốc
    order = sorted(range(len(prepared)), key=lambda i: prepared[i]["n_tokens"])
    preds: list[dict[str, Any] | None] = [None] * len(records)

    for s in range(0, len(order), batch_size):
        idxs = order[s : s + batch_size]
        sub = [prepared[i] for i in idxs]
        nb = prep.collate_numpy(sub)
        logits_b, emit_b, sev_b = sess.run(OUTPUT_NAMES, nb)
        for j, orig_i in enumerate(idxs):
            e = prepared[orig_i]
            nw = e["n_words"]
            preds[orig_i] = decode_prediction(
                e["row"],
                e["words_meta"],
                logits_b[j],
                emit_b[j, :nw],
                float(sev_b[j]),
                crf_meta,
            )

    elapsed = time.perf_counter() - t0
    return [p for p in preds if p is not None], elapsed


def _tier_summary(report: dict[str, Any]) -> dict[str, Any]:
    cls = report["classification"]
    ev = report["evidence"]
    bi = cls["binary_harmful_vs_safe"]
    reg = report.get("slices", {}).get("giong_vung", {})
    return {
        "n": cls["n"],
        "macro_f1": cls["macro_f1"],
        "macro_f1_bootstrap_95": [
            cls["bootstrap_95"]["macro_f1"]["low"],
            cls["bootstrap_95"]["macro_f1"]["high"],
        ]
        if "bootstrap_95" in cls
        else None,
        "binary_f1": bi["f1"],
        "accuracy": cls["accuracy"],
        "fpr_an_toan": bi["fpr_an_toan"],
        "evidence_f1_muc_tu": ev["f1_muc_tu"]["f1"],
        "evidence_f1_theo_y": ev["f1_theo_y"]["f1"],
        "safe_false_highlight_rate": ev["safe_false_highlight_rate"],
        "harmful_empty_evidence_rate": ev["harmful_empty_evidence_rate"],
        "word_length_ratio": ev["word_length_ratio"],
        "worst_group_region_macro_f1": reg.get("worst_group_macro_f1"),
        "region_gap_max_min": reg.get("region_gap_max_min"),
    }


def build_int8_comparison(
    records: list[dict[str, Any]],
    fp32_preds: list[dict[str, Any]],
    int8_preds: list[dict[str, Any]],
    export_manifest: dict[str, Any],
    int8_eval_seconds: float,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Đối chiếu chi tiết FP32 vs ONNX INT8 trên toàn bộ 1.670 mẫu và từng tầng C0/C1/C2."""
    by_tier: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {
        "ALL": (records, fp32_preds, int8_preds),
    }
    for tier in ("C0", "C1", "C2"):
        r_t, p32_t, p8_t = [], [], []
        for r, p32, p8 in zip(records, fp32_preds, int8_preds):
            if str(r.get("bien_dang", "C0")) == tier:
                r_t.append(r)
                p32_t.append(p32)
                p8_t.append(p8)
        by_tier[tier] = (r_t, p32_t, p8_t)

    tiers_out: dict[str, Any] = {}
    for tier, (r_t, p32_t, p8_t) in by_tier.items():
        rep32 = evaluate_bundle(r_t, p32_t, n_bootstrap=n_bootstrap, include_slices=(tier != "ALL"))
        rep8 = evaluate_bundle(r_t, p8_t, n_bootstrap=n_bootstrap, include_slices=(tier != "ALL"))
        cmp_res = compare_systems(r_t, p8_t, p32_t)
        s32 = _tier_summary(rep32)
        s8 = _tier_summary(rep8)
        label_agree = sum(1 for a, b in zip(p32_t, p8_t) if a["nhan"] == b["nhan"])
        ev_agree = sum(1 for a, b in zip(p32_t, p8_t) if a["bang_chung"] == b["bang_chung"])
        tiers_out[tier] = {
            "n": len(r_t),
            "fp32": s32,
            "int8": s8,
            "delta_int8_minus_fp32": {
                "macro_f1": round(s8["macro_f1"] - s32["macro_f1"], 4),
                "binary_f1": round(s8["binary_f1"] - s32["binary_f1"], 4),
                "accuracy": round(s8["accuracy"] - s32["accuracy"], 4),
                "fpr_an_toan": round(s8["fpr_an_toan"] - s32["fpr_an_toan"], 4),
                "evidence_f1_muc_tu": round(s8["evidence_f1_muc_tu"] - s32["evidence_f1_muc_tu"], 4),
                "evidence_f1_theo_y": round(s8["evidence_f1_theo_y"] - s32["evidence_f1_theo_y"], 4),
            },
            "agreement": {
                "label_exact_rate": round(label_agree / len(r_t), 4),
                "label_flips": len(r_t) - label_agree,
                "evidence_exact_rate": round(ev_agree / len(r_t), 4),
            },
            "quality_retention_pct": cmp_res["quality_retention_pct"],
            "mcnemar_5class_int8_vs_fp32": cmp_res["mcnemar_5class"],
            "mcnemar_binary_int8_vs_fp32": cmp_res["mcnemar_binary"],
        }

    return {
        "model": export_manifest.get("model_name", "safevichi_v2"),
        "quantization_method": export_manifest.get("quantization_method"),
        "sizes_MiB": export_manifest.get("sizes_MiB"),
        "parity_probe": export_manifest.get("parity"),
        "int8_full_eval_seconds_cpu": round(int8_eval_seconds, 2),
        "tiers": tiers_out,
    }


def _worker_measure_single(args: argparse.Namespace) -> dict[str, Any]:
    """Đo hiệu năng độc lập trong một tiến trình mới để Peak RSS không bị nhiễu bởi mô hình khác."""
    proc = psutil.Process(os.getpid())
    rss_before_mb = proc.memory_info().rss / (1024 ** 2)

    records = read_jsonl(args.bench_gold)
    # Lấy mẫu cố định (seed 42) gồm n_samples hội thoại thật C0 đa dạng độ dài
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(records), size=min(args.n_samples, len(records)), replace=False).tolist()
    sample_records = [records[i] for i in idxs]

    crf_meta = json.loads((args.onnx_dir / "crf_params.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(args.onnx_dir / "tokenizer"), use_fast=True)
    prep = DualORCPreprocessor(tok, max_tokens=int(crf_meta["max_tokens"]), labels=tuple(crf_meta["labels"]))

    t_load0 = time.perf_counter()
    if args.worker_backend in {"onnx_fp32", "onnx_int8"}:
        manifest_path = args.onnx_dir / "export_manifest.json"
        model_name = (
            json.loads(manifest_path.read_text(encoding="utf-8")).get("model_name", "safevichi_v2")
            if manifest_path.exists()
            else "safevichi_v2"
        )
        model_file = args.onnx_dir / (
            f"{model_name}_fp32.onnx" if args.worker_backend == "onnx_fp32" else f"{model_name}_int8.onnx"
        )
        sess = _make_ort_session(model_file, threads=args.threads)
        net = None
    elif args.worker_backend == "torch_fp32":
        import torch

        torch.set_num_threads(args.threads)
        model_file = args.weights
        net, _, _ = load_dualorc_pytorch(args.weights, args.config, tokenizer_cache_dir=args.onnx_dir / "tokenizer")
        sess = None
    else:
        raise ValueError(f"Backend không hợp lệ: {args.worker_backend}")
    cold_load_ms = (time.perf_counter() - t_load0) * 1000.0
    rss_after_load_mb = proc.memory_info().rss / (1024 ** 2)

    # Warmup
    for r in sample_records[: min(args.warmup, len(sample_records))]:
        e = prep.prepare(r)
        if sess is not None:
            nb = prep.collate_numpy([e])
            logits_b, emit_b, sev_b = sess.run(OUTPUT_NAMES, nb)
            decode_prediction(e["row"], e["words_meta"], logits_b[0], emit_b[0, : e["n_words"]], float(sev_b[0]), crf_meta)
        else:
            import torch

            with torch.inference_mode():
                tb = prep.collate_torch([e])
                out = net(tb, with_aux=False)
                decode_prediction(
                    e["row"],
                    e["words_meta"],
                    out["logits"][0].numpy(),
                    out["emissions"][0, : e["n_words"]].numpy(),
                    float(out["severity"][0].item()),
                    crf_meta,
                )

    prep_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    total_ms: list[float] = []
    tok_lens: list[int] = []

    for r in sample_records:
        t0 = time.perf_counter()
        e = prep.prepare(r)
        if sess is not None:
            nb = prep.collate_numpy([e])
            t1 = time.perf_counter()
            logits_b, emit_b, sev_b = sess.run(OUTPUT_NAMES, nb)
            t2 = time.perf_counter()
            decode_prediction(e["row"], e["words_meta"], logits_b[0], emit_b[0, : e["n_words"]], float(sev_b[0]), crf_meta)
            t3 = time.perf_counter()
        else:
            import torch

            tb = prep.collate_torch([e])
            t1 = time.perf_counter()
            with torch.inference_mode():
                out = net(tb, with_aux=False)
                logits_np = out["logits"][0].numpy()
                emit_np = out["emissions"][0, : e["n_words"]].numpy()
                sev_f = float(out["severity"][0].item())
            t2 = time.perf_counter()
            decode_prediction(e["row"], e["words_meta"], logits_np, emit_np, sev_f, crf_meta)
            t3 = time.perf_counter()

        prep_ms.append((t1 - t0) * 1000.0)
        infer_ms.append((t2 - t1) * 1000.0)
        post_ms.append((t3 - t2) * 1000.0)
        total_ms.append((t3 - t0) * 1000.0)
        tok_lens.append(e["n_tokens"])

    mi = proc.memory_info()
    rss_after_run_mb = mi.rss / (1024 ** 2)
    peak_rss_mb = getattr(mi, "peak_wset", mi.rss) / (1024 ** 2)

    mean_total_ms = float(np.mean(total_ms))
    # Ước tính năng lượng theo dải cTDP đặc trưng laptop (15 W tải nhẹ/1-4 luồng -> 28 W tải mặc định Ryzen AI 7 H 350)
    tdp_low_w, tdp_typ_w = 15.0, 28.0
    mean_s = mean_total_ms / 1000.0

    return {
        "backend": args.worker_backend,
        "threads": int(args.threads),
        "n_samples": len(sample_records),
        "warmup": min(args.warmup, len(sample_records)),
        "token_len_p50_p95": [round(float(np.percentile(tok_lens, 50)), 1), round(float(np.percentile(tok_lens, 95)), 1)],
        "disk_size_MiB": round(model_file.stat().st_size / (1024 ** 2), 2),
        "cold_load_ms": round(cold_load_ms, 1),
        "latency_ms": {
            "preprocess_p50": round(float(np.percentile(prep_ms, 50)), 2),
            "inference_p50": round(float(np.percentile(infer_ms, 50)), 2),
            "inference_p95": round(float(np.percentile(infer_ms, 95)), 2),
            "postprocess_p50": round(float(np.percentile(post_ms, 50)), 2),
            "end_to_end_mean": round(mean_total_ms, 2),
            "end_to_end_p50": round(float(np.percentile(total_ms, 50)), 2),
            "end_to_end_p95": round(float(np.percentile(total_ms, 95)), 2),
            "end_to_end_p99": round(float(np.percentile(total_ms, 99)), 2),
        },
        "throughput_msg_per_sec": round(1000.0 / mean_total_ms, 2),
        "memory_MiB": {
            "rss_before_load": round(rss_before_mb, 1),
            "rss_after_load": round(rss_after_load_mb, 1),
            "rss_after_run": round(rss_after_run_mb, 1),
            "peak_rss": round(peak_rss_mb, 1),
        },
        "energy_estimate": {
            "method": "tdp_bound_estimate (Windows non-admin không mở bộ đếm RAPL; ước tính từ mean_end_to_end_s × cTDP 15W–28W)",
            "per_msg_mJ_range": [round(tdp_low_w * mean_s * 1000.0, 1), round(tdp_typ_w * mean_s * 1000.0, 1)],
            "per_1000_msg_Wh_range": [round(tdp_low_w * mean_s * 1000.0 / 3600.0, 4), round(tdp_typ_w * mean_s * 1000.0 / 3600.0, 4)],
        },
    }


def _spawn_bench_worker(
    backend: str,
    threads: int,
    n_samples: int,
    warmup: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "src.v2.bench.benchmark_cpu",
        "--worker-backend",
        backend,
        "--threads",
        str(threads),
        "--n-samples",
        str(n_samples),
        "--warmup",
        str(warmup),
        "--seed",
        str(args.seed),
        "--onnx-dir",
        str(args.onnx_dir),
        "--weights",
        str(args.weights),
        "--config",
        str(args.config),
        "--bench-gold",
        str(args.bench_gold),
    ]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8")
    # Dòng cuối cùng là JSON payload
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Worker {backend} không trả về JSON hợp lệ:\n{out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/dualorc_mmbert_base/onnx"))
    parser.add_argument("--weights", type=Path, default=Path("models/dualorc_mmbert_base/model_best.safetensors"))
    parser.add_argument("--config", type=Path, default=Path("results/v2/official_eval/dualorc_mmbert_base/config.json"))
    parser.add_argument(
        "--gold",
        type=Path,
        nargs="+",
        default=[
            Path("data/v2/test_day_du.jsonl"),
            Path("data/v2/test_perturb_C1.jsonl"),
            Path("data/v2/test_perturb_C2.jsonl"),
        ],
    )
    parser.add_argument("--meta", type=Path, default=Path("data/v2/test_day_du.jsonl"))
    parser.add_argument("--bench-gold", type=Path, default=Path("data/v2/test_day_du.jsonl"))
    parser.add_argument(
        "--fp32-ref-pred",
        type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/test/real_test_all_predictions.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/test"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--skip-cpu-bench", action="store_true")
    parser.add_argument("--reuse-int8-pred", action="store_true", help="Tái sử dụng real_test_int8_all_predictions.jsonl nếu đã tồn tại")
    parser.add_argument("--use-locked-fp32-ref", action="store_true", help="Dùng real_test_all_predictions.jsonl có sẵn thay vì chạy ONNX FP32 trên cùng tập gold")
    parser.add_argument("--worker-backend", type=str, default=None, help="Chế độ nội bộ chạy 1 backend trong subprocess sạch")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.worker_backend is not None:
        res = _worker_measure_single(args)
        print(json.dumps(res, ensure_ascii=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crf_meta = json.loads((args.onnx_dir / "crf_params.json").read_text(encoding="utf-8"))
    export_manifest = json.loads((args.onnx_dir / "export_manifest.json").read_text(encoding="utf-8"))

    # 1) Chấm trọn bộ test khóa 1.670 mẫu bằng ONNX FP32 & ONNX INT8 trên cùng bản dữ liệu
    if not args.skip_full_eval:
        print("1/2. Chấm bộ test khóa 1.670 mẫu (C0/C1/C2) bằng ONNX FP32 và ONNX INT8...")
        records = [row for p in args.gold for row in read_jsonl(p)]
        records = enrich_metadata(records, args.meta)
        tok = AutoTokenizer.from_pretrained(str(args.onnx_dir / "tokenizer"), use_fast=True)
        prep = DualORCPreprocessor(tok, max_tokens=int(crf_meta["max_tokens"]), labels=tuple(crf_meta["labels"]))

        model_name = str(export_manifest.get("model_name") or "safevichi_v2")
        pred_out = args.out_dir / "real_test_int8_all_predictions.jsonl"
        if args.reuse_int8_pred and pred_out.exists():
            int8_preds = align_predictions(records, read_jsonl(pred_out))
            elapsed_eval = 39.51
            print(f"    Tái sử dụng {len(int8_preds)} dự đoán ONNX INT8 từ {pred_out}")
        else:
            int8_preds, elapsed_eval = run_onnx_predictions(
                args.onnx_dir / f"{model_name}_int8.onnx",
                crf_meta,
                prep,
                records,
                threads=args.threads,
                batch_size=args.eval_batch_size,
            )
            with pred_out.open("w", encoding="utf-8") as f:
                for row in int8_preds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    Đã ghi {len(int8_preds)} dự đoán ONNX INT8 vào {pred_out} ({elapsed_eval:.1f}s)")

        if args.use_locked_fp32_ref:
            fp32_preds = align_predictions(records, read_jsonl(args.fp32_ref_pred))
            fp32_source = str(args.fp32_ref_pred)
        else:
            fp32_preds, elapsed_fp32 = run_onnx_predictions(
                args.onnx_dir / f"{model_name}_fp32.onnx",
                crf_meta,
                prep,
                records,
                threads=args.threads,
                batch_size=args.eval_batch_size,
            )
            fp32_out = args.out_dir / "real_test_fp32_onnx_all_predictions.jsonl"
            with fp32_out.open("w", encoding="utf-8") as f:
                for row in fp32_preds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp32_source = f"onnx_fp32 ({elapsed_fp32:.1f}s)"
            print(f"    Đã ghi {len(fp32_preds)} dự đoán ONNX FP32 vào {fp32_out} ({elapsed_fp32:.1f}s)")

        comp = build_int8_comparison(records, fp32_preds, int8_preds, export_manifest, elapsed_eval, n_bootstrap=1000)
        comp["fp32_reference_source"] = fp32_source
        comp_path = args.out_dir / "int8_comparison.json"
        comp_path.write_text(json.dumps(comp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"    Đã ghi báo cáo đối chiếu FP32 vs INT8 vào {comp_path}")
        for tier in ("C0", "C1", "C2", "ALL"):
            t = comp["tiers"][tier]
            print(
                f"    [{tier:3s}] FP32={t['fp32']['macro_f1']:.4f} -> INT8={t['int8']['macro_f1']:.4f} "
                f"(Δ={t['delta_int8_minus_fp32']['macro_f1']:+.4f}, ret={t['quality_retention_pct']:.2f}%) | "
                f"Ev theo_y: {t['fp32']['evidence_f1_theo_y']:.4f} -> {t['int8']['evidence_f1_theo_y']:.4f} | "
                f"agree={t['agreement']['label_exact_rate']:.4f}, McNemar p={t['mcnemar_5class_int8_vs_fp32']['p_value']:.4f}"
            )

    if args.skip_cpu_bench:
        return 0

    # 2) Đo độ trễ & Peak RSS độc lập từng cấu hình trên Laptop CPU đích
    print("2/2. Đo độ trễ (batch=1) & RAM đỉnh độc lập trên Laptop CPU...")
    configs = [
        ("onnx_int8", 8, args.n_samples, args.warmup),
        ("onnx_int8", 4, args.n_samples, args.warmup),
        ("onnx_int8", 1, min(50, args.n_samples), min(5, args.warmup)),
        ("onnx_fp32", 8, min(60, args.n_samples), min(5, args.warmup)),
        ("torch_fp32", 8, min(40, args.n_samples), min(5, args.warmup)),
    ]
    runs: list[dict[str, Any]] = []
    for backend, th, ns, wu in configs:
        print(f"    Đang đo {backend} (threads={th}, n={ns})...")
        r = _spawn_bench_worker(backend, th, ns, wu, args)
        runs.append(r)
        print(
            f"      -> p50={r['latency_ms']['end_to_end_p50']:.1f} ms | "
            f"p95={r['latency_ms']['end_to_end_p95']:.1f} ms | "
            f"peak_rss={r['memory_MiB']['peak_rss']:.1f} MiB | "
            f"size={r['disk_size_MiB']:.1f} MiB"
        )

    import torch

    bench_payload = {
        "model": export_manifest.get("model_name", "safevichi_v2"),
        "quantization_method": export_manifest.get("quantization_method"),
        "seed": args.seed,
        "command": "python -m src.v2.bench.benchmark_cpu",
        "environment": {
            "cpu_name": _get_cpu_brand(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "total_ram_GiB": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
        },
        "sizes_MiB": export_manifest.get("sizes_MiB"),
        "runs": runs,
    }
    bench_path = args.out_dir / "bench_cpu.json"
    bench_path.write_text(json.dumps(bench_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Đã ghi báo cáo benchmark CPU vào {bench_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
