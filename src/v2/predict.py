"""CLI suy luận cục bộ (on-device) cho SafeViChi v2 qua ONNX Runtime.

Chạy hoàn toàn trên CPU với ONNX Hybrid INT8 (`safevichi_v2_int8.onnx`), KHÔNG nạp PyTorch
hay Transformers lúc suy luận (chỉ dùng `onnxruntime`, `tokenizers`, `numpy`).
Mô hình suy luận trực tiếp trên văn bản gốc, không dùng bộ chuẩn hóa trong luồng mặc định.

Ví dụ sử dụng sau khi cài `pip install -e .`:
    # 1. Mở chế độ tương tác liên tục (nạp mô hình 1 lần)
    safevichi
    safevichi --chat   # Cửa sổ trượt hội thoại 1–8 lượt

    # 2. Quét nhanh 1 tin nhắn hoặc hội thoại nhiều lượt (ngăn lượt bằng "||")
    safevichi "Tài khoản sẽ bị khóa sau 2 giờ, bấm link xác minh OTP ngay"
    safevichi "Em không muốn nói chuyện nữa || Đừng giả vờ, tối nay đi với anh không thì biết tay"

    # 3. Trả về JSON thuần (tiện tích hợp ứng dụng / script)
    safevichi --json "Mày là đồ ngu"

    # 4. Quét hàng loạt tệp .txt hoặc .jsonl
    safevichi --jsonl data/v2/test_day_du.jsonl --out preds.jsonl
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.shared.normalization.normalizer import (
    DEPRECATION_NOTICE as LEGACY_NORM_NOTICE,
    normalize as legacy_normalize_v1,
)
from src.v2.download_model import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    DEFAULT_ONNX_DIR,
    REPO_ROOT,
    ensure_onnx_model,
    missing_onnx_files,
)

__version__ = "2.0.0"

OUTPUT_NAMES = ["logits", "emissions", "severity"]
LABELS_DEFAULT = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")

_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]")

LABEL_ACTIONS: dict[str, dict[str, str]] = {
    "AN_TOAN": {
        "ten": "An toàn",
        "hanh_dong": "Hiển thị bình thường",
        "ansi": "\033[92m",
        "badge": "\033[42;30;1m",
    },
    "XUC_PHAM": {
        "ten": "Xúc phạm / Lăng mạ",
        "hanh_dong": "Làm mờ tin nhắn, bấm mới xem",
        "ansi": "\033[93m",
        "badge": "\033[43;30;1m",
    },
    "QUAY_ROI": {
        "ten": "Quấy rối / Bêu riếu",
        "hanh_dong": "Gợi ý chặn và báo cáo người gửi",
        "ansi": "\033[95m",
        "badge": "\033[45;97;1m",
    },
    "DE_DOA": {
        "ten": "Đe dọa / Tống tiền",
        "hanh_dong": "Cảnh báo mạnh, gợi ý lưu bằng chứng",
        "ansi": "\033[91m",
        "badge": "\033[41;97;1m",
    },
    "LUA_DAO": {
        "ten": "Lừa đảo / Dụ OTP & link",
        "hanh_dong": "Cảnh báo lừa đảo, vô hiệu hóa liên kết",
        "ansi": "\033[91;1m",
        "badge": "\033[41;97;1m",
    },
}

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_HIGHLIGHT = "\033[41;97;1m"


def _strip_invisibles(text: str) -> str:
    return _INVISIBLE_RE.sub("", text)


# ==============================================================================
# 1. TÙY CHỌN KẾ THỪA LỖI THỜI (DEPRECATED): BỘ CHUẨN HÓA V1 GIỮ XUỐNG DÒNG
# ==============================================================================


def legacy_normalize_v1_with_alignment(text: str) -> tuple[str, list[int]]:
    """[Lỗi thời] Chạy bộ chuẩn hóa 5 bước của SafeViChi v1 theo từng dòng và ánh xạ chỉ số từ về bản gốc.

    Chỉ dùng khi người dùng chủ động bật cờ `--legacy-normalize`. Không dùng trong luồng mặc định.
    """
    norm_lines: list[str] = []
    norm_to_orig: list[int] = []
    orig_idx = 0
    for line in text.split("\n"):
        out_tokens: list[str] = []
        for match in re.finditer(r"\S+", line):
            raw_tok = match.group(0)
            norm_tok = legacy_normalize_v1(raw_tok)
            pieces = norm_tok.split() if norm_tok.strip() else [raw_tok]
            for piece in pieces:
                out_tokens.append(piece)
                norm_to_orig.append(orig_idx)
            orig_idx += 1
        norm_lines.append(" ".join(out_tokens))
    return "\n".join(norm_lines), norm_to_orig


# ==============================================================================
# 2. TIỀN XỬ LÝ & GIẢI MÃ BIO-CRF THUẦN NUMPY (KHÔNG CẦN PYTORCH / TRANSFORMERS)
# ==============================================================================


def _allocate_turn_budgets(turn_lens: list[int], budget: int) -> list[int]:
    n_t = len(turn_lens)
    if sum(turn_lens) <= budget:
        return list(turn_lens)
    caps = [0] * n_t
    rem = budget
    active = list(range(n_t))
    while active and rem > 0:
        share = max(1, rem // len(active))
        next_active: list[int] = []
        for i in active:
            need = turn_lens[i] - caps[i]
            take = min(need, share, rem)
            caps[i] += take
            rem -= take
            if caps[i] < turn_lens[i]:
                next_active.append(i)
        if not next_active or (len(next_active) == len(active) and share == 1 and rem == 0):
            break
        active = next_active
    return caps


def _logsumexp_np(arr: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def _crf_marginals_single_np(
    emissions: np.ndarray,
    turn_starts: list[bool],
    start_tr: np.ndarray,
    end_tr: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    w_len = len(turn_starts)
    if w_len == 0:
        return np.zeros((0, 3), dtype=np.float64)
    neg_inf = -1e4
    em = np.asarray(emissions[:w_len], dtype=np.float64)
    st = np.asarray(start_tr, dtype=np.float64).copy()
    st[2] = neg_inf
    tr = np.asarray(trans, dtype=np.float64).copy()
    tr[0, 2] = neg_inf
    en = np.asarray(end_tr, dtype=np.float64)

    alphas = np.zeros((w_len, 3), dtype=np.float64)
    alphas[0] = st + em[0]
    for t in range(1, w_len):
        step = tr.copy()
        if turn_starts[t]:
            step[:, 2] = neg_inf
        alphas[t] = _logsumexp_np(alphas[t - 1][:, None] + step, axis=0) + em[t]
    log_z = float(_logsumexp_np(alphas[w_len - 1] + en, axis=0))

    betas = np.zeros((w_len, 3), dtype=np.float64)
    betas[w_len - 1] = en
    for t in range(w_len - 2, -1, -1):
        step = tr.copy()
        if turn_starts[t + 1]:
            step[:, 2] = neg_inf
        betas[t] = _logsumexp_np(step + (em[t + 1] + betas[t + 1])[None, :], axis=1)

    probs = np.exp(alphas + betas - log_z)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
    return probs


def _viterbi_single_np(
    emissions: np.ndarray,
    turn_starts: list[bool],
    start_tr: np.ndarray,
    end_tr: np.ndarray,
    trans: np.ndarray,
    o_bias: float = 0.0,
) -> list[int]:
    w_len = len(turn_starts)
    if w_len == 0:
        return []
    em = np.asarray(emissions[:w_len], dtype=np.float64).copy()
    em[:, 0] += float(o_bias)
    st = np.asarray(start_tr, dtype=np.float64).copy()
    st[2] = -1e4
    tr = np.asarray(trans, dtype=np.float64).copy()
    tr[0, 2] = -1e4

    dp = st + em[0]
    bp = np.zeros((w_len, 3), dtype=np.int64)
    for t in range(1, w_len):
        step = tr.copy()
        if turn_starts[t]:
            step[:, 2] = -1e4
        cand = dp[:, None] + step
        bp[t] = cand.argmax(axis=0)
        dp = cand[bp[t], np.arange(3)] + em[t]
    dp = dp + np.asarray(end_tr, dtype=np.float64)
    seq = [0] * w_len
    cur = int(dp.argmax())
    seq[w_len - 1] = cur
    for t in range(w_len - 2, -1, -1):
        cur = int(bp[t + 1, cur])
        seq[t] = cur
    return seq


def _apply_marginal_postproc(
    words: list[tuple[int, int, float, int, bool]],
    text: str,
    threshold: float,
    single_word_threshold: float | None,
    bridge_ratio: float,
) -> list[bool]:
    n_w = len(words)
    marks = [w[2] >= threshold for w in words]
    if bridge_ratio > 0 and n_w >= 3:
        bridge_thr = threshold * bridge_ratio
        bridged = marks[:]
        for i in range(1, n_w - 1):
            if not marks[i] and marks[i - 1] and marks[i + 1] and words[i][2] >= bridge_thr:
                if "\n" not in text[words[i - 1][1] : words[i + 1][0]]:
                    bridged[i] = True
        marks = bridged
    if single_word_threshold is not None and single_word_threshold > threshold:
        filtered = marks[:]
        for i in range(n_w):
            if not marks[i]:
                continue
            left_same = i > 0 and marks[i - 1] and "\n" not in text[words[i - 1][1] : words[i][0]]
            right_same = i + 1 < n_w and marks[i + 1] and "\n" not in text[words[i][1] : words[i + 1][0]]
            if not left_same and not right_same and words[i][2] < single_word_threshold:
                filtered[i] = False
        marks = filtered
    return marks


def _apply_empty_fallback(
    marks: list[bool],
    words: list[tuple[int, int, float, int, bool]],
    text: str,
    fallback_min_prob: float | None,
    expand_ratio: float = 0.70,
    max_words: int = 5,
) -> list[bool]:
    if any(marks) or not words or fallback_min_prob is None or fallback_min_prob > 1.0:
        return marks
    probs = [w[2] for w in words]
    peak_i = int(np.argmax(probs))
    peak_p = probs[peak_i]
    if peak_p < fallback_min_prob:
        return marks
    out = list(marks)
    out[peak_i] = True
    cut = peak_p * expand_ratio
    lo = hi = peak_i
    while (hi - lo + 1) < max_words:
        can_l = lo > 0 and "\n" not in text[words[lo - 1][1] : words[lo][0]] and probs[lo - 1] >= cut
        can_r = hi + 1 < len(words) and "\n" not in text[words[hi][1] : words[hi + 1][0]] and probs[hi + 1] >= cut
        if not can_l and not can_r:
            break
        if can_l and (not can_r or probs[lo - 1] >= probs[hi + 1]):
            lo -= 1
            out[lo] = True
        else:
            hi += 1
            out[hi] = True
    return out


def _marks_to_runs(words: list[tuple[int, int, float, int, bool]], marks: list[bool], text: str) -> list[list[int]]:
    runs: list[list[int]] = []
    for w, marked in zip(words, marks):
        if not marked:
            continue
        a, b = w[0], w[1]
        if runs:
            gap = text[runs[-1][1] : a]
            if "\n" not in gap and re.fullmatch(r"[^\S\n]*", _strip_invisibles(gap)):
                runs[-1][1] = b
                continue
        runs.append([a, b])
    return runs


def normalize_input_turns(raw_text: str) -> str:
    """Định dạng đầu vào hội thoại: hỗ trợ '||' hoặc '\\n' ngăn lượt, giữ tối đa 8 lượt gần nhất."""
    text = raw_text.replace("\\n", "\n")
    if "\n" not in text and "||" in text:
        lines = [part.strip() for part in text.split("||")]
    else:
        lines = [part.strip() for part in text.splitlines()]
    lines = [ln for ln in lines if ln and _strip_invisibles(ln).strip()]
    if len(lines) > 8:
        lines = lines[-8:]
    return "\n".join(lines)


class SafeViChiPredictor:
    """Bộ suy luận SafeViChi v2 thuần ONNX Runtime (không cần PyTorch/Transformers)."""

    def __init__(
        self,
        onnx_dir: Path = DEFAULT_ONNX_DIR,
        use_fp32: bool = False,
        threads: int | None = None,
        auto_download: bool = False,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_revision: str = DEFAULT_HF_REVISION,
    ) -> None:
        self.onnx_dir = Path(onnx_dir)
        if auto_download and missing_onnx_files(self.onnx_dir, use_fp32=use_fp32):
            ensure_onnx_model(
                self.onnx_dir,
                use_fp32=use_fp32,
                repo_id=hf_repo_id,
                revision=hf_revision,
            )

        crf_path = self.onnx_dir / "crf_params.json"
        if not crf_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy bộ mô hình ONNX tại: {self.onnx_dir}.\n"
                f"Chạy `python -m src.v2.download_model` (hoặc `safevichi --download`) để tự tải từ "
                f"https://huggingface.co/{hf_repo_id}, hoặc truyền `--onnx-dir <duong_dan>` / `SAFEVICHI_ONNX_DIR`."
            )
        self.crf_meta = json.loads(crf_path.read_text(encoding="utf-8"))
        self.model_path = self.onnx_dir / ("safevichi_v2_fp32.onnx" if use_fp32 else "safevichi_v2_int8.onnx")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy tệp mô hình ONNX: {self.model_path}.\n"
                f"Chạy `python -m src.v2.download_model{' --fp32' if use_fp32 else ''}` để tải từ Hugging Face "
                "hoặc chạy `python -m src.v2.bench.export_onnx`."
            )
        tok_json = self.onnx_dir / "tokenizer" / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"Không tìm thấy tokenizer.json tại {tok_json}.")

        self.tokenizer = Tokenizer.from_file(str(tok_json))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

        self.labels = tuple(self.crf_meta.get("labels", LABELS_DEFAULT))
        self.max_tokens = int(self.crf_meta.get("max_tokens", 512))
        self.max_word_bytes = int(self.crf_meta.get("max_word_bytes", 24))
        self.max_subwords_per_word = int(self.crf_meta.get("max_subwords_per_word", 12))
        self.bos_id = int(self.crf_meta["bos_id"])
        self.sep_id = int(self.crf_meta["sep_id"])
        self.pad_id = int(self.crf_meta["pad_id"])
        self.start_tr = np.asarray(self.crf_meta["start"], dtype=np.float64)
        self.end_tr = np.asarray(self.crf_meta["end"], dtype=np.float64)
        self.trans = np.asarray(self.crf_meta["trans"], dtype=np.float64)

        n_threads = int(threads) if threads is not None else min(8, os.cpu_count() or 4)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, n_threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        self.threads = max(1, n_threads)

    def _prepare_single(self, text: str) -> dict[str, Any]:
        lines = text.split("\n")
        n_turns = len(lines)
        words_meta: list[tuple[int, int, int, int, bool, list[int]]] = []
        cursor = 0
        for t_idx, line in enumerate(lines):
            recency = n_turns - 1 - t_idx
            first_in_turn = True
            for m in re.finditer(r"\S+", line):
                a, b = cursor + m.start(), cursor + m.end()
                b_arr = [x + 1 for x in m.group(0).encode("utf-8")[: self.max_word_bytes]] or [1]
                words_meta.append((a, b, t_idx, recency, first_in_turn, b_arr))
                first_in_turn = False
            cursor += len(line) + 1

        if not words_meta:
            raise ValueError("Văn bản rỗng hoặc không có từ nào.")

        turn_enc: list[tuple[list[int], list[tuple[int, int]]]] = []
        cursor = 0
        for line in lines:
            enc = self.tokenizer.encode(line, add_special_tokens=False)
            t_ids = list(enc.ids)
            t_offs: list[tuple[int, int]] = []
            for sa, sb in enc.offsets:
                while sa < sb and sb < len(line) and unicodedata.combining(line[sb]):
                    sb += 1
                t_offs.append((cursor + sa, cursor + sb))
            turn_enc.append((t_ids, t_offs))
            cursor += len(line) + 1

        budget = max(16, self.max_tokens - (1 + n_turns))
        caps = _allocate_turn_budgets([len(x[0]) for x in turn_enc], budget)
        seq_ids = [self.bos_id]
        seq_offs: list[tuple[int, int]] = [(0, 0)]
        for (t_ids, t_offs), cap in zip(turn_enc, caps):
            if len(t_ids) > cap:
                head = cap // 3
                tail = cap - head
                kept_ids = t_ids[:head] + t_ids[-tail:] if tail > 0 else t_ids[:head]
                kept_offs = t_offs[:head] + t_offs[-tail:] if tail > 0 else t_offs[:head]
            else:
                kept_ids, kept_offs = t_ids, t_offs
            seq_ids.extend(kept_ids)
            seq_offs.extend(kept_offs)
            seq_ids.append(self.sep_id)
            seq_offs.append((0, 0))

        word_sub_idxs: list[list[int]] = []
        pos_tok = 1
        n_seq = len(seq_offs)
        for a, b, *_ in words_meta:
            while pos_tok < n_seq and (seq_offs[pos_tok][0] == seq_offs[pos_tok][1] or seq_offs[pos_tok][1] <= a):
                pos_tok += 1
            idxs: list[int] = []
            j = pos_tok
            while j < n_seq:
                sa, sb = seq_offs[j]
                if sa == sb:
                    j += 1
                    continue
                if sa >= b:
                    break
                if sb > a:
                    idxs.append(j)
                j += 1
            word_sub_idxs.append(idxs[: self.max_subwords_per_word])

        max_l = len(seq_ids)
        max_w = len(words_meta)
        max_s = max(1, max((len(s) for s in word_sub_idxs), default=1))

        ids = np.full((1, max_l), self.pad_id, dtype=np.int64)
        attn_mask = np.zeros((1, max_l), dtype=np.bool_)
        word_sub_idx = np.zeros((1, max_w, max_s), dtype=np.int64)
        word_sub_mask = np.zeros((1, max_w, max_s), dtype=np.bool_)
        word_mask = np.zeros((1, max_w), dtype=np.bool_)
        word_turn = np.zeros((1, max_w), dtype=np.int64)
        word_recency = np.zeros((1, max_w), dtype=np.int64)
        word_bytes = np.zeros((1, max_w, self.max_word_bytes), dtype=np.int64)

        ids[0, :max_l] = seq_ids
        attn_mask[0, :max_l] = True
        for w_i, (meta, sub_idxs) in enumerate(zip(words_meta, word_sub_idxs)):
            _, _, t_idx, recency, _, b_arr = meta
            word_mask[0, w_i] = True
            word_turn[0, w_i] = t_idx
            word_recency[0, w_i] = recency
            word_bytes[0, w_i, : len(b_arr)] = b_arr
            if sub_idxs:
                word_sub_idx[0, w_i, : len(sub_idxs)] = sub_idxs
                word_sub_mask[0, w_i, : len(sub_idxs)] = True

        return {
            "words_meta": words_meta,
            "feed": {
                "ids": ids,
                "attn_mask": attn_mask,
                "word_sub_idx": word_sub_idx,
                "word_sub_mask": word_sub_mask,
                "word_mask": word_mask,
                "word_turn": word_turn,
                "word_recency": word_recency,
                "word_bytes": word_bytes,
            },
        }

    def _run_forward(
        self, model_text: str
    ) -> tuple[np.ndarray, float, list[tuple[int, int, int, int, bool, list[int]]], np.ndarray, list[bool], np.ndarray]:
        prep = self._prepare_single(model_text)
        logits_b, emit_b, sev_b = self.session.run(OUTPUT_NAMES, prep["feed"])

        logits = logits_b[0].astype(np.float64)
        temp = float(self.crf_meta["temperature"])
        shifted = (logits / temp) - np.max(logits / temp)
        prob = np.exp(shifted)
        prob = prob / np.sum(prob)
        severity = float(sev_b[0])

        words_meta_model = prep["words_meta"]
        nw_model = len(words_meta_model)
        em_model = emit_b[0, :nw_model]
        ts_model = [bool(m[4]) for m in words_meta_model]
        marg_model = _crf_marginals_single_np(em_model, ts_model, self.start_tr, self.end_tr, self.trans)
        ev_probs_model = np.clip(marg_model[:, 1] + marg_model[:, 2], 0.0, 1.0)
        return prob, severity, words_meta_model, em_model, ts_model, ev_probs_model

    def predict(
        self,
        raw_text: str,
        legacy_normalize: bool = False,
    ) -> dict[str, Any]:
        """Phân loại 5 nhãn và trích xuất cụm từ bằng chứng verbatim cho 1 đoạn hội thoại (1–8 lượt).

        Mặc định suy luận trực tiếp trên văn bản gốc (`legacy_normalize=False`).
        `legacy_normalize=True` *(lỗi thời)* chỉ giữ lại để gọi bộ chuẩn hóa v1 khi đối chiếu.
        """
        text = normalize_input_turns(raw_text)
        if not text:
            raise ValueError("Tin nhắn đầu vào đang rỗng.")

        t0 = time.perf_counter()
        norm_to_orig: list[int] | None = None
        model_text = text

        if legacy_normalize:
            model_text, norm_to_orig = legacy_normalize_v1_with_alignment(text)

        prob, severity, words_meta_model, em_model, ts_model, ev_probs_model = self._run_forward(model_text)
        nw_model = len(words_meta_model)

        if legacy_normalize and norm_to_orig is not None:
            orig_words_raw: list[tuple[int, int, int, bool]] = []
            cursor = 0
            for t_idx, line in enumerate(text.split("\n")):
                first_in_turn = True
                for m in re.finditer(r"\S+", line):
                    orig_words_raw.append((cursor + m.start(), cursor + m.end(), t_idx, first_in_turn))
                    first_in_turn = False
                cursor += len(line) + 1
            by_orig: dict[int, list[int]] = defaultdict(list)
            for j, oi in enumerate(norm_to_orig):
                if j < nw_model:
                    by_orig[oi].append(j)
            words = []
            seen_turns: set[int] = set()
            for oi, (a, b, t_idx, _is_ts) in enumerate(orig_words_raw):
                if not _strip_invisibles(text[a:b]):
                    continue
                is_first_vis = t_idx not in seen_turns
                seen_turns.add(t_idx)
                js = by_orig.get(oi)
                sc = float(max(ev_probs_model[j] for j in js)) if js else 0.0
                words.append((a, b, sc, t_idx, is_first_vis))
        else:
            words = [
                (m[0], m[1], float(ev_probs_model[i]), int(m[2]), bool(m[4]))
                for i, m in enumerate(words_meta_model)
            ]

        pred_idx = int(np.argmax(prob))
        label = self.labels[pred_idx]

        runs: list[list[int]] = []
        if pred_idx != 0 and words:
            mode = str(self.crf_meta.get("chosen_decoder", "marginal"))
            fb = self.crf_meta.get("fallback_min_prob")
            if mode == "crf" and not legacy_normalize:
                tags = _viterbi_single_np(
                    em_model,
                    ts_model,
                    self.start_tr,
                    self.end_tr,
                    self.trans,
                    o_bias=float(self.crf_meta.get("o_bias", 0.0)),
                )
                if any(t != 0 for t in tags):
                    for w, tag in zip(words, tags):
                        if tag == 0:
                            continue
                        a, b = w[0], w[1]
                        if tag == 2 and runs and re.fullmatch(r"[^\S\n]*", text[runs[-1][1] : a]):
                            runs[-1][1] = b
                        else:
                            runs.append([a, b])
                elif fb is not None:
                    marks = _apply_empty_fallback([False] * len(words), words, text, float(fb))
                    runs = _marks_to_runs(words, marks, text)
            else:
                marks = _apply_marginal_postproc(
                    words,
                    text,
                    threshold=float(self.crf_meta["marginal_threshold"]),
                    single_word_threshold=self.crf_meta.get("single_word_threshold"),
                    bridge_ratio=float(self.crf_meta.get("bridge_ratio", 0.0)),
                )
                if fb is not None:
                    marks = _apply_empty_fallback(marks, words, text, float(fb))
                runs = _marks_to_runs(words, marks, text)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        info = LABEL_ACTIONS.get(label, {"ten": label, "hanh_dong": ""})
        out: dict[str, Any] = {
            "nhan": label,
            "ten_nhan": info["ten"],
            "hanh_dong": info["hanh_dong"],
            "tin_cay": round(float(prob[pred_idx]), 6),
            "probs": {lbl: round(float(p), 6) for lbl, p in zip(self.labels, prob)},
            "rui_ro": round(severity * 100.0, 2),
            "bang_chung": [text[a:b] for a, b in runs],
            "vi_tri": runs,
            "text": text,
            "thoi_gian_ms": round(elapsed_ms, 2),
        }
        if legacy_normalize:
            out["legacy_normalize"] = True
        return out


# ==============================================================================
# 3. HIỂN THỊ ĐẸP TRÊN CMD WINDOWS & CÁC CHẾ ĐỘ CLI
# ==============================================================================


def _init_windows_utf8() -> None:
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _enable_windows_ansi() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.name != "nt":
        return sys.stdout.isatty()
    if not sys.stdout.isatty():
        return False
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return True
    except Exception:
        return False
    return False


def _prob_bar(prob: float, width: int = 16) -> str:
    filled = max(0, min(width, int(round(prob * width))))
    return "█" * filled + "░" * (width - filled)


def highlight_text(text: str, spans: list[list[int]], use_color: bool) -> str:
    """Tô sáng các cụm bằng chứng trực tiếp trong văn bản gốc (giữ nguyên xuống dòng)."""
    if not spans:
        return text
    ordered = sorted(spans, key=lambda x: (x[0], x[1]))
    parts: list[str] = []
    pos = 0
    for a, b in ordered:
        if a < pos:
            continue
        parts.append(text[pos:a])
        chunk = text[a:b]
        if use_color:
            parts.append(f"{ANSI_HIGHLIGHT}{chunk}{ANSI_RESET}")
        else:
            parts.append(f"[[{chunk}]]")
        pos = b
    parts.append(text[pos:])
    return "".join(parts)


def render_pretty(res: dict[str, Any], use_color: bool) -> str:
    """Hiển thị thẻ kết quả gọn, căn lề sạch, tương thích tốt mọi độ rộng cửa sổ CMD/PowerShell."""
    lbl = res["nhan"]
    info = LABEL_ACTIONS.get(lbl, {"ten": lbl, "hanh_dong": "", "ansi": "", "badge": ""})
    c_lbl = info["ansi"] if use_color else ""
    c_badge = info.get("badge", info["ansi"]) if use_color else ""
    c_bold = ANSI_BOLD if use_color else ""
    c_dim = ANSI_DIM if use_color else ""
    c_warn = ANSI_YELLOW if use_color else ""
    c_rst = ANSI_RESET if use_color else ""

    rail = f"{c_lbl}┃{c_rst}"
    is_legacy = bool(res.get("legacy_normalize"))
    legacy_tag = f"  {c_warn}[Chuẩn hóa v1 · Lỗi thời]{c_rst}" if is_legacy else ""
    badge = f"{c_badge} {lbl} {c_rst}" if use_color else f"[{lbl}]"
    title_plain = "SafeViChi v2  [Chuẩn hóa v1 · Lỗi thời]" if is_legacy else "SafeViChi v2"
    card_width = 66
    header_rule = "─" * max(8, card_width - len(title_plain) - 3)

    lines = [
        f"{c_lbl}╭─{c_rst} {c_bold}SafeViChi v2{c_rst}{legacy_tag} {c_dim}{header_rule}{c_rst}",
        (
            f"{rail}  {badge} {c_bold}{info['ten']}{c_rst}   "
            f"{c_dim}Tin cậy{c_rst} {c_bold}{res['tin_cay'] * 100:.1f}%{c_rst}  ·  "
            f"{c_dim}Rủi ro{c_rst} {c_bold}{res['rui_ro']:.1f}/100{c_rst}  ·  "
            f"{c_dim}{res['thoi_gian_ms']:.1f} ms{c_rst}"
        ),
        f"{rail}  {c_dim}Hành động:{c_rst}  {info['hanh_dong']}",
    ]
    if is_legacy:
        lines.append(
            f"{rail}  {c_dim}Lưu ý:{c_rst}      {c_warn}Đang bật --legacy-normalize (v1 lỗi thời; không khuyến nghị).{c_rst}"
        )
    lines.append(f"{rail}")

    hl = highlight_text(res["text"], res["vi_tri"], use_color=use_color)
    turns = hl.split("\n")
    if len(turns) == 1:
        lines.append(f"{rail}  {c_dim}Nội dung:{c_rst}   {turns[0]}")
    else:
        lines.append(f"{rail}  {c_dim}Hội thoại ({len(turns)} lượt):{c_rst}")
        for idx, t in enumerate(turns, 1):
            speaker = "A" if idx % 2 == 1 else "B"
            lines.append(f"{rail}    {c_dim}{idx} [{speaker}]{c_rst}  {t}")

    if res["bang_chung"]:
        quoted = "  ·  ".join(f'"{p}"' for p in res["bang_chung"])
        lines.append(f"{rail}  {c_dim}Bằng chứng:{c_rst} {c_lbl}{quoted}{c_rst}")
    elif lbl != "AN_TOAN":
        lines.append(f"{rail}  {c_dim}Bằng chứng: (không trích cụm rõ ràng){c_rst}")

    sorted_probs = sorted(res["probs"].items(), key=lambda kv: kv[1], reverse=True)
    top_k, top_v = sorted_probs[0]
    others = "  ·  ".join(f"{k} {v * 100:.1f}%" for k, v in sorted_probs[1:])
    lines.append(
        f"{rail}  {c_dim}Phân bố:{c_rst}    "
        f"{c_lbl}{top_k} {_prob_bar(top_v, 14)} {top_v * 100:.1f}%{c_rst}  "
        f"{c_dim}|  {others}{c_rst}"
    )
    lines.append(f"{c_lbl}╰{'─' * 66}{c_rst}")
    return "\n".join(lines)


def _print_interactive_help(use_color: bool) -> None:
    c_bold = ANSI_BOLD if use_color else ""
    c_cyan = ANSI_CYAN if use_color else ""
    c_dim = ANSI_DIM if use_color else ""
    c_rst = ANSI_RESET if use_color else ""
    cmds = [
        ("/chat", "Bật/tắt chế độ cửa sổ hội thoại trượt 1–8 lượt"),
        ("/history", "Xem các lượt đang có trong cửa sổ hội thoại"),
        ("/undo", "Bỏ lượt vừa nhập trong chế độ /chat"),
        ("/clear", "Xóa toàn bộ lịch sử hội thoại"),
        ("/json", "Bật/tắt định dạng đầu ra JSON thuần"),
        ("/help", "Hiển thị bảng lệnh này"),
        ("/quit", "Thoát chương trình (hoặc nhấn Ctrl+C)"),
    ]
    print(f"{c_bold}Danh sách lệnh nhanh:{c_rst}")
    for cmd, desc in cmds:
        print(f"  {c_cyan}{cmd:<10}{c_rst} {c_dim}{desc}{c_rst}")
    print(f"  {c_dim}Mẹo: dùng '||' để nhập nhiều lượt trong 1 dòng; gõ '//...' nếu tin nhắn bắt đầu bằng '/'.{c_rst}\n")


def _run_interactive(
    predictor: SafeViChiPredictor,
    chat_mode: bool,
    legacy_normalize: bool,
    as_json: bool,
    use_color: bool,
) -> int:
    c_bold = ANSI_BOLD if use_color else ""
    c_dim = ANSI_DIM if use_color else ""
    c_cyan = ANSI_CYAN if use_color else ""
    c_warn = ANSI_YELLOW if use_color else ""
    c_red = ANSI_RED if use_color else ""
    c_rst = ANSI_RESET if use_color else ""

    prec_tag = "FP32" if "fp32" in predictor.model_path.name.lower() else "INT8"
    mode_name = "Hội thoại 1–8 lượt" if chat_mode else "Quét từng tin nhắn"
    print(
        f"{c_cyan}╭─{c_rst} {c_bold}SafeViChi v{__version__}{c_rst} "
        f"{c_dim}· ONNX {prec_tag} · {predictor.threads} luồng CPU · nạp {predictor.load_ms:.0f} ms{c_rst}"
    )
    print(f"{c_cyan}┃{c_rst}  Chế độ: {c_bold}{mode_name}{c_rst}")
    if legacy_normalize:
        print(f"{c_cyan}┃{c_rst}  {c_warn}[Lỗi thời] Đang bật --legacy-normalize (bộ chuẩn hóa v1).{c_rst}")
    print(f"{c_cyan}╰─{c_rst} {c_dim}Gõ tin nhắn rồi nhấn Enter · dùng '||' ngăn lượt · /help xem lệnh · /quit thoát{c_rst}\n")

    history: list[str] = []
    while True:
        if chat_mode:
            prompt = (
                f"{c_cyan}safevichi{c_rst} {c_dim}[{len(history) + 1}/8]{c_rst} › "
                if use_color
                else f"Lượt {len(history) + 1}> "
            )
        else:
            prompt = f"{c_cyan}safevichi{c_rst} › " if use_color else "SafeViChi> "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{c_dim}Đã thoát SafeViChi.{c_rst}")
            return 0
        if not line:
            continue

        if line.startswith("//"):
            line = line[1:]
        elif line.startswith("/") or line.lower() in {"exit", "quit"}:
            low = line.lower()
            if low in {"/quit", "/exit", "exit", "quit"}:
                print(f"{c_dim}Đã thoát SafeViChi.{c_rst}")
                return 0
            if low in {"/help", "/?"}:
                _print_interactive_help(use_color)
                continue
            if low == "/clear":
                history.clear()
                print(f"{c_dim}Đã xóa lịch sử hội thoại.{c_rst}\n")
                continue
            if low == "/undo":
                if history:
                    removed = history.pop()
                    print(f'{c_dim}Đã bỏ lượt cuối: "{removed}" (còn {len(history)} lượt).{c_rst}\n')
                else:
                    print(f"{c_dim}Chưa có lượt nào trong lịch sử.{c_rst}\n")
                continue
            if low == "/history":
                if not history:
                    print(f"{c_dim}Lịch sử hội thoại đang trống.{c_rst}\n")
                else:
                    print(f"{c_bold}Cửa sổ hội thoại hiện tại ({len(history)}/8 lượt):{c_rst}")
                    for idx, turn in enumerate(history, 1):
                        speaker = "A" if idx % 2 == 1 else "B"
                        print(f"  {c_dim}{idx} [{speaker}]{c_rst}  {turn}")
                    print()
                continue
            if low == "/chat":
                chat_mode = not chat_mode
                history.clear()
                print(
                    f"{c_dim}Đã chuyển chế độ: {'Hội thoại 1–8 lượt' if chat_mode else 'Quét từng tin nhắn'}.{c_rst}\n"
                )
                continue
            if low == "/json":
                as_json = not as_json
                print(f"{c_dim}Định dạng xuất JSON: {'BẬT' if as_json else 'TẮT'}.{c_rst}\n")
                continue
            print(
                f"{c_red}Lệnh không hợp lệ:{c_rst} {line} "
                f"{c_dim}(gõ /help để xem lệnh, hoặc '//...' để quét chữ bắt đầu bằng '/'){c_rst}\n"
            )
            continue

        cleaned_in = normalize_input_turns(line)
        if not cleaned_in:
            print(f"{c_red}Cảnh báo:{c_rst} Tin nhắn rỗng hoặc chỉ chứa ký tự tàng hình.\n")
            continue

        if chat_mode:
            history.extend(cleaned_in.splitlines())
            if len(history) > 8:
                history = history[-8:]
            text_in = "\n".join(history)
        else:
            text_in = cleaned_in

        try:
            res = predictor.predict(text_in, legacy_normalize=legacy_normalize)
        except ValueError as exc:
            print(f"{c_red}Lỗi:{c_rst} {exc}\n")
            continue
        if as_json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            print(render_pretty(res, use_color=use_color))
        print()


def _run_batch(
    predictor: SafeViChiPredictor,
    in_path: Path,
    is_jsonl: bool,
    out_path: Path | None,
    legacy_normalize: bool,
) -> int:
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp đầu vào: {in_path}")
    rows_in: list[dict[str, Any]] = []
    for idx, line in enumerate(in_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if is_jsonl:
            rows_in.append(json.loads(line))
        else:
            rows_in.append({"id": f"line_{idx:05d}", "text": line})

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for row in rows_in:
        pred = predictor.predict(str(row["text"]), legacy_normalize=legacy_normalize)
        rec_out: dict[str, Any] = {
            "id": row.get("id", ""),
            "nhan": pred["nhan"],
            "tin_cay": pred["tin_cay"],
            "probs": [pred["probs"][lbl] for lbl in predictor.labels],
            "rui_ro": pred["rui_ro"],
            "bang_chung": pred["bang_chung"],
            "vi_tri": pred["vi_tri"],
            "hanh_dong": pred["hanh_dong"],
        }
        if legacy_normalize:
            rec_out["legacy_normalize"] = True
        if "nhan" in row:
            rec_out["nhan_dung"] = row["nhan"]
        results.append(rec_out)
    elapsed = time.perf_counter() - t0

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"Đã quét {len(results)} mẫu trong {elapsed:.2f}s ({len(results) / max(elapsed, 1e-6):.1f} mẫu/s) -> {out_path}"
        )
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "Ví dụ nhanh:\n"
        '  safevichi                                        Mở chế độ tương tác liên tục\n'
        '  safevichi --chat                                 Mô phỏng cửa sổ chat trượt 1–8 lượt\n'
        '  safevichi "Bấm link xác minh OTP ngay"           Quét nhanh 1 tin nhắn\n'
        '  safevichi "Lượt 1 || Lượt 2"                     Quét hội thoại nhiều lượt\n'
        '  safevichi --json "Mày là đồ ngu"                 Xuất JSON thuần cho script/ứng dụng\n'
        '  safevichi --file messages.txt --out kq.jsonl     Quét hàng loạt theo tệp\n'
    )
    parser = argparse.ArgumentParser(
        prog="safevichi",
        description="SafeViChi v2 — Phát hiện tin nhắn tiếng Việt nguy hại chạy cục bộ (ONNX Hybrid INT8).",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Tin nhắn hoặc hội thoại cần kiểm tra (dùng '||' để ngăn các lượt hội thoại 1–8 lượt).",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Mở chế độ gõ tương tác liên tục.")
    parser.add_argument("-c", "--chat", action="store_true", help="Mở chế độ chat cửa sổ trượt 1–8 lượt.")
    parser.add_argument("--file", type=Path, default=None, help="Quét tệp .txt (mỗi dòng một tin nhắn/hội thoại).")
    parser.add_argument("--jsonl", type=Path, default=None, help="Quét tệp .jsonl (đọc trường 'text').")
    parser.add_argument("-o", "--out", type=Path, default=None, help="Tệp JSONL đầu ra cho chế độ --file / --jsonl.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Xuất kết quả dạng JSON thuần.")
    parser.add_argument(
        "--legacy-normalize",
        action="store_true",
        help=(
            "[Lỗi thời / Deprecated] Bật bộ chuẩn hóa 5 bước kế thừa từ SafeViChi v1 trước khi suy luận. "
            "Không khuyến nghị: sản phẩm v2 chạy trực tiếp trên văn bản gốc và bộ chuẩn hóa v1 có thể "
            "làm giảm độ chính xác trên câu sạch (C0) và điểm mù (C2)."
        ),
    )
    parser.add_argument("--fp32", action="store_true", help="Dùng ONNX FP32 thay vì mặc định ONNX Hybrid INT8.")
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help=f"Thư mục chứa tệp ONNX và crf_params.json (mặc định: {DEFAULT_ONNX_DIR}; tự tải từ Hugging Face nếu chưa có).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Chủ động tải gói mô hình từ Hugging Face về thư mục --onnx-dir nếu còn thiếu.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Tắt tự động tải mô hình từ Hugging Face (chế độ offline; hoặc đặt SAFEVICHI_OFFLINE=1).",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help=f"Kho Hugging Face để tải mô hình (mặc định: {DEFAULT_HF_REPO_ID}).",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=DEFAULT_HF_REVISION,
        help=f"Nhánh/commit trên Hugging Face (mặc định: {DEFAULT_HF_REVISION}).",
    )
    parser.add_argument("--threads", type=int, default=None, help="Số luồng CPU cho ONNX Runtime.")
    parser.add_argument("--no-color", action="store_true", help="Tắt tô màu ANSI (hoặc đặt biến NO_COLOR=1).")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    _init_windows_utf8()

    args = build_parser().parse_args(argv)
    use_color = (not args.no_color) and (not args.as_json) and _enable_windows_ansi()

    if args.legacy_normalize:
        print(f"Cảnh báo [Lỗi thời]: --legacy-normalize — {LEGACY_NORM_NOTICE}", file=sys.stderr)

    explicit_onnx_dir = args.onnx_dir is not None
    onnx_dir = args.onnx_dir if explicit_onnx_dir else DEFAULT_ONNX_DIR
    offline = args.no_download or os.environ.get("SAFEVICHI_OFFLINE", "").strip().lower() in {"1", "true", "yes"}
    should_auto_download = (not offline) and (args.download or not explicit_onnx_dir)

    try:
        if args.download:
            if offline:
                raise ValueError("Không thể dùng đồng thời --download và --no-download / SAFEVICHI_OFFLINE=1.")
            ensure_onnx_model(
                onnx_dir,
                use_fp32=args.fp32,
                repo_id=args.hf_repo,
                revision=args.hf_revision,
            )
            has_work = bool(
                args.jsonl is not None
                or args.file is not None
                or args.interactive
                or args.chat
                or " ".join(args.text).strip()
            )
            if not has_work:
                return 0

        predictor = SafeViChiPredictor(
            onnx_dir=onnx_dir,
            use_fp32=args.fp32,
            threads=args.threads,
            auto_download=should_auto_download,
            hf_repo_id=args.hf_repo,
            hf_revision=args.hf_revision,
        )

        if args.jsonl is not None:
            return _run_batch(
                predictor, args.jsonl, is_jsonl=True, out_path=args.out, legacy_normalize=args.legacy_normalize
            )
        if args.file is not None:
            return _run_batch(
                predictor, args.file, is_jsonl=False, out_path=args.out, legacy_normalize=args.legacy_normalize
            )
        if args.interactive or args.chat:
            return _run_interactive(
                predictor,
                chat_mode=args.chat,
                legacy_normalize=args.legacy_normalize,
                as_json=args.as_json,
                use_color=use_color,
            )

        raw_text = " ".join(args.text).strip()
        if not raw_text and not sys.stdin.isatty():
            raw_text = sys.stdin.read().strip()

        if not raw_text:
            return _run_interactive(
                predictor,
                chat_mode=False,
                legacy_normalize=args.legacy_normalize,
                as_json=args.as_json,
                use_color=use_color,
            )

        res = predictor.predict(raw_text, legacy_normalize=args.legacy_normalize)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(render_pretty(res, use_color=use_color))
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError, RuntimeError, OSError) as exc:
        prefix = f"{ANSI_RED}{ANSI_BOLD}Lỗi:{ANSI_RESET} " if use_color else "Lỗi: "
        print(f"{prefix}{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
