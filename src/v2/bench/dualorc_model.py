"""Kiến trúc SafeVi-DualORC-CRF (Schema v3), tiền xử lý và giải mã bằng chứng thuần NumPy.

Mô-đun tái dựng chính xác 100% kiến trúc và quy trình tiền xử lý / hậu xử lý
trong `notebooks/v2_dualorc_crf_colab.ipynb` để:
1. Nạp trọng số `model_best.safetensors` cục bộ trên CPU mà không cần tải trọng số gốc từ Hugging Face.
2. Xuất đồ thị ONNX (FP32 và INT8 nén cả bảng từ vựng 256k lẫn các lớp Transformer).
3. Suy luận và giải mã bằng chứng thuần NumPy (Forward-Backward BIO-CRF & Viterbi) trên ONNX Runtime.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS_DEFAULT = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
MAX_WORD_BYTES_DEFAULT = 24
MAX_SUBWORDS_PER_WORD_DEFAULT = 12


# ==============================================================================
# 1. KIẾN TRÚC PYTORCH: TurnAwareBIOCRF, WordByteCNN, SafeViDualORCCRF, PerRowInt8Embedding
# ==============================================================================


class TurnAwareBIOCRF(nn.Module):
    """Linear-Chain BIO-CRF 3 trạng thái [0=O, 1=B-EV, 2=I-EV] có ràng buộc biên lượt thoại."""

    NEG_INF = -1e4

    def __init__(self) -> None:
        super().__init__()
        self.start_trans = nn.Parameter(torch.tensor([0.2, 0.0, -5.0]))
        self.end_trans = nn.Parameter(torch.zeros(3))
        self.trans = nn.Parameter(
            torch.tensor(
                [
                    [0.5, 0.0, -5.0],
                    [-0.2, -0.3, 0.6],
                    [0.0, -0.4, 0.6],
                ]
            )
        )

    def _effective_trans(self) -> torch.Tensor:
        t = self.trans.clone()
        t[0, 2] = self.NEG_INF
        return t

    def _effective_start(self) -> torch.Tensor:
        s = self.start_trans.clone()
        s[2] = self.NEG_INF
        return s

    def _step_trans(self, trans: torch.Tensor, is_turn_start_t: torch.Tensor) -> torch.Tensor:
        batch_size = is_turn_start_t.size(0)
        step = trans.unsqueeze(0).expand(batch_size, 3, 3).clone()
        if is_turn_start_t.any():
            step[is_turn_start_t, :, 2] = self.NEG_INF
        return step

    def forward_alpha(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        turn_start: torch.Tensor,
        allowed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, width, _ = emissions.shape
        trans = self._effective_trans()
        start = self._effective_start()
        emit = emissions
        if allowed is not None:
            emit = emit.masked_fill(~allowed, self.NEG_INF)

        alpha = start.unsqueeze(0) + emit[:, 0]
        alphas = [alpha]
        for t in range(1, width):
            m_t = mask[:, t].unsqueeze(-1)
            step_tr = self._step_trans(trans, turn_start[:, t])
            nxt = torch.logsumexp(alpha.unsqueeze(2) + step_tr, dim=1) + emit[:, t]
            alpha = torch.where(m_t, nxt, alpha)
            alphas.append(alpha)

        log_z = torch.logsumexp(alpha + self.end_trans.unsqueeze(0), dim=-1)
        return log_z, torch.stack(alphas, dim=1)

    def marginals(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        turn_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, width, _ = emissions.shape
        trans = self._effective_trans()
        log_z, alphas = self.forward_alpha(emissions, mask, turn_start, allowed=None)

        beta = self.end_trans.unsqueeze(0).expand(batch_size, 3).clone()
        betas: list[torch.Tensor | None] = [None] * width
        betas[width - 1] = beta
        for t in range(width - 2, -1, -1):
            m_next = mask[:, t + 1].unsqueeze(-1)
            step_tr = self._step_trans(trans, turn_start[:, t + 1])
            nxt_beta = torch.logsumexp(step_tr + (emissions[:, t + 1] + beta).unsqueeze(1), dim=2)
            beta = torch.where(m_next, nxt_beta, beta)
            betas[t] = beta

        betas_t = torch.stack([b for b in betas if b is not None], dim=1)
        log_marg = alphas + betas_t - log_z[:, None, None]
        probs = log_marg.exp()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs * mask.unsqueeze(-1)
        return probs, log_z


class WordByteCNN(nn.Module):
    """Luồng UTF-8 Byte-CNN đa tỷ lệ (k=2, 3, 4) ở mức từ."""

    def __init__(self, out_dim: int, emb_dim: int = 32, filters: int = 48) -> None:
        super().__init__()
        self.byte_emb = nn.Embedding(257, emb_dim, padding_idx=0)
        self.conv2 = nn.Conv1d(emb_dim, filters, kernel_size=2)
        self.conv3 = nn.Conv1d(emb_dim, filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(emb_dim, filters, kernel_size=4)
        self.proj = nn.Linear(filters * 3, out_dim)

    def forward(self, word_bytes: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        batch_size, width, seq_len = word_bytes.shape
        flat_b = word_bytes.view(batch_size * width, seq_len)
        valid_b = flat_b != 0
        x = self.byte_emb(flat_b).transpose(1, 2)
        x = x * valid_b.unsqueeze(1).to(x.dtype)

        def pool_conv(conv: nn.Conv1d, x_in: torch.Tensor, pad_right: int = 0, pad_left: int = 0) -> torch.Tensor:
            if pad_left or pad_right:
                x_in = F.pad(x_in, (pad_left, pad_right))
            y = F.gelu(conv(x_in))
            y = y.masked_fill(~valid_b.unsqueeze(1), -1e4)
            pooled = y.max(dim=-1).values
            has_any = valid_b.any(dim=-1, keepdim=True)
            return torch.where(has_any, pooled, torch.zeros_like(pooled))

        f2 = pool_conv(self.conv2, x, pad_right=1)
        f3 = pool_conv(self.conv3, x)
        f4 = pool_conv(self.conv4, x, pad_left=1, pad_right=2)
        feat = self.proj(torch.cat([f2, f3, f4], dim=-1)).view(batch_size, width, -1)
        return feat * word_mask.unsqueeze(-1).to(feat.dtype)


def _mha_forward_onnx_safe(
    mha: nn.MultiheadAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-Head Attention tương đương toán học với nn.MultiheadAttention (eval), giữ toàn bộ chiều động khi xuất ONNX."""
    d = mha.embed_dim
    h = mha.num_heads
    hd = d // h
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)

    bsz = q.size(0)
    tq = q.size(1)
    tk = k.size(1)
    q_h = F.linear(q, w_q, b_q).reshape(bsz, tq, h, hd).transpose(1, 2)
    k_h = F.linear(k, w_k, b_k).reshape(bsz, tk, h, hd).transpose(1, 2)
    v_h = F.linear(v, w_v, b_v).reshape(bsz, tk, h, hd).transpose(1, 2)

    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(hd)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = F.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v_h).transpose(1, 2).reshape(bsz, tq, d)
    return F.linear(ctx, mha.out_proj.weight, mha.out_proj.bias)


class SafeViDualORCCRF(nn.Module):
    """Kiến trúc SafeVi-DualORC-CRF (Schema v3) tham số hóa từ config.json."""

    def __init__(self, backbone: nn.Module, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        d_model = int(backbone.config.hidden_size)
        width = int(cfg.get("width", 256))
        num_labels = len(cfg.get("labels", LABELS_DEFAULT))

        self.d = width
        self.use_byte_cnn = bool(cfg.get("use_byte_cnn", True))
        self.use_hat_zoom = bool(cfg.get("use_hat_zoom", True))
        self.use_bio_crf = bool(cfg.get("use_bio_crf", True))
        self.use_orc = bool(cfg.get("use_orc", True))
        self.proto_weight = float(cfg.get("proto_weight", 0.08))
        self.M = int(cfg.get("num_subcenters", 3))

        self.in_norm = nn.LayerNorm(d_model)
        self.sub_attn = nn.Linear(d_model, 1)
        self.sub_proj = nn.Linear(d_model, width)

        if self.use_byte_cnn:
            self.byte_cnn = WordByteCNN(width)
            self.char_gate = nn.Linear(width * 2, width)
        self.fuse_norm = nn.LayerNorm(width)

        self.emb_parity = nn.Embedding(2, width)
        self.emb_turn = nn.Embedding(8, width)
        self.emb_recency = nn.Embedding(8, width)

        self.word_turn_attn = nn.Linear(width, 1)
        if self.use_hat_zoom:
            self.turn_self_attn = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.1)
            self.t_norm1 = nn.LayerNorm(width)
            self.turn_word_xattn = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.1)
            self.t_norm2 = nn.LayerNorm(width)
            self.turn_ffn = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(width * 2, width),
            )
            self.t_norm3 = nn.LayerNorm(width)

        self.span_conv3 = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width, bias=False)
        self.span_conv5 = nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width, bias=False)
        self.span_mix = nn.Linear(width * 2, width)
        self.span_norm = nn.LayerNorm(width)
        if self.use_bio_crf:
            self.bio_head = nn.Linear(width, 3)
            self.crf = TurnAwareBIOCRF()
        else:
            self.bin_span_head = nn.Linear(width, 1)

        self.ctx_attn = nn.Linear(width, 1)
        self.ev_norm = nn.LayerNorm(width)
        self.harm_head = nn.Sequential(
            nn.Linear(width * 2, width // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(width * 2, width // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width // 2, 1),
        )

        self.queries = nn.Parameter(torch.randn(num_labels, self.M, width) / math.sqrt(width))
        self.class_vectors = nn.Parameter(torch.randn(num_labels, self.M, width) / math.sqrt(width))
        self.sub_bias = nn.Parameter(torch.zeros(num_labels, self.M))

        if self.use_orc:
            self.orc_cls_head = nn.Sequential(
                nn.Linear(width * 2 + 2, width),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(width, num_labels),
            )
            self.orc_blend_logit = nn.Parameter(torch.tensor(math.log(0.30 / 0.70)))

        if self.proto_weight > 0:
            self.prototypes = nn.Parameter(torch.randn(num_labels, self.M, width) / math.sqrt(width))
        self.drop = nn.Dropout(0.1)

    def forward(self, b: dict[str, torch.Tensor], with_aux: bool = False) -> dict[str, torch.Tensor | None]:
        hidden = self.backbone(input_ids=b["ids"], attention_mask=b["attn_mask"].long()).last_hidden_state

        h_tok = self.in_norm(hidden.float())
        batch_size, num_words, num_sub = b["word_sub_idx"].shape
        w_mask = b["word_mask"]
        sub_mask = b["word_sub_mask"]

        flat_idx = b["word_sub_idx"].view(batch_size, num_words * num_sub)
        gathered = h_tok.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, h_tok.size(-1))).view(
            batch_size, num_words, num_sub, -1
        )
        sm_f = sub_mask.unsqueeze(-1).float()
        sub_cnt = sm_f.sum(dim=2).clamp_min(1.0)
        sub_mean = (gathered * sm_f).sum(dim=2) / sub_cnt
        s_score = self.sub_attn(gathered).squeeze(-1).clamp(-20.0, 20.0).masked_fill(~sub_mask, -1e4)
        s_w = F.softmax(s_score, dim=-1) * sub_mask.float()
        s_w = s_w / s_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        sub_att = (gathered * s_w.unsqueeze(-1)).sum(dim=2)
        has_sub = sub_mask.any(dim=-1, keepdim=True).float()
        u_sub = self.sub_proj(0.5 * sub_mean + 0.5 * sub_att) * has_sub

        if self.use_byte_cnn:
            u_char = self.byte_cnn(b["word_bytes"], w_mask)
            gate = torch.sigmoid(self.char_gate(torch.cat([u_sub, u_char], dim=-1)))
            w_feat = self.fuse_norm(u_sub + gate * u_char)
        else:
            w_feat = self.fuse_norm(u_sub)

        parity = b["word_turn"] % 2
        struct_emb = self.emb_parity(parity) + self.emb_turn(b["word_turn"]) + self.emb_recency(b["word_recency"])
        w_feat = (w_feat + struct_emb) * w_mask.unsqueeze(-1).float()

        turn_ids = torch.arange(8, device=w_feat.device)[None, :, None]
        belongs = (b["word_turn"][:, None, :] == turn_ids) & w_mask[:, None, :]
        t_counts = belongs.sum(dim=-1)
        turn_present = t_counts > 0
        t_mean = torch.einsum("btw,bwd->btd", belongs.float(), w_feat) / t_counts.clamp_min(1).unsqueeze(-1).float()
        wt_score = self.word_turn_attn(w_feat).squeeze(-1).clamp(-20.0, 20.0)
        wt_masked = wt_score[:, None, :].expand(-1, 8, -1).masked_fill(~belongs, -1e4)
        wt_w = F.softmax(wt_masked, dim=-1) * belongs.float()
        wt_w = wt_w / wt_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        t_att = torch.einsum("btw,bwd->btd", wt_w, w_feat)
        turns = (0.5 * t_mean + 0.5 * t_att) * turn_present.unsqueeze(-1).float()

        if self.use_hat_zoom:
            t_self = _mha_forward_onnx_safe(self.turn_self_attn, turns, turns, turns, key_padding_mask=~turn_present)
            turns = self.t_norm1(turns + self.drop(t_self))
            t_zoom = _mha_forward_onnx_safe(self.turn_word_xattn, turns, w_feat, w_feat, key_padding_mask=~w_mask)
            turns = self.t_norm2(turns + self.drop(t_zoom))
            turns = self.t_norm3(turns + self.turn_ffn(turns))
            turns = turns * turn_present.unsqueeze(-1).float()

        t_for_w = turns.gather(1, b["word_turn"].unsqueeze(-1).expand(-1, -1, self.d))
        w_ctx = torch.tanh(w_feat + t_for_w) * w_mask.unsqueeze(-1).float()

        wc_t = w_ctx.transpose(1, 2)
        c3 = self.span_conv3(wc_t).transpose(1, 2) * w_mask.unsqueeze(-1).float()
        c5 = self.span_conv5(wc_t).transpose(1, 2) * w_mask.unsqueeze(-1).float()
        span_feat = (
            self.span_norm(w_ctx + F.gelu(self.span_mix(self.drop(torch.cat([c3, c5], dim=-1)))))
            * w_mask.unsqueeze(-1).float()
        )

        if self.use_bio_crf:
            emissions = self.bio_head(self.drop(span_feat))
            ev_prob = F.softmax(emissions, dim=-1)[:, :, 1:].sum(dim=-1) * w_mask.float()
        else:
            emissions = self.bin_span_head(self.drop(span_feat)).squeeze(-1)
            ev_prob = emissions.sigmoid() * w_mask.float()

        ctx_sc = self.ctx_attn(turns).squeeze(-1).clamp(-20.0, 20.0).masked_fill(~turn_present, -1e4)
        ctx_w = F.softmax(ctx_sc, dim=-1) * turn_present.float()
        z_ctx = (turns * ctx_w.unsqueeze(-1)).sum(dim=1)

        ev_sum = ev_prob.sum(dim=1, keepdim=True)
        z_ev = self.ev_norm((ev_prob.unsqueeze(-1) * span_feat).sum(dim=1) / (ev_sum + 1.0))

        orc_pair = torch.cat([z_ctx, z_ev], dim=-1)
        harm_logit = self.harm_head(self.drop(orc_pair)).squeeze(-1)
        severity = torch.sigmoid(self.severity_head(self.drop(orc_pair)).squeeze(-1))

        mem = torch.cat([turns, z_ev.unsqueeze(1)], dim=1)
        mem_mask = torch.cat([turn_present, torch.ones((batch_size, 1), dtype=torch.bool, device=turns.device)], dim=1)
        q_scores = torch.einsum("btd,lmd->blmt", mem, self.queries) / math.sqrt(self.d)
        q_att = q_scores.masked_fill(~mem_mask[:, None, None, :], -1e4).softmax(dim=-1)
        q_pooled = torch.einsum("blmt,btd->blmd", q_att, mem)
        sub_logits = (self.drop(q_pooled) * self.class_vectors).sum(dim=-1) + self.sub_bias
        logits_direct = torch.logsumexp(sub_logits, dim=-1) - math.log(self.M)

        if self.use_orc:
            orc_in = torch.cat([z_ctx, z_ev, harm_logit.sigmoid().unsqueeze(-1), severity.unsqueeze(-1)], dim=-1)
            logits_orc = self.orc_cls_head(self.drop(orc_in))
            beta = torch.sigmoid(self.orc_blend_logit)
            logits = (1.0 - beta) * logits_direct + beta * logits_orc
        else:
            logits = logits_direct

        proto_logits = None
        if with_aux and self.proto_weight > 0:
            doc_n = F.normalize(z_ctx, dim=-1)
            p_n = F.normalize(self.prototypes, dim=-1)
            sim = torch.einsum("bd,lmd->blm", doc_n, p_n) / 0.15
            proto_logits = sim.max(dim=-1).values

        return {
            "logits": logits.float(),
            "emissions": emissions.float(),
            "ev_prob": ev_prob.float(),
            "harm_logit": harm_logit.float(),
            "severity": severity.float(),
            "proto": proto_logits,
        }


class PerRowInt8Embedding(nn.Module):
    """Lượng tử hóa đối xứng INT8 theo từng hàng từ vựng (nén 4×, scale FP32, tương thích ONNX opset 17)."""

    def __init__(self, emb: nn.Embedding) -> None:
        super().__init__()
        w = emb.weight.detach().float().cpu()
        self.num_embeddings = emb.num_embeddings
        self.embedding_dim = emb.embedding_dim
        self.padding_idx = emb.padding_idx
        scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        qweight = (w / scale).round().clamp(-127, 127).to(torch.int8)
        if self.padding_idx is not None:
            qweight[self.padding_idx].zero_()
            scale[self.padding_idx].zero_()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale.to(torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        q = F.embedding(input_ids, self.qweight).float()
        s = F.embedding(input_ids, self.scale)
        return q * s


def quantize_embeddings_int8(module: nn.Module, min_num_embeddings: int = 4096) -> int:
    """Thay thế đệ quy mọi nn.Embedding lớn (>= min_num_embeddings) bằng PerRowInt8Embedding."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Embedding) and child.num_embeddings >= min_num_embeddings:
            setattr(module, name, PerRowInt8Embedding(child))
            replaced += 1
        else:
            replaced += quantize_embeddings_int8(child, min_num_embeddings=min_num_embeddings)
    return replaced


def load_dualorc_pytorch(
    weights_path: Path,
    config_path: Path,
    tokenizer_cache_dir: Path | None = None,
    attn_implementation: str = "eager",
) -> tuple[SafeViDualORCCRF, Any, dict[str, Any]]:
    """Khởi tạo SafeViDualORCCRF từ config HuggingFace (không tải trọng số gốc) và nạp model_best.safetensors.

    Dùng `attn_implementation="eager"` mặc định để đảm bảo đồ thị ONNX opset 17 xuất ra sạch,
    không phụ thuộc toán tử SDPA mới và tương thích cả ONNX Runtime CPU lẫn ONNX Runtime Web.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    backbone_id = cfg["backbone"]
    backbone_rev = cfg.get("backbone_rev")

    local_tok = tokenizer_cache_dir if (tokenizer_cache_dir and (tokenizer_cache_dir / "tokenizer.json").exists()) else None
    local_cfg = tokenizer_cache_dir if (tokenizer_cache_dir and (tokenizer_cache_dir / "backbone_config.json").exists()) else None

    if local_cfg is not None:
        bb_cfg = AutoConfig.from_pretrained(str(local_cfg / "backbone_config.json"))
    else:
        bb_cfg = AutoConfig.from_pretrained(backbone_id, revision=backbone_rev)
        if tokenizer_cache_dir is not None:
            tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
            (tokenizer_cache_dir / "backbone_config.json").write_text(bb_cfg.to_json_string(), encoding="utf-8")

    if hasattr(bb_cfg, "reference_compile"):
        bb_cfg.reference_compile = False
    if hasattr(bb_cfg, "use_cache"):
        bb_cfg.use_cache = False
    if getattr(bb_cfg, "model_type", "") == "modernbert":
        bb_cfg._attn_implementation = attn_implementation

    if local_tok is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(local_tok), use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone_id, revision=backbone_rev, use_fast=True)
        if tokenizer_cache_dir is not None:
            tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(tokenizer_cache_dir))

    model_kwargs: dict[str, Any] = {}
    if getattr(bb_cfg, "model_type", "") in {"xlm-roberta", "roberta", "bert"}:
        model_kwargs["add_pooling_layer"] = False
    if getattr(bb_cfg, "model_type", "") == "modernbert":
        model_kwargs["attn_implementation"] = attn_implementation
    backbone = AutoModel.from_config(bb_cfg, **model_kwargs)

    net = SafeViDualORCCRF(backbone, cfg)
    state_dict = load_file(str(weights_path), device="cpu")
    net.load_state_dict(state_dict, strict=True)
    net.float().eval()
    return net, tokenizer, cfg


# ==============================================================================
# 2. TIỀN XỬ LÝ HỘI THOẠI (PREPARE & COLLATE — TƯƠNG THÍCH CẢ PYTORCH VÀ NUMPY/ONNX)
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
        next_active = []
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


class DualORCPreprocessor:
    """Bộ tiền xử lý hội thoại 1–8 lượt trùng khớp từng token với quá trình huấn luyện."""

    def __init__(
        self,
        tokenizer: Any,
        max_tokens: int = 512,
        max_word_bytes: int = MAX_WORD_BYTES_DEFAULT,
        max_subwords_per_word: int = MAX_SUBWORDS_PER_WORD_DEFAULT,
        labels: tuple[str, ...] = LABELS_DEFAULT,
    ) -> None:
        self.tok = tokenizer
        self.max_tokens = int(max_tokens)
        self.max_word_bytes = int(max_word_bytes)
        self.max_subwords_per_word = int(max_subwords_per_word)
        self.labels = tuple(labels)

        self.bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if self.bos_id is None or self.sep_id is None or self.pad_id is None:
            raise ValueError("Tokenizer thiếu BOS/SEP/PAD token ID.")

    def prepare(self, row: dict[str, Any]) -> dict[str, Any]:
        text = str(row["text"])
        lines = text.split("\n")
        n_turns = len(lines)
        if not (1 <= n_turns <= 8):
            raise ValueError(f"Số lượt ngoài khoảng 1..8: {row.get('id', '?')} ({n_turns})")

        words_meta: list[tuple[int, int, int, int, bool, list[int]]] = []
        cursor = 0
        for t_idx, line in enumerate(lines):
            recency = n_turns - 1 - t_idx
            first_in_turn = True
            for m in re.finditer(r"\S+", line):
                a, b = cursor + m.start(), cursor + m.end()
                raw_w = m.group(0)
                b_arr = [x + 1 for x in raw_w.encode("utf-8")[: self.max_word_bytes]] or [1]
                words_meta.append((a, b, t_idx, recency, first_in_turn, b_arr))
                first_in_turn = False
            cursor += len(line) + 1

        if not words_meta:
            raise ValueError(f"Văn bản không có từ: {row.get('id', '?')}")

        turn_enc: list[tuple[list[int], list[tuple[int, int]]]] = []
        cursor = 0
        for line in lines:
            enc = self.tok(line, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            t_ids = enc["input_ids"]
            t_offs: list[tuple[int, int]] = []
            for sa, sb in enc["offset_mapping"]:
                while sa < sb and sb < len(line) and unicodedata.combining(line[sb]):
                    sb += 1
                t_offs.append((cursor + sa, cursor + sb))
            turn_enc.append((t_ids, t_offs))
            cursor += len(line) + 1

        budget = max(16, self.max_tokens - (1 + n_turns))
        caps = _allocate_turn_budgets([len(x[0]) for x in turn_enc], budget)
        seq_ids = [int(self.bos_id)]
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
            seq_ids.append(int(self.sep_id))
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

        return {
            "row": row,
            "ids": seq_ids,
            "n_tokens": len(seq_ids),
            "n_words": len(words_meta),
            "words_meta": words_meta,
            "word_sub_idxs": word_sub_idxs,
        }

    def collate_numpy(self, examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        batch_size = len(examples)
        max_l = max(e["n_tokens"] for e in examples)
        max_w = max(e["n_words"] for e in examples)
        max_s = max(1, max((len(s) for e in examples for s in e["word_sub_idxs"]), default=1))

        ids = np.full((batch_size, max_l), self.pad_id, dtype=np.int64)
        attn_mask = np.zeros((batch_size, max_l), dtype=np.bool_)
        word_sub_idx = np.zeros((batch_size, max_w, max_s), dtype=np.int64)
        word_sub_mask = np.zeros((batch_size, max_w, max_s), dtype=np.bool_)
        word_mask = np.zeros((batch_size, max_w), dtype=np.bool_)
        word_turn = np.zeros((batch_size, max_w), dtype=np.int64)
        word_recency = np.zeros((batch_size, max_w), dtype=np.int64)
        word_bytes = np.zeros((batch_size, max_w, self.max_word_bytes), dtype=np.int64)

        for i, e in enumerate(examples):
            n_tok = e["n_tokens"]
            ids[i, :n_tok] = e["ids"]
            attn_mask[i, :n_tok] = True
            for w_i, (meta, sub_idxs) in enumerate(zip(e["words_meta"], e["word_sub_idxs"])):
                _, _, t_idx, recency, _, b_arr = meta
                word_mask[i, w_i] = True
                word_turn[i, w_i] = t_idx
                word_recency[i, w_i] = recency
                word_bytes[i, w_i, : len(b_arr)] = b_arr
                if sub_idxs:
                    word_sub_idx[i, w_i, : len(sub_idxs)] = sub_idxs
                    word_sub_mask[i, w_i, : len(sub_idxs)] = True

        return {
            "ids": ids,
            "attn_mask": attn_mask,
            "word_sub_idx": word_sub_idx,
            "word_sub_mask": word_sub_mask,
            "word_mask": word_mask,
            "word_turn": word_turn,
            "word_recency": word_recency,
            "word_bytes": word_bytes,
        }

    def collate_torch(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        np_batch = self.collate_numpy(examples)
        return {k: torch.from_numpy(v) for k, v in np_batch.items()}


# ==============================================================================
# 3. GIẢI MÃ BIO-CRF & HẬU XỬ LÝ BẰNG CHỨNG THUẦN NUMPY (KHÔNG CẦN PYTORCH KHI SUY LUẬN)
# ==============================================================================


def _logsumexp_np(arr: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(arr, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis=axis)


def crf_marginals_single_np(
    emissions: np.ndarray,
    turn_starts: list[bool],
    start_tr: np.ndarray,
    end_tr: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    """Thuật toán Forward-Backward thuần NumPy khớp chính xác TurnAwareBIOCRF.marginals -> (W, 3)."""
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

    log_marg = alphas + betas - log_z
    probs = np.exp(log_marg)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
    return probs


def viterbi_single_np(
    emissions: np.ndarray,
    turn_starts: list[bool],
    start_tr: np.ndarray,
    end_tr: np.ndarray,
    trans: np.ndarray,
    o_bias: float = 0.0,
) -> list[int]:
    """Giải mã Viterbi thuần NumPy cho 1 mẫu."""
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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    ex = np.exp(shifted)
    return ex / np.sum(ex)


def _apply_marginal_postproc(
    words: list[tuple[int, int, float, int, bool]],
    text: str,
    threshold: float = 0.5,
    single_word_threshold: float | None = None,
    bridge_ratio: float = 0.0,
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
    fallback_min_prob: float | None = None,
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


def decode_prediction(
    row: dict[str, Any],
    words_meta: list[tuple[int, int, int, int, bool, list[int]]],
    logits: np.ndarray,
    emissions: np.ndarray,
    severity: float,
    crf_meta: dict[str, Any],
) -> dict[str, Any]:
    """Chuyển đầu ra thô (logits, emissions, severity) thành bản ghi JSON chuẩn SafeViChi v2."""
    text = str(row["text"])
    labels = tuple(crf_meta.get("labels", LABELS_DEFAULT))
    temp = float(crf_meta["temperature"])
    prob = _softmax_np(np.asarray(logits, dtype=np.float64) / temp)
    pred_idx = int(np.argmax(prob))

    nw = len(words_meta)
    turn_starts = [bool(m[4]) for m in words_meta]
    start_tr = np.asarray(crf_meta["start"], dtype=np.float64)
    end_tr = np.asarray(crf_meta["end"], dtype=np.float64)
    trans = np.asarray(crf_meta["trans"], dtype=np.float64)

    marg = crf_marginals_single_np(emissions[:nw], turn_starts, start_tr, end_tr, trans)
    ev_probs = np.clip(marg[:, 1] + marg[:, 2], 0.0, 1.0)
    words = [(m[0], m[1], float(ev_probs[i]), int(m[2]), bool(m[4])) for i, m in enumerate(words_meta)]

    runs: list[list[int]] = []
    if pred_idx != 0 and words:
        mode = str(crf_meta.get("chosen_decoder", "marginal"))
        fb = crf_meta.get("fallback_min_prob")
        if mode == "crf":
            tags = viterbi_single_np(
                emissions[:nw], turn_starts, start_tr, end_tr, trans, o_bias=float(crf_meta.get("o_bias", 0.0))
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
                marks = _apply_empty_fallback([False] * len(words), words, text, fallback_min_prob=float(fb))
                for w, marked in zip(words, marks):
                    if not marked:
                        continue
                    a, b = w[0], w[1]
                    if runs and re.fullmatch(r"[^\S\n]*", text[runs[-1][1] : a]):
                        runs[-1][1] = b
                    else:
                        runs.append([a, b])
        else:
            marks = _apply_marginal_postproc(
                words,
                text,
                threshold=float(crf_meta["marginal_threshold"]),
                single_word_threshold=crf_meta.get("single_word_threshold"),
                bridge_ratio=float(crf_meta.get("bridge_ratio", 0.0)),
            )
            if fb is not None:
                marks = _apply_empty_fallback(marks, words, text, fallback_min_prob=float(fb))
            for w, marked in zip(words, marks):
                if not marked:
                    continue
                a, b = w[0], w[1]
                if runs and re.fullmatch(r"[^\S\n]*", text[runs[-1][1] : a]):
                    runs[-1][1] = b
                else:
                    runs.append([a, b])

    out: dict[str, Any] = {
        "id": row["id"],
        "nhan": labels[pred_idx],
        "tin_cay": float(prob[pred_idx]),
        "probs": [round(float(x), 6) for x in prob],
        "rui_ro": round(float(severity) * 100.0, 2),
        "bang_chung": [text[a:b] for a, b in runs],
        "vi_tri": runs,
    }
    if "nhan" in row:
        out["nhan_dung"] = row["nhan"]
    for meta_k in ("nguon", "cach_gan", "bien_dang", "goc_id", "kieu_bien_dang", "nhom_bien_dang", "phan_test", "giong_vung", "kich_ban", "do_lo", "kenh"):
        if meta_k in row:
            out[meta_k] = row[meta_k]
    return out
