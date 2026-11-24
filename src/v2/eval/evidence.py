"""Độ đo bằng chứng SafeViChi v2: F1 theo ý (docs/v2/DO_DO_BANG_CHUNG.md) song song F1 mức từ.

Nguyên tắc (tách riêng hoàn toàn khỏi phân loại nhãn):
- Chỉ tính F1 trên các đoạn nguy hại có cụm bằng chứng người gán.
- Đoạn AN_TOAN không tính F1 mà đo riêng tỉ lệ bị tô oan (safe_false_highlight_rate).
- Báo song song F1 theo ý (chấm trúng từ mang ý trong [ ] và vùng cho phép ±2 từ trong cùng mệnh đề)
  cạnh F1 mức từ (khớp từng từ chặt), kèm tỉ lệ cụm không có nguyên văn trong câu (non-verbatim),
  tỉ lệ phình độ dài (word_length_ratio), và tỉ lệ câu nguy hại nhưng trắng bằng chứng.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

SAFE_LABEL = "AN_TOAN"

# Dấu ngắt mệnh đề theo mục 3 của docs/v2/DO_DO_BANG_CHUNG.md (, . ! ? ; : …),
# kể cả khi đứng trước ngoặc kép/ngoặc đơn đóng hoặc emoticon cuối từ, nhưng không tách giữa từ (vd. "thế.u").
_CLAUSE_END_RE = re.compile(r"(?:[,!?;:…]|\.)(?=(?:[\"”’')\]}>]*)(?:$|\s|(?![a-zA-ZÀ-ỹ0-9])))")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True)
class WordToken:
    index: int
    start: int
    end: int
    line: int
    clause: int
    text: str


def tokenize_words_and_clauses(text: str) -> list[WordToken]:
    """Tách văn bản thành từ theo dấu cách, gắn chỉ số dòng và mệnh đề cho từng từ."""
    tokens: list[WordToken] = []
    offset = 0
    clause_id = 0
    lines = text.split("\n")
    for line_idx, line in enumerate(lines):
        if line_idx > 0:
            clause_id += 1
        for m in re.finditer(r"\S+", line):
            w_text = m.group(0)
            tokens.append(
                WordToken(
                    index=len(tokens),
                    start=offset + m.start(),
                    end=offset + m.end(),
                    line=line_idx,
                    clause=clause_id,
                    text=w_text,
                )
            )
            if _CLAUSE_END_RE.search(w_text):
                clause_id += 1
        offset += len(line) + 1
    return tokens


def _parse_gold_phrase(raw: str | dict[str, Any]) -> tuple[str, list[tuple[int, int]], int | None]:
    """Tách cụm người gán có thể chứa ngoặc [từ mang ý] thành:
    (cụm_sạch_không_ngoặc, danh_sách_vị_trí_ký_tự_từ_mang_ý, dòng_chỉ_định_nếu_có).
    """
    line_filter: int | None = None
    if isinstance(raw, dict):
        phrase = str(raw.get("text", ""))
        if raw.get("line") is not None:
            line_filter = int(raw["line"])
    else:
        phrase = str(raw)

    clean_parts: list[str] = []
    core_spans: list[tuple[int, int]] = []
    cursor = 0
    clean_len = 0
    for m in _BRACKET_RE.finditer(phrase):
        before = phrase[cursor : m.start()]
        clean_parts.append(before)
        clean_len += len(before)
        inner = m.group(1)
        core_spans.append((clean_len, clean_len + len(inner)))
        clean_parts.append(inner)
        clean_len += len(inner)
        cursor = m.end()
    clean_parts.append(phrase[cursor:])
    return "".join(clean_parts), core_spans, line_filter


def _char_span_to_word_indices(
    tokens: Sequence[WordToken],
    start: int,
    end: int,
    line_filter: int | None = None,
) -> set[int]:
    out: set[int] = set()
    for tok in tokens:
        if line_filter is not None and tok.line != line_filter:
            continue
        if tok.start < end and start < tok.end:
            out.add(tok.index)
    return out


def _find_all_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    spans: list[tuple[int, int]] = []
    pos = 0
    step = len(phrase)
    while True:
        idx = text.find(phrase, pos)
        if idx < 0:
            break
        spans.append((idx, idx + step))
        pos = idx + 1
    return spans


def find_pred_word_indices(
    text: str,
    tokens: Sequence[WordToken],
    pred_phrases: Sequence[str],
) -> tuple[set[int], int, int]:
    """Khớp nguyên văn từng cụm mô hình trả về (xuất hiện nhiều lần thì tô hết).

    Trả về: (tập_chỉ_số_từ_được_tô, số_cụm_hợp_lệ, số_cụm_không_có_trong_câu).
    """
    pred_words: set[int] = set()
    total_phrases = 0
    non_verbatim = 0
    for raw in pred_phrases:
        phrase = str(raw).strip()
        if not phrase:
            continue
        total_phrases += 1
        hits = _find_all_occurrences(text, phrase)
        if not hits:
            non_verbatim += 1
            continue
        for a, b in hits:
            pred_words |= _char_span_to_word_indices(tokens, a, b)
    return pred_words, total_phrases, non_verbatim


def score_evidence_item(
    text: str,
    gold_phrases: Sequence[str | dict[str, Any]],
    pred_phrases: Sequence[str],
    margin: int = 2,
) -> dict[str, Any]:
    """Chấm song song F1 theo ý và F1 mức từ cho một đoạn hội thoại."""
    tokens = tokenize_words_and_clauses(text)
    pred_words, total_pred_phrases, non_verbatim = find_pred_word_indices(text, tokens, pred_phrases)

    gold_union: set[int] = set()
    allowed_union: set[int] = set()
    phrase_recalls: list[float] = []

    for raw in gold_phrases:
        clean_phrase, core_char_spans, line_filter = _parse_gold_phrase(raw)
        clean_phrase = clean_phrase.strip()
        if not clean_phrase:
            continue
        hits = _find_all_occurrences(text, clean_phrase)
        if line_filter is not None:
            filtered_hits: list[tuple[int, int]] = []
            for a, b in hits:
                if _char_span_to_word_indices(tokens, a, b, line_filter=line_filter):
                    filtered_hits.append((a, b))
            hits = filtered_hits
        if not hits:
            continue

        # Chọn lần khớp đầu tiên nếu người gán không chỉ định dòng (khớp hành vi thực tế từng cụm)
        best_recall = 0.0
        for a, b in hits:
            span_words = _char_span_to_word_indices(tokens, a, b, line_filter=line_filter)
            if not span_words:
                continue
            gold_union |= span_words

            # Vùng cho phép: cụm người gán + tối đa `margin` từ mỗi bên trong CÙNG mệnh đề và CÙNG dòng
            min_idx = min(span_words)
            max_idx = max(span_words)
            left_tok = tokens[min_idx]
            right_tok = tokens[max_idx]
            allowed_union |= span_words
            for d in range(1, margin + 1):
                li = min_idx - d
                if li >= 0 and tokens[li].line == left_tok.line and tokens[li].clause == left_tok.clause:
                    allowed_union.add(li)
                else:
                    break
            for d in range(1, margin + 1):
                ri = max_idx + d
                if ri < len(tokens) and tokens[ri].line == right_tok.line and tokens[ri].clause == right_tok.clause:
                    allowed_union.add(ri)
                else:
                    break

            # Từ mang ý trong [ ]; nếu không có ngoặc thì mọi từ trong cụm tính như nhau
            core_words: set[int] = set()
            if core_char_spans:
                for ca, cb in core_char_spans:
                    core_words |= _char_span_to_word_indices(tokens, a + ca, a + cb, line_filter=line_filter)
            if not core_words:
                core_words = span_words
            rec = len(core_words & pred_words) / len(core_words)
            if rec > best_recall:
                best_recall = rec
        phrase_recalls.append(best_recall)

    # F1 theo ý
    if phrase_recalls:
        r_intent = sum(phrase_recalls) / len(phrase_recalls)
    else:
        r_intent = 0.0
    p_intent = (len(pred_words & allowed_union) / len(pred_words)) if pred_words else 0.0
    f1_intent = (2 * p_intent * r_intent / (p_intent + r_intent)) if (p_intent + r_intent) > 0 else 0.0

    # F1 mức từ (chặt)
    overlap = len(pred_words & gold_union)
    p_word = (overlap / len(pred_words)) if pred_words else 0.0
    r_word = (overlap / len(gold_union)) if gold_union else 0.0
    f1_word = (2 * p_word * r_word / (p_word + r_word)) if (p_word + r_word) > 0 else 0.0

    return {
        "f1_theo_y": {"precision": p_intent, "recall": r_intent, "f1": f1_intent},
        "f1_muc_tu": {"precision": p_word, "recall": r_word, "f1": f1_word},
        "n_gold_words": len(gold_union),
        "n_pred_words": len(pred_words),
        "n_pred_phrases": total_pred_phrases,
        "n_non_verbatim_phrases": non_verbatim,
    }


def evaluate_evidence(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    margin: int = 2,
) -> dict[str, Any]:
    """Tổng hợp toàn bộ chỉ số bằng chứng trên một tập dữ liệu."""
    if len(records) != len(predictions):
        raise ValueError(f"Số mẫu gold ({len(records)}) khác số mẫu dự đoán ({len(predictions)})")

    intent_p: list[float] = []
    intent_r: list[float] = []
    intent_f1: list[float] = []
    word_p: list[float] = []
    word_r: list[float] = []
    word_f1: list[float] = []

    n_safe = 0
    n_safe_highlighted = 0
    n_harmful_with_gold = 0
    n_harmful_empty_pred = 0
    total_gold_words = 0
    total_pred_words_on_gold = 0
    total_pred_phrases = 0
    total_non_verbatim = 0

    for rec, pred in zip(records, predictions):
        text = str(rec["text"])
        gold_label = str(rec["nhan"])
        gold_ev = rec.get("bang_chung") or []
        pred_ev = pred.get("bang_chung") or []
        if not isinstance(pred_ev, list):
            pred_ev = []

        item = score_evidence_item(text, gold_ev, pred_ev, margin=margin)
        total_pred_phrases += item["n_pred_phrases"]
        total_non_verbatim += item["n_non_verbatim_phrases"]

        if gold_label == SAFE_LABEL:
            n_safe += 1
            if item["n_pred_phrases"] > 0:
                n_safe_highlighted += 1
            continue

        if item["n_gold_words"] > 0:
            n_harmful_with_gold += 1
            intent_p.append(item["f1_theo_y"]["precision"])
            intent_r.append(item["f1_theo_y"]["recall"])
            intent_f1.append(item["f1_theo_y"]["f1"])
            word_p.append(item["f1_muc_tu"]["precision"])
            word_r.append(item["f1_muc_tu"]["recall"])
            word_f1.append(item["f1_muc_tu"]["f1"])
            total_gold_words += item["n_gold_words"]
            total_pred_words_on_gold += item["n_pred_words"]
            if item["n_pred_words"] == 0:
                n_harmful_empty_pred += 1

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_harmful_evaluated": n_harmful_with_gold,
        "n_safe_evaluated": n_safe,
        "margin_words": margin,
        "f1_theo_y": {
            "precision": _mean(intent_p),
            "recall": _mean(intent_r),
            "f1": _mean(intent_f1),
        },
        "f1_muc_tu": {
            "precision": _mean(word_p),
            "recall": _mean(word_r),
            "f1": _mean(word_f1),
        },
        "safe_false_highlight_rate": round(n_safe_highlighted / n_safe, 4) if n_safe else 0.0,
        "harmful_empty_evidence_rate": (
            round(n_harmful_empty_pred / n_harmful_with_gold, 4) if n_harmful_with_gold else 0.0
        ),
        "word_length_ratio": (
            round(total_pred_words_on_gold / total_gold_words, 4) if total_gold_words else 0.0
        ),
        "non_verbatim_phrase_rate": (
            round(total_non_verbatim / total_pred_phrases, 4) if total_pred_phrases else 0.0
        ),
        "total_pred_phrases": total_pred_phrases,
        "total_non_verbatim_phrases": total_non_verbatim,
    }
