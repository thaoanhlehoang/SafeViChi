// Tiền xử lý hội thoại 1–8 lượt & giải mã BIO-CRF cho SafeVi-DualORC-CRF (Schema v3).
// Khớp trực tiếp với `src/v2/predict.py` và `src/v2/bench/dualorc_model.py`:
// suy luận thẳng trên văn bản gốc (KHÔNG dùng bộ chuẩn hóa v1).

const UTF8 = new TextEncoder();
const INVISIBLE_RE = /[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]/g;
const COMBINING_MARK_RE = /^\p{Mn}$/u;
const NON_NEWLINE_SPACE_RE = /^[^\S\n]*$/u;
const NEG_INF = -1e4;

export function stripInvisibles(text) {
  return text.replace(INVISIBLE_RE, '');
}

/** Phân bổ ngân sách token công bằng giữa các lượt thoại khi tổng vượt max_tokens. */
export function allocateTurnBudgets(turnLens, budget) {
  const nT = turnLens.length;
  const total = turnLens.reduce((a, b) => a + b, 0);
  if (total <= budget) return turnLens.slice();

  const caps = new Array(nT).fill(0);
  let rem = budget;
  let active = Array.from({ length: nT }, (_, i) => i);

  while (active.length > 0 && rem > 0) {
    const share = Math.max(1, Math.floor(rem / active.length));
    const nextActive = [];
    for (const i of active) {
      const need = turnLens[i] - caps[i];
      const take = Math.min(need, share, rem);
      caps[i] += take;
      rem -= take;
      if (caps[i] < turnLens[i]) nextActive.push(i);
    }
    if (nextActive.length === 0 || (nextActive.length === active.length && share === 1 && rem === 0)) {
      break;
    }
    active = nextActive;
  }
  return caps;
}

/** Chuẩn bị toàn bộ 8 tensor đầu vào cho SafeVi-DualORC-CRF từ danh sách 1–8 lượt thoại. */
export function prepareDualORCInput(rawLines, tokenizer, meta) {
  const maxTokens = Number(meta.max_tokens ?? 512);
  const maxWordBytes = Number(meta.max_word_bytes ?? 24);
  const maxSubwordsPerWord = Number(meta.max_subwords_per_word ?? 12);
  const bosId = Number(meta.bos_id);
  const sepId = Number(meta.sep_id);
  const padId = Number(meta.pad_id);

  const lines = rawLines.slice(-8).map((ln) => String(ln ?? ''));
  const text = lines.join('\n');
  const nTurns = lines.length;

  const lineStarts = [];
  const wordsMeta = [];
  let cursor = 0;
  for (let tIdx = 0; tIdx < nTurns; tIdx++) {
    const line = lines[tIdx];
    lineStarts.push(cursor);
    const recency = nTurns - 1 - tIdx;
    let firstInTurn = true;
    for (const m of line.matchAll(/\S+/g)) {
      const a = cursor + m.index;
      const b = a + m[0].length;
      const encoded = UTF8.encode(m[0]);
      const byteLen = Math.min(encoded.length, maxWordBytes);
      const bArr = byteLen > 0 ? new Array(byteLen) : [1];
      for (let bi = 0; bi < byteLen; bi++) bArr[bi] = encoded[bi] + 1;
      wordsMeta.push({ a, b, tIdx, recency, firstInTurn, bArr });
      firstInTurn = false;
    }
    cursor += line.length + 1;
  }

  if (wordsMeta.length === 0) {
    return null;
  }

  const turnEnc = [];
  cursor = 0;
  for (let tIdx = 0; tIdx < nTurns; tIdx++) {
    const line = lines[tIdx];
    const enc = tokenizer.encode(line);
    const tOffs = [];
    for (let i = 0; i < enc.offsets.length; i++) {
      const sa = enc.offsets[i][0];
      let sb = enc.offsets[i][1];
      while (sa < sb && sb < line.length && COMBINING_MARK_RE.test(line[sb])) {
        sb++;
      }
      tOffs.push([cursor + sa, cursor + sb]);
    }
    turnEnc.push({ ids: enc.ids, offs: tOffs });
    cursor += line.length + 1;
  }

  const budget = Math.max(16, maxTokens - (1 + nTurns));
  const caps = allocateTurnBudgets(turnEnc.map((x) => x.ids.length), budget);

  const seqIds = [bosId];
  const seqOffs = [[0, 0]];
  for (let tIdx = 0; tIdx < nTurns; tIdx++) {
    const { ids: tIds, offs: tOffs } = turnEnc[tIdx];
    const cap = caps[tIdx];
    if (tIds.length > cap) {
      const head = Math.floor(cap / 3);
      const tail = cap - head;
      for (let i = 0; i < head; i++) {
        seqIds.push(tIds[i]);
        seqOffs.push(tOffs[i]);
      }
      if (tail > 0) {
        for (let i = tIds.length - tail; i < tIds.length; i++) {
          seqIds.push(tIds[i]);
          seqOffs.push(tOffs[i]);
        }
      }
    } else {
      for (let i = 0; i < tIds.length; i++) {
        seqIds.push(tIds[i]);
        seqOffs.push(tOffs[i]);
      }
    }
    seqIds.push(sepId);
    seqOffs.push([0, 0]);
  }

  const wordSubIdxs = [];
  let posTok = 1;
  const nSeq = seqOffs.length;
  for (let wIdx = 0; wIdx < wordsMeta.length; wIdx++) {
    const { a, b } = wordsMeta[wIdx];
    while (posTok < nSeq && (seqOffs[posTok][0] === seqOffs[posTok][1] || seqOffs[posTok][1] <= a)) {
      posTok++;
    }
    const idxs = [];
    let j = posTok;
    while (j < nSeq) {
      const sa = seqOffs[j][0];
      const sb = seqOffs[j][1];
      if (sa === sb) {
        j++;
        continue;
      }
      if (sa >= b) break;
      if (sb > a) {
        idxs.push(j);
        if (idxs.length === maxSubwordsPerWord) break;
      }
      j++;
    }
    wordSubIdxs.push(idxs);
  }

  const maxL = seqIds.length;
  const maxW = wordsMeta.length;
  let maxS = 1;
  for (const idxs of wordSubIdxs) {
    if (idxs.length > maxS) maxS = idxs.length;
  }

  const ids = new BigInt64Array(maxL);
  ids.fill(BigInt(padId));
  const attnMask = new Uint8Array(maxL);
  for (let i = 0; i < maxL; i++) {
    ids[i] = BigInt(seqIds[i]);
    attnMask[i] = 1;
  }

  const wordSubIdx = new BigInt64Array(maxW * maxS);
  const wordSubMask = new Uint8Array(maxW * maxS);
  const wordMask = new Uint8Array(maxW);
  const wordTurn = new BigInt64Array(maxW);
  const wordRecency = new BigInt64Array(maxW);
  const wordBytes = new BigInt64Array(maxW * maxWordBytes);

  for (let w = 0; w < maxW; w++) {
    const metaW = wordsMeta[w];
    wordMask[w] = 1;
    wordTurn[w] = BigInt(metaW.tIdx);
    wordRecency[w] = BigInt(metaW.recency);
    const bBase = w * maxWordBytes;
    for (let bi = 0; bi < metaW.bArr.length; bi++) {
      wordBytes[bBase + bi] = BigInt(metaW.bArr[bi]);
    }
    const idxs = wordSubIdxs[w];
    const sBase = w * maxS;
    for (let si = 0; si < idxs.length; si++) {
      wordSubIdx[sBase + si] = BigInt(idxs[si]);
      wordSubMask[sBase + si] = 1;
    }
  }

  return {
    text,
    lines,
    lineStarts,
    wordsMeta,
    shapes: { maxL, maxW, maxS, maxWordBytes },
    arrays: {
      ids,
      attn_mask: attnMask,
      word_sub_idx: wordSubIdx,
      word_sub_mask: wordSubMask,
      word_mask: wordMask,
      word_turn: wordTurn,
      word_recency: wordRecency,
      word_bytes: wordBytes,
    },
  };
}

function logsumexp3(a, b, c) {
  const m = Math.max(a, b, c);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m) + Math.exp(c - m));
}

/** Thuật toán Forward-Backward BIO-CRF (3 trạng thái: 0=O, 1=B-EV, 2=I-EV) có ràng buộc biên lượt thoại. */
export function crfMarginalsSingle(emissionsFlat, wLen, turnStarts, startTr, endTr, trans) {
  if (wLen === 0) return new Float64Array(0);

  const st0 = startTr[0];
  const st1 = startTr[1];
  const st2 = NEG_INF;

  const t00 = trans[0][0], t01 = trans[0][1], t02 = NEG_INF;
  const t10 = trans[1][0], t11 = trans[1][1], t12 = trans[1][2];
  const t20 = trans[2][0], t21 = trans[2][1], t22 = trans[2][2];

  const alphas = new Float64Array(wLen * 3);
  alphas[0] = st0 + emissionsFlat[0];
  alphas[1] = st1 + emissionsFlat[1];
  alphas[2] = st2 + emissionsFlat[2];

  for (let t = 1; t < wLen; t++) {
    const p = (t - 1) * 3;
    const c = t * 3;
    const a0 = alphas[p], a1 = alphas[p + 1], a2 = alphas[p + 2];
    const isStart = turnStarts[t];
    const s02 = isStart ? NEG_INF : t02;
    const s12 = isStart ? NEG_INF : t12;
    const s22 = isStart ? NEG_INF : t22;

    alphas[c]     = logsumexp3(a0 + t00, a1 + t10, a2 + t20) + emissionsFlat[c];
    alphas[c + 1] = logsumexp3(a0 + t01, a1 + t11, a2 + t21) + emissionsFlat[c + 1];
    alphas[c + 2] = logsumexp3(a0 + s02, a1 + s12, a2 + s22) + emissionsFlat[c + 2];
  }

  const last = (wLen - 1) * 3;
  const logZ = logsumexp3(
    alphas[last] + endTr[0],
    alphas[last + 1] + endTr[1],
    alphas[last + 2] + endTr[2],
  );

  const betas = new Float64Array(wLen * 3);
  betas[last] = endTr[0];
  betas[last + 1] = endTr[1];
  betas[last + 2] = endTr[2];

  for (let t = wLen - 2; t >= 0; t--) {
    const c = t * 3;
    const n = (t + 1) * 3;
    const nb0 = emissionsFlat[n] + betas[n];
    const nb1 = emissionsFlat[n + 1] + betas[n + 1];
    const nb2 = emissionsFlat[n + 2] + betas[n + 2];
    const isNextStart = turnStarts[t + 1];
    const s02 = isNextStart ? NEG_INF : t02;
    const s12 = isNextStart ? NEG_INF : t12;
    const s22 = isNextStart ? NEG_INF : t22;

    betas[c]     = logsumexp3(t00 + nb0, t01 + nb1, s02 + nb2);
    betas[c + 1] = logsumexp3(t10 + nb0, t11 + nb1, s12 + nb2);
    betas[c + 2] = logsumexp3(t20 + nb0, t21 + nb1, s22 + nb2);
  }

  const probs = new Float64Array(wLen * 3);
  for (let t = 0; t < wLen; t++) {
    const c = t * 3;
    const p0 = Math.exp(alphas[c] + betas[c] - logZ);
    const p1 = Math.exp(alphas[c + 1] + betas[c + 1] - logZ);
    const p2 = Math.exp(alphas[c + 2] + betas[c + 2] - logZ);
    const denom = Math.max(p0 + p1 + p2, 1e-8);
    probs[c] = p0 / denom;
    probs[c + 1] = p1 / denom;
    probs[c + 2] = p2 / denom;
  }
  return probs;
}

/** Giải mã Viterbi BIO-CRF (khi chosen_decoder === 'crf'). */
export function viterbiSingle(emissionsFlat, wLen, turnStarts, startTr, endTr, trans, oBias = 0.0) {
  if (wLen === 0) return [];

  const t00 = trans[0][0], t01 = trans[0][1], t02 = NEG_INF;
  const t10 = trans[1][0], t11 = trans[1][1], t12 = trans[1][2];
  const t20 = trans[2][0], t21 = trans[2][1], t22 = trans[2][2];

  let dp0 = startTr[0] + emissionsFlat[0] + oBias;
  let dp1 = startTr[1] + emissionsFlat[1];
  let dp2 = NEG_INF + emissionsFlat[2];

  const bp = new Int8Array(wLen * 3);

  for (let t = 1; t < wLen; t++) {
    const c = t * 3;
    const isStart = turnStarts[t];
    const s02 = isStart ? NEG_INF : t02;
    const s12 = isStart ? NEG_INF : t12;
    const s22 = isStart ? NEG_INF : t22;

    const cands = [
      [dp0 + t00, dp1 + t10, dp2 + t20],
      [dp0 + t01, dp1 + t11, dp2 + t21],
      [dp0 + s02, dp1 + s12, dp2 + s22],
    ];
    const nextDp = [0, 0, 0];
    for (let s = 0; s < 3; s++) {
      let bestPrev = 0;
      let bestVal = cands[s][0];
      if (cands[s][1] > bestVal) { bestVal = cands[s][1]; bestPrev = 1; }
      if (cands[s][2] > bestVal) { bestVal = cands[s][2]; bestPrev = 2; }
      bp[c + s] = bestPrev;
      nextDp[s] = bestVal + emissionsFlat[c + s] + (s === 0 ? oBias : 0);
    }
    dp0 = nextDp[0];
    dp1 = nextDp[1];
    dp2 = nextDp[2];
  }

  dp0 += endTr[0];
  dp1 += endTr[1];
  dp2 += endTr[2];

  let cur = 0;
  let best = dp0;
  if (dp1 > best) { best = dp1; cur = 1; }
  if (dp2 > best) { cur = 2; }

  const seq = new Array(wLen);
  seq[wLen - 1] = cur;
  for (let t = wLen - 2; t >= 0; t--) {
    cur = bp[(t + 1) * 3 + cur];
    seq[t] = cur;
  }
  return seq;
}

function applyMarginalPostproc(words, text, threshold, singleWordThreshold, bridgeRatio) {
  const nW = words.length;
  let marks = words.map((w) => w.prob >= threshold);

  if (bridgeRatio > 0 && nW >= 3) {
    const bridgeThr = threshold * bridgeRatio;
    const bridged = marks.slice();
    for (let i = 1; i < nW - 1; i++) {
      if (!marks[i] && marks[i - 1] && marks[i + 1] && words[i].prob >= bridgeThr) {
        if (!text.slice(words[i - 1].b, words[i + 1].a).includes('\n')) {
          bridged[i] = true;
        }
      }
    }
    marks = bridged;
  }

  if (singleWordThreshold != null && singleWordThreshold > threshold) {
    const filtered = marks.slice();
    for (let i = 0; i < nW; i++) {
      if (!marks[i]) continue;
      const leftSame = i > 0 && marks[i - 1] && !text.slice(words[i - 1].b, words[i].a).includes('\n');
      const rightSame = i + 1 < nW && marks[i + 1] && !text.slice(words[i].b, words[i + 1].a).includes('\n');
      if (!leftSame && !rightSame && words[i].prob < singleWordThreshold) {
        filtered[i] = false;
      }
    }
    marks = filtered;
  }
  return marks;
}

function applyEmptyFallback(marks, words, text, fallbackMinProb, expandRatio = 0.70, maxWords = 5) {
  if (marks.some(Boolean) || words.length === 0 || fallbackMinProb == null || fallbackMinProb > 1.0) {
    return marks;
  }
  let peakI = 0;
  let peakP = words[0].prob;
  for (let i = 1; i < words.length; i++) {
    if (words[i].prob > peakP) {
      peakP = words[i].prob;
      peakI = i;
    }
  }
  if (peakP < fallbackMinProb) return marks;

  const out = marks.slice();
  out[peakI] = true;
  const cut = peakP * expandRatio;
  let lo = peakI;
  let hi = peakI;
  while (hi - lo + 1 < maxWords) {
    const canL = lo > 0 && !text.slice(words[lo - 1].b, words[lo].a).includes('\n') && words[lo - 1].prob >= cut;
    const canR = hi + 1 < words.length && !text.slice(words[hi].b, words[hi + 1].a).includes('\n') && words[hi + 1].prob >= cut;
    if (!canL && !canR) break;
    if (canL && (!canR || words[lo - 1].prob >= words[hi + 1].prob)) {
      lo--;
      out[lo] = true;
    } else {
      hi++;
      out[hi] = true;
    }
  }
  return out;
}

function marksToRuns(words, marks, text) {
  const runs = [];
  for (let i = 0; i < words.length; i++) {
    if (!marks[i]) continue;
    const { a, b } = words[i];
    if (runs.length > 0) {
      const gap = text.slice(runs[runs.length - 1][1], a);
      if (!gap.includes('\n') && NON_NEWLINE_SPACE_RE.test(stripInvisibles(gap))) {
        runs[runs.length - 1][1] = b;
        continue;
      }
    }
    runs.push([a, b]);
  }
  return runs;
}

/** Giải mã kết quả từ 3 tensor đầu ra (logits, emissions, severity) theo crf_params.json. */
export function decodeDualORCOutput({ prep, logits, emissions, severity, meta }) {
  const labels = meta.labels || ['AN_TOAN', 'XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO'];
  const temp = Number(meta.temperature || 1.0);

  // Softmax hiệu chuẩn theo nhiệt độ T
  const scaled = new Float64Array(labels.length);
  let maxVal = -Infinity;
  for (let i = 0; i < labels.length; i++) {
    const v = Number(logits[i]) / temp;
    scaled[i] = v;
    if (v > maxVal) maxVal = v;
  }
  let sumExp = 0;
  const probs = new Float64Array(labels.length);
  for (let i = 0; i < labels.length; i++) {
    const e = Math.exp(scaled[i] - maxVal);
    probs[i] = e;
    sumExp += e;
  }
  let predIdx = 0;
  let bestP = -1;
  const xac_suat = {};
  for (let i = 0; i < labels.length; i++) {
    const p = probs[i] / sumExp;
    xac_suat[labels[i]] = p;
    if (p > bestP) {
      bestP = p;
      predIdx = i;
    }
  }

  const nhan = labels[predIdx];
  const rui_ro = Math.max(0, Math.min(100, Math.round(Number(severity) * 10000) / 100));

  const { text, lines, lineStarts, wordsMeta } = prep;
  const wLen = wordsMeta.length;
  const turnStarts = wordsMeta.map((w) => w.firstInTurn);
  const marg = crfMarginalsSingle(emissions, wLen, turnStarts, meta.start, meta.end, meta.trans);

  const words = wordsMeta.map((w, i) => ({
    a: w.a,
    b: w.b,
    tIdx: w.tIdx,
    firstInTurn: w.firstInTurn,
    prob: Math.max(0, Math.min(1, marg[i * 3 + 1] + marg[i * 3 + 2])),
  }));

  let runs = [];
  if (predIdx !== 0 && wLen > 0) {
    const mode = String(meta.chosen_decoder || 'marginal');
    const fb = meta.fallback_min_prob;
    if (mode === 'crf') {
      const tags = viterbiSingle(emissions, wLen, turnStarts, meta.start, meta.end, meta.trans, Number(meta.o_bias || 0));
      if (tags.some((t) => t !== 0)) {
        for (let i = 0; i < wLen; i++) {
          const tag = tags[i];
          if (tag === 0) continue;
          const { a, b } = words[i];
          if (tag === 2 && runs.length > 0 && NON_NEWLINE_SPACE_RE.test(text.slice(runs[runs.length - 1][1], a))) {
            runs[runs.length - 1][1] = b;
          } else {
            runs.push([a, b]);
          }
        }
      } else if (fb != null) {
        const marks = applyEmptyFallback(new Array(wLen).fill(false), words, text, Number(fb));
        runs = marksToRuns(words, marks, text);
      }
    } else {
      let marks = applyMarginalPostproc(
        words,
        text,
        Number(meta.marginal_threshold),
        meta.single_word_threshold != null ? Number(meta.single_word_threshold) : null,
        Number(meta.bridge_ratio || 0),
      );
      if (fb != null) {
        marks = applyEmptyFallback(marks, words, text, Number(fb));
      }
      runs = marksToRuns(words, marks, text);
    }
  }

  // Đổi các khoảng [a, b) trên văn bản nối về { dong, start, end, text } từng dòng
  const bang_chung = runs.map(([a, b]) => {
    let dong = 0;
    for (let i = lineStarts.length - 1; i >= 0; i--) {
      if (a >= lineStarts[i]) {
        dong = i;
        break;
      }
    }
    const start = a - lineStarts[dong];
    const end = b - lineStarts[dong];
    return {
      dong,
      start,
      end,
      text: lines[dong].slice(start, end),
    };
  });

  return { nhan, xac_suat, rui_ro, bang_chung };
}
