// Luồng phân tích đa tầm nhìn (multi-scale) chống pha loãng ngữ cảnh cho web-v2.
//
// Nguyên tắc:
// - Không thay đổi mô hình hay ngưỡng huấn luyện; giữ nguyên giao diện tối giản của cột AI.
// - Luôn chạy ngữ cảnh đầy đủ (tối đa 8 lượt) trước, chỉ mở rộng sang các tầng cục bộ khi cần:
//     1. Toàn cửa sổ (≤ 8 lượt)
//     2. Ngữ cảnh cục bộ gần (3 lượt cuối; sau đó cặp 2 lượt cuối [lượt trước, tin mới])
//        → giữ ngữ cảnh giải oan câu đùa game/bạn bè, đồng thời chống loãng ở cửa sổ 6–8 lượt
//        và chống pha loãng khi nhiều tin cùng một phe bị gộp vào 1 lượt.
//     3. Tin đơn có điều kiện:
//        - Khi chỉ có 1 phe nhắn liên tiếp (không có đối đáp luân phiên): kiểm tra tin mới độc lập.
//        - Khi có đối đáp 2 phe nhưng tầng 2 vẫn AN_TOAN: chỉ lật sang LUA_DAO / QUAY_ROI khi
//          độ tin cậy cao (>= 0.85) và có bằng chứng rõ (chống lọt dạng nick bạn bị hack xin OTP / gạ gẫm).
//     4. Quét mệnh đề nội tin cho tin nhắn đơn dài (>= 24 từ) vẫn ra AN_TOAN: tách theo dấu kết câu
//        (bảo vệ URL và từ né lọc kiểu n.g.u) hoặc cửa sổ trượt theo từ nếu không có dấu câu.
//     5. Trí nhớ cửa sổ 8 lượt + giải oan lượt kế tiếp:
//        - Tin nguy hại đã phát hiện trong cửa sổ 8 lượt không bị "vùi" mất bởi các tin an toàn nhắn sau.
//        - Ngoại lệ giải oan: tin lượt đầu gắn cờ ranh giới (p < 0.78) nếu ngay lượt trả lời tiếp theo
//          làm ngữ cảnh 2 lượt trở về AN_TOAN chắc chắn (p_safe >= 0.75) thì tự gỡ cờ.
// - Tách rõ đánh giá bản thân tin nháp (khi `Tôi` chuẩn bị gửi trong hội thoại đã có cờ cũ)
//   khỏi trạng thái chung của hội thoại.

import { MAX_LINES } from '../labels.js';
import { stripInvisibles } from './dualorc.js';

export const SEVERITY_RANK = {
  AN_TOAN: 0,
  XUC_PHAM: 1,
  QUAY_ROI: 2,
  LUA_DAO: 3,
  DE_DOA: 4,
};

// Ngưỡng kiểm soát chất lượng giữa chống bỏ lọt và chống báo nhầm:
const SOLO_HIGH_CONF_LABELS = new Set(['LUA_DAO', 'QUAY_ROI', 'XUC_PHAM']);
const SOLO_HIGH_CONF_MIN_P = 0.85;
const SUBWINDOW_OVERRIDE_MIN_P = 0.58;
const ESCALATION_MIN_P = 0.82;
const LONG_MSG_MIN_WORDS = 24;
const CLAUSE_MIN_WORDS = 3;
const CLAUSE_MIN_P = 0.72;
const WORD_CHUNK_SIZE = 20;
const WORD_CHUNK_STRIDE = 10;
const MAX_CLAUSE_SCANS = 6;
const DISAMBIG_PRIOR_MAX_P = 0.86;
const DISAMBIG_SAFE_MIN_P = 0.80;

const URL_RE = /https?:\/\/\S+|www\.\S+/i;

export function countWords(text) {
  const cleaned = stripInvisibles(String(text || '')).trim();
  if (!cleaned) return 0;
  return cleaned.split(/\s+/).length;
}

export function isHigherSeverity(candLabel, baseLabel, candProb = 0, baseProb = 0) {
  const rc = SEVERITY_RANK[candLabel] ?? 0;
  const rb = SEVERITY_RANK[baseLabel] ?? 0;
  if (rc !== rb) return rc > rb;
  return candProb > baseProb;
}

/**
 * Gộp chuỗi tin nhắn từ 0..idx (hoặc kèm draftMsg) thành tối đa MAX_LINES lượt luân phiên 2 phe.
 * Không cắt ngang các tin an toàn giữa lượt để tránh làm mờ `.out-ctx` bất thường trên khung chat.
 */
export function buildTurnWindow(messages, idx, { draftMsg = null, whoOf, maxLines = MAX_LINES } = {}) {
  const rawItems = messages.slice(0, idx).map((m, i) => ({ i, m }));
  if (draftMsg != null) {
    rawItems.push({ i: -1, m: draftMsg });
  } else if (messages[idx]) {
    rawItems.push({ i: idx, m: messages[idx] });
  }

  const turns = [];
  for (const item of rawItems) {
    const party = item.m.from === 'me' ? 'me' : 'them';
    const authorLabel = whoOf ? whoOf(item.m) : (party === 'me' ? 'Bạn' : item.m.author || 'Người kia');
    const prev = turns[turns.length - 1];
    if (prev && prev.party === party) {
      const start = prev.text.length + 1;
      prev.text += ` ${item.m.text}`;
      prev.time = item.m.time;
      if (!prev.authors.includes(authorLabel)) prev.authors.push(authorLabel);
      prev.who = prev.authors.join(', ');
      prev.parts.push({ i: item.i, m: item.m, start, end: prev.text.length });
    } else {
      turns.push({
        party,
        authors: [authorLabel],
        who: authorLabel,
        time: item.m.time,
        text: item.m.text,
        parts: [{ i: item.i, m: item.m, start: 0, end: item.m.text.length }],
      });
    }
  }

  return turns.slice(-maxLines);
}

/**
 * Tách một tin nhắn dài thành các mệnh đề kèm vị trí UTF-16 gốc [start, end).
 * Không tách dấu chấm bên trong URL (`http://a.b`) hay ký tự né lọc (`n.g.u`, `n. g. u`).
 */
export function splitMessageSegments(text) {
  const s = String(text || '');
  const len = s.length;
  if (!len) return [];

  // Bước 1: tìm điểm ngắt câu theo xuống dòng hoặc dấu kết câu có khoảng trắng theo sau.
  const breaks = [0];
  for (let i = 0; i < len; i++) {
    const ch = s[i];
    if (ch === '\n') {
      let j = i;
      while (j < len && s[j] === '\n') j++;
      breaks.push(j);
      i = j - 1;
      continue;
    }
    if ((ch === '!' || ch === '?' || ch === ';' || ch === '.') && i + 1 < len && /\s/.test(s[i + 1])) {
      // Nếu là dấu chấm, bỏ qua nếu ngay trước đó là 1 ký tự chữ đơn lẻ (kiểu "n. g. u") hoặc chữ số viết tắt ("1. ")
      if (ch === '.') {
        const before = s.slice(0, i);
        if (/(?:^|[\s.])[\p{L}\p{N}]$/u.test(before)) continue;
      }
      let j = i + 1;
      while (j < len && /\s/.test(s[j])) j++;
      if (j < len) breaks.push(j);
    }
  }
  breaks.push(len);

  const sentenceSegs = [];
  for (let k = 0; k < breaks.length - 1; k++) {
    let a = breaks[k];
    let b = breaks[k + 1];
    while (a < b && /\s/.test(s[a])) a++;
    while (b > a && /\s/.test(s[b - 1])) b--;
    if (b > a) sentenceSegs.push({ start: a, end: b, text: s.slice(a, b) });
  }

  // Bước 2: mở rộng các đoạn còn quá dài (ví dụ đoạn văn không có dấu câu) bằng cửa sổ trượt theo từ.
  const expanded = [];
  for (const seg of sentenceSegs) {
    const segWords = [];
    for (const m of seg.text.matchAll(/\S+/g)) {
      segWords.push({ start: seg.start + m.index, end: seg.start + m.index + m[0].length });
    }
    if (segWords.length >= LONG_MSG_MIN_WORDS + 4) {
      for (let wi = 0; wi < segWords.length; wi += WORD_CHUNK_STRIDE) {
        const endIdx = Math.min(segWords.length - 1, wi + WORD_CHUNK_SIZE - 1);
        const a = segWords[wi].start;
        const b = segWords[endIdx].end;
        expanded.push({ start: a, end: b, text: s.slice(a, b) });
        if (endIdx === segWords.length - 1) break;
      }
    } else if (segWords.length >= CLAUSE_MIN_WORDS || URL_RE.test(seg.text)) {
      expanded.push(seg);
    }
  }

  // Loại bỏ trường hợp duy nhất trùng nguyên tin gốc (đã chạy ở tầng solo rồi)
  if (expanded.length === 1 && expanded[0].start === 0 && expanded[0].end === len) {
    return [];
  }
  return expanded.slice(0, MAX_CLAUSE_SCANS);
}

/** Tạo một sub-turn từ danh sách slice để đưa vào ONNX và ánh xạ ngược offset chính xác. */
function makeSubTurn(rowIdx, sliceSpecs) {
  let text = '';
  const slices = [];
  for (const spec of sliceSpecs) {
    const origOffset = spec.origOffset ?? 0;
    const chunkText = spec.text ?? spec.part.m.text.slice(origOffset);
    const subStart = text ? text.length + 1 : 0;
    text = text ? `${text} ${chunkText}` : chunkText;
    slices.push({
      rowIdx,
      part: spec.part,
      origOffset,
      subStart,
      subEnd: text.length,
    });
  }
  return { text, slices };
}

function dedupeSpans(spans) {
  if (spans.length <= 1) return spans;
  const sorted = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const out = [];
  for (const sp of sorted) {
    const prev = out[out.length - 1];
    if (!prev) {
      out.push(sp);
    } else if (sp.start < prev.end) {
      // Giữ khoảng rộng hơn hoặc gộp nếu cùng nhãn
      if (sp.end > prev.end && (!prev.nhan || !sp.nhan || prev.nhan === sp.nhan)) {
        prev.end = sp.end;
      }
    } else {
      out.push(sp);
    }
  }
  return out;
}

/** Ánh xạ kết quả từ một truy vấn con (subTurns) về đúng cấu trúc rows & bubbleSpans của cửa sổ `turns`. */
export function packSubAnalysis(turns, subTurns, out, { scope = 'context', totalMs = out.ms } = {}) {
  const bubbleSpans = new Map();
  const rowSpans = turns.map(() => []);

  for (const t of turns) {
    for (const p of t.parts) bubbleSpans.set(p.i, []);
  }

  for (const sp of out.bang_chung || []) {
    const st = subTurns[sp.dong];
    if (!st) continue;
    for (const sl of st.slices) {
      const ovStart = Math.max(sp.start, sl.subStart);
      const ovEnd = Math.min(sp.end, sl.subEnd);
      if (ovEnd <= ovStart) continue;
      const localStart = sl.origOffset + (ovStart - sl.subStart);
      const localEnd = sl.origOffset + (ovEnd - sl.subStart);
      const verbatim = sl.part.m.text.slice(localStart, localEnd);
      if (!verbatim) continue;

      const rowStart = sl.part.start + localStart;
      const rowEnd = sl.part.start + localEnd;
      bubbleSpans.get(sl.part.i)?.push({
        start: localStart,
        end: localEnd,
        text: verbatim,
        nhan: out.nhan,
      });
      rowSpans[sl.rowIdx].push({
        dong: sl.rowIdx,
        start: rowStart,
        end: rowEnd,
        text: turns[sl.rowIdx].text.slice(rowStart, rowEnd),
      });
    }
  }

  for (const [k, list] of bubbleSpans.entries()) {
    if (list.length > 1) {
      const partObj = turns.flatMap((t) => t.parts).find((p) => p.i === k);
      const merged = dedupeSpans(list);
      if (partObj) {
        for (const sp of merged) sp.text = partObj.m.text.slice(sp.start, sp.end);
      }
      bubbleSpans.set(k, merged);
    }
  }

  const rows = turns.map((t, dong) => {
    const spans = dedupeSpans(rowSpans[dong]);
    for (const sp of spans) sp.text = t.text.slice(sp.start, sp.end);
    return {
      who: t.who,
      time: t.time,
      text: t.text,
      msgIndices: t.parts.map((p) => p.i),
      partRanges: t.parts.map((p) => [p.start, p.end]),
      spans,
    };
  });

  return {
    nhan: out.nhan,
    xac_suat: out.xac_suat,
    rui_ro: out.rui_ro,
    ms: totalMs,
    scope,
    rows,
    bubbleSpans,
  };
}

/** Dựng kết quả cửa sổ từ cảnh báo đã lưu của một hoặc nhiều tin nằm trong cửa sổ 8 lượt hiện tại. */
function buildFromWindowAlerts(turns, primaryAlert, allActiveAlerts, totalMs) {
  const bubbleSpans = new Map();
  const rowSpans = turns.map(() => []);

  for (const t of turns) {
    for (const p of t.parts) bubbleSpans.set(p.i, []);
  }

  // Đưa toàn bộ bằng chứng của các tin nguy hại còn trong cửa sổ về đúng vị trí verbatim
  const alertByPartIdx = new Map();
  for (const a of allActiveAlerts) alertByPartIdx.set(a.msgIdx, a);

  turns.forEach((t, dong) => {
    for (const p of t.parts) {
      const alert = alertByPartIdx.get(p.i);
      if (!alert || !alert.spans?.length) continue;
      const validSpans = [];
      for (const sp of alert.spans) {
        if (sp.start >= 0 && sp.end <= p.m.text.length && sp.end > sp.start) {
          const text = p.m.text.slice(sp.start, sp.end);
          validSpans.push({ start: sp.start, end: sp.end, text, nhan: alert.nhan });
          rowSpans[dong].push({
            dong,
            start: p.start + sp.start,
            end: p.start + sp.end,
            text: t.text.slice(p.start + sp.start, p.start + sp.end),
          });
        }
      }
      bubbleSpans.set(p.i, dedupeSpans(validSpans));
    }
  });

  const rows = turns.map((t, dong) => {
    const spans = dedupeSpans(rowSpans[dong]);
    for (const sp of spans) sp.text = t.text.slice(sp.start, sp.end);
    return {
      who: t.who,
      time: t.time,
      text: t.text,
      msgIndices: t.parts.map((p) => p.i),
      partRanges: t.parts.map((p) => [p.start, p.end]),
      spans,
    };
  });

  return {
    nhan: primaryAlert.nhan,
    xac_suat: primaryAlert.xac_suat,
    rui_ro: primaryAlert.rui_ro,
    ms: totalMs,
    scope: 'window_memory',
    rows,
    bubbleSpans,
  };
}

/** Kết hợp thêm bằng chứng từ các cảnh báo cũ cùng cửa sổ vào một kết quả đang có (không làm mất tô sáng tin cũ). */
function mergePriorBubbleSpans(analysis, turns, activeAlerts) {
  if (!activeAlerts.length) return analysis;
  const alertByIdx = new Map(activeAlerts.map((a) => [a.msgIdx, a]));
  const bubbleSpans = new Map(analysis.bubbleSpans);
  const rows = analysis.rows.map((r) => ({ ...r, spans: r.spans.slice() }));

  turns.forEach((t, dong) => {
    const row = rows[dong];
    let rowModified = false;
    for (const p of t.parts) {
      const alert = alertByIdx.get(p.i);
      if (!alert || !alert.spans?.length) continue;
      const existing = bubbleSpans.get(p.i) || [];
      if (existing.length > 0) continue; // Đã có bằng chứng từ lần chạy mới
      const restored = [];
      for (const sp of alert.spans) {
        if (sp.start >= 0 && sp.end <= p.m.text.length && sp.end > sp.start) {
          restored.push({
            start: sp.start,
            end: sp.end,
            text: p.m.text.slice(sp.start, sp.end),
            nhan: alert.nhan,
          });
          row.spans.push({
            dong,
            start: p.start + sp.start,
            end: p.start + sp.end,
            text: t.text.slice(p.start + sp.start, p.start + sp.end),
          });
          rowModified = true;
        }
      }
      if (restored.length) bubbleSpans.set(p.i, dedupeSpans(restored));
    }
    if (rowModified) {
      row.spans = dedupeSpans(row.spans);
      for (const sp of row.spans) sp.text = t.text.slice(sp.start, sp.end);
    }
  });
  return { ...analysis, rows, bubbleSpans };
}

/**
 * Quét mệnh đề nội tin cho một tin nhắn dài (chỉ gọi khi toàn tin đã trả AN_TOAN).
 * Trả về { out, subTurns } thắng hoặc null nếu mọi mệnh đề đều an toàn.
 */
async function scanLongMessageClauses(rowIdx, targetPart, runCached) {
  const text = targetPart.m.text;
  if (countWords(text) < LONG_MSG_MIN_WORDS) return null;
  const segs = splitMessageSegments(text);
  if (!segs.length) return null;

  let best = null;
  for (const seg of segs) {
    if (!stripInvisibles(seg.text).trim()) continue;
    const subTurns = [makeSubTurn(rowIdx, [{ part: targetPart, origOffset: seg.start, text: seg.text }])];
    const out = await runCached(subTurns);
    if (out.nhan === 'AN_TOAN') continue;
    const pTop = out.xac_suat?.[out.nhan] ?? 0;
    if (pTop < CLAUSE_MIN_P || !(out.bang_chung?.length > 0)) continue;

    if (!best || isHigherSeverity(out.nhan, best.out.nhan, pTop, best.pTop)) {
      best = { out, subTurns, pTop };
    }
  }
  return best;
}

/**
 * Chạy luồng phân tích đa tầm nhìn đầy đủ cho cửa sổ `turns`.
 *
 * Trả về:
 * - `analysis`: kết quả hiển thị cho hội thoại & cột AI (`{ nhan, xac_suat, rui_ro, ms, scope, rows, bubbleSpans }`)
 * - `msgAlert`: cảnh báo gắn riêng cho tin mới nhất (hoặc `null` nếu tin mới nhất an toàn)
 * - `clearedIndices`: danh sách chỉ số tin cũ được giải oan ở lượt kế tiếp (nếu có)
 * - `draftHarmful`: riêng khi `isDraft && draftParty === 'me'` — bản thân tin nháp có nên chặn/cảnh báo trước khi gửi không
 * - `draftAnalysis`: phân tích riêng cho tin nháp (dùng cho hộp cảnh báo trước khi gửi & cột AI lúc đang nháp)
 */
export async function analyzeWindow({
  turns,
  classify,
  priority = false,
  isDraft = false,
  draftParty = null,
  priorActiveAlerts = [], // [{ msgIdx, nhan, xac_suat, rui_ro, spans, prob, wasTurnZero }]
}) {
  let totalMs = 0;
  const memo = new Map();

  async function runCached(subTurns) {
    const lines = subTurns.map((st) => st.text);
    const key = lines.join('\n');
    if (memo.has(key)) return memo.get(key);
    const out = await classify(lines, { priority });
    totalMs += out.ms || 0;
    memo.set(key, out);
    return out;
  }

  const fullSubTurns = turns.map((t, rowIdx) => makeSubTurn(rowIdx, t.parts.map((part) => ({ part }))));
  const lastRowIdx = turns.length - 1;
  const lastTurn = turns[lastRowIdx];
  const targetPart = lastTurn.parts[lastTurn.parts.length - 1];
  const targetHasText = Boolean(targetPart?.m?.text && stripInvisibles(targetPart.m.text).trim());
  const windowMsgCount = turns.reduce((sum, t) => sum + t.parts.length, 0);

  // Tầng 1: Toàn cửa sổ (≤ 8 lượt)
  const ctxOut = await runCached(fullSubTurns);
  let selectedSub = { out: ctxOut, subTurns: fullSubTurns, scope: 'context' };

  // Tầng 2: Ngữ cảnh cục bộ gần (chỉ chạy khi Tầng 1 ra AN_TOAN và cửa sổ có khả năng bị pha loãng)
  if (selectedSub.out.nhan === 'AN_TOAN' && windowMsgCount > 1 && targetHasText) {
    // 2a. Cửa sổ 3 lượt cuối (nếu lượt cuối gộp > 2 tin thì chỉ giữ 2 tin cuối của lượt đó)
    if (turns.length > 3 || (turns.length >= 2 && lastTurn.parts.length > 2)) {
      const startRow = Math.max(0, turns.length - 3);
      const local3Sub = [];
      for (let r = startRow; r < turns.length; r++) {
        const parts = (r === lastRowIdx && turns[r].parts.length > 2)
          ? turns[r].parts.slice(-2)
          : turns[r].parts;
        local3Sub.push(makeSubTurn(r, parts.map((part) => ({ part }))));
      }
      const out3 = await runCached(local3Sub);
      if (out3.nhan !== 'AN_TOAN' && (out3.xac_suat?.[out3.nhan] ?? 0) >= SUBWINDOW_OVERRIDE_MIN_P) {
        selectedSub = { out: out3, subTurns: local3Sub, scope: 'local_3t' };
      }
    }

    // 2b. Cặp 2 lượt cuối: chỉ chạy khi có >= 4 lượt (chống loãng cửa sổ dài) HOẶC lượt cuối bị gộp nhiều tin.
    //     Không cắt bỏ lượt 0 ở hội thoại 3 lượt sạch (tránh mất ngữ cảnh đùa game/bạn bè).
    if (
      selectedSub.out.nhan === 'AN_TOAN'
      && (turns.length >= 4 || (turns.length >= 2 && lastTurn.parts.length > 1))
    ) {
      const prevRowIdx = lastRowIdx - 1;
      const prevParts = turns[prevRowIdx].parts.slice(-2);
      const pair2Sub = [
        makeSubTurn(prevRowIdx, prevParts.map((part) => ({ part }))),
        makeSubTurn(lastRowIdx, [{ part: targetPart }]),
      ];
      const out2 = await runCached(pair2Sub);
      if (out2.nhan !== 'AN_TOAN' && (out2.xac_suat?.[out2.nhan] ?? 0) >= SUBWINDOW_OVERRIDE_MIN_P) {
        selectedSub = { out: out2, subTurns: pair2Sub, scope: 'pair_2t' };
      }
    }
  }

  // Tầng 3: Kiểm tra tin đơn / lượt cuối có điều kiện
  if (selectedSub.out.nhan === 'AN_TOAN' && targetHasText) {
    const soloSub = [makeSubTurn(lastRowIdx, [{ part: targetPart }])];
    if (turns.length === 1 && lastTurn.parts.length > 1) {
      // Toàn bộ là 1 phe nhắn nhiều tin liên tiếp (không có đối đáp): kiểm tra độc lập tin mới nhất
      const soloOut = await runCached(soloSub);
      if (soloOut.nhan !== 'AN_TOAN') {
        selectedSub = { out: soloOut, subTurns: soloSub, scope: 'solo_burst' };
      }
    } else if (turns.length >= 2) {
      // Có đối đáp 2 phe: nếu lượt cuối gộp nhiều tin ngắn (VD: 2 tin chửi liên tiếp trong chat 2 lượt),
      // kiểm tra cả lượt cuối trước để giữ đủ bằng chứng trên cả hai bóng tin, rồi mới thử tin cuối đơn lẻ.
      // Chỉ chấp nhận khi thuộc nhóm ít bị ngữ cảnh đùa phủ định (LUA_DAO, QUAY_ROI, XUC_PHAM độ tin cậy cao)
      // và có bằng chứng rõ — không lật câu đùa game kiểu "tao giết mày luôn giờ" (DE_DOA) khi 3 lượt là AN_TOAN.
      const candidates = [];
      if (lastTurn.parts.length > 1 && lastTurn.parts.length <= 3) {
        candidates.push({
          sub: [makeSubTurn(lastRowIdx, lastTurn.parts.map((part) => ({ part })))],
          scope: 'last_turn_high_conf',
        });
      }
      candidates.push({ sub: soloSub, scope: 'solo_high_conf' });

      for (const cand of candidates) {
        const soloOut = await runCached(cand.sub);
        const pSolo = soloOut.xac_suat?.[soloOut.nhan] ?? 0;
        const allowLabel = SOLO_HIGH_CONF_LABELS.has(soloOut.nhan)
          || (priorActiveAlerts.length > 0 && soloOut.nhan === 'DE_DOA');
        if (
          allowLabel
          && pSolo >= SOLO_HIGH_CONF_MIN_P
          && (soloOut.bang_chung?.length > 0)
        ) {
          selectedSub = { out: soloOut, subTurns: cand.sub, scope: cand.scope };
          break;
        }
      }
    }
  }

  // Tầng 4: Quét mệnh đề nội tin cho tin nhắn đơn dài (>= 24 từ) vẫn ra AN_TOAN
  if (selectedSub.out.nhan === 'AN_TOAN' && targetHasText) {
    const clauseHit = await scanLongMessageClauses(lastRowIdx, targetPart, runCached);
    if (clauseHit) {
      selectedSub = { out: clauseHit.out, subTurns: clauseHit.subTurns, scope: 'clause_scan' };
    }
  }

  // Kiểm tra leo thang nhãn: nếu hội thoại đã mang nhãn nguy hại nhẹ hơn DE_DOA ở Tầng 1,
  // kiểm tra tin mới nhất xem có mang nhãn nặng hơn hẳn (VD: từ XUC_PHAM leo lên DE_DOA / LUA_DAO) không.
  if (
    selectedSub.out.nhan !== 'AN_TOAN'
    && selectedSub.out.nhan !== 'DE_DOA'
    && windowMsgCount > 1
    && priorActiveAlerts.length > 0
    && targetHasText
  ) {
    const soloSub = [makeSubTurn(lastRowIdx, [{ part: targetPart }])];
    const soloOut = await runCached(soloSub);
    const pSolo = soloOut.xac_suat?.[soloOut.nhan] ?? 0;
    if (
      soloOut.nhan !== 'AN_TOAN'
      && (SEVERITY_RANK[soloOut.nhan] > SEVERITY_RANK[selectedSub.out.nhan])
      && pSolo >= ESCALATION_MIN_P
    ) {
      selectedSub = { out: soloOut, subTurns: soloSub, scope: 'escalation' };
    }
  }

  const baseAnalysis = packSubAnalysis(turns, selectedSub.subTurns, selectedSub.out, {
    scope: selectedSub.scope,
    totalMs,
  });

  // Đánh giá riêng tin nháp của `Tôi` khi hội thoại đã có cờ nguy hại cũ:
  // tránh chặn oan tin an toàn, đồng thời tránh bỏ lọt tin nguy hại mới của `Tôi`.
  let draftHarmful = false;
  let draftAnalysis = baseAnalysis;

  if (isDraft && draftParty === 'me') {
    if (priorActiveAlerts.length === 0) {
      draftHarmful = baseAnalysis.nhan !== 'AN_TOAN';
    } else if (targetHasText) {
      const soloSub = [makeSubTurn(lastRowIdx, [{ part: targetPart }])];
      let draftSub = { out: await runCached(soloSub), subTurns: soloSub, scope: 'draft_solo' };
      if (draftSub.out.nhan === 'AN_TOAN') {
        const clauseHit = await scanLongMessageClauses(lastRowIdx, targetPart, runCached);
        if (clauseHit) {
          draftSub = { out: clauseHit.out, subTurns: clauseHit.subTurns, scope: 'draft_clause' };
        }
      }
      // Nếu lượt liền kề trước đó là câu hỏi/phản hồi an toàn của đối phương (không mang cờ cũ),
      // kiểm tra thêm cặp [lượt trước, nháp] để bắt câu nối tiếp ẩn ý (VD: "đọc 6 số vừa gửi về máy").
      if (draftSub.out.nhan === 'AN_TOAN' && turns.length >= 2) {
        const prevRow = turns[lastRowIdx - 1];
        const priorIdxSet = new Set(priorActiveAlerts.map((a) => a.msgIdx));
        const prevHasAlert = prevRow.parts.some((p) => priorIdxSet.has(p.i));
        if (!prevHasAlert && (baseAnalysis.bubbleSpans.get(targetPart.i)?.length || 0) > 0) {
          const pairSub = [
            makeSubTurn(lastRowIdx - 1, prevRow.parts.slice(-2).map((part) => ({ part }))),
            makeSubTurn(lastRowIdx, [{ part: targetPart }]),
          ];
          const pairOut = await runCached(pairSub);
          if (pairOut.nhan !== 'AN_TOAN') {
            draftSub = { out: pairOut, subTurns: pairSub, scope: 'draft_pair' };
          }
        }
      }
      if (draftSub.out.nhan !== 'AN_TOAN') {
        draftHarmful = true;
        draftAnalysis = packSubAnalysis(turns, draftSub.subTurns, draftSub.out, {
          scope: draftSub.scope,
          totalMs,
        });
      } else {
        draftHarmful = false;
      }
    }
  }

  // Tầng 5: Giải oan lượt kế tiếp & Trí nhớ cửa sổ 8 lượt (chống "vùi" tin nguy hại giữa dàn tin an toàn)
  const clearedIndices = [];
  const activeAlerts = [];

  for (const prior of priorActiveAlerts) {
    // Cơ chế giải oan có kiểm soát: chỉ áp dụng cho tin lượt đầu tiên bị gắn cờ ranh giới (p < 0.78)
    // khi tin trả lời trực tiếp ở lượt ngay kế tiếp làm ngữ cảnh 2 lượt trở về AN_TOAN chắc chắn.
    const isImmediatePrevTurn = turns.length === 2
      && prior.wasTurnZero
      && turns[0].parts.some((p) => p.i === prior.msgIdx);

    if (
      isImmediatePrevTurn
      && prior.prob < DISAMBIG_PRIOR_MAX_P
      && ctxOut.nhan === 'AN_TOAN'
      && (ctxOut.xac_suat?.AN_TOAN ?? 0) >= DISAMBIG_SAFE_MIN_P
    ) {
      clearedIndices.push(prior.msgIdx);
      continue;
    }
    activeAlerts.push(prior);
  }

  // Lọc hiện tượng tràn biên bằng chứng CRF sang tin an toàn mới gộp chung lượt:
  // Nếu trong cùng lượt cuối đã có tin trước đó mang cờ nguy hại và ngữ cảnh gộp tràn span sang tin mới,
  // kiểm tra nhanh tin mới đơn lẻ: nếu tin mới thực sự AN_TOAN thì gỡ phần tô tràn biên.
  if (
    baseAnalysis.nhan !== 'AN_TOAN'
    && selectedSub.scope === 'context'
    && lastTurn.parts.length > 1
    && priorActiveAlerts.length > 0
    && (baseAnalysis.bubbleSpans.get(targetPart.i)?.length || 0) > 0
    && targetHasText
  ) {
    const priorIdxSet = new Set(priorActiveAlerts.map((a) => a.msgIdx));
    const sameTurnHasPrior = lastTurn.parts.slice(0, -1).some((p) => priorIdxSet.has(p.i));
    if (sameTurnHasPrior) {
      const soloSub = [makeSubTurn(lastRowIdx, [{ part: targetPart }])];
      const soloCheck = await runCached(soloSub);
      if (soloCheck.nhan === 'AN_TOAN') {
        baseAnalysis.bubbleSpans.set(targetPart.i, []);
        const row = baseAnalysis.rows[lastRowIdx];
        row.spans = row.spans
          .map((sp) => {
            if (sp.start >= targetPart.start) return null;
            if (sp.end > targetPart.start) {
              const trimmedEnd = Math.min(sp.end, targetPart.start - 1);
              if (trimmedEnd <= sp.start) return null;
              return { ...sp, end: trimmedEnd, text: row.text.slice(sp.start, trimmedEnd) };
            }
            return sp;
          })
          .filter(Boolean);
      }
    }
  }

  // Xác định cảnh báo gắn riêng cho tin mới nhất (`msgAlert`) để lưu làm trí nhớ cho các bước sau:
  // - Nếu không có cờ cũ: bất kỳ kết quả nguy hại nào ở trên đều gắn cho tin mới nhất.
  // - Nếu đã có cờ cũ: chỉ gắn `msgAlert` mới khi chính tin mới nhất (nhánh cục bộ/solo/clause/escalation hoặc
  //   có bằng chứng nằm trên tin mới nhất) thực sự mang nội dung nguy hại.
  let msgAlert = null;
  const targetSpans = baseAnalysis.bubbleSpans.get(targetPart.i) || [];
  if (isDraft && draftParty === 'me' && priorActiveAlerts.length > 0) {
    if (draftHarmful) {
      const dSpans = draftAnalysis.bubbleSpans.get(targetPart.i) || [];
      msgAlert = {
        nhan: draftAnalysis.nhan,
        xac_suat: draftAnalysis.xac_suat,
        rui_ro: draftAnalysis.rui_ro,
        spans: dSpans,
        prob: draftAnalysis.xac_suat?.[draftAnalysis.nhan] ?? 0,
        wasTurnZero: turns.length === 1 && lastTurn.parts.length === 1,
      };
    }
  } else if (baseAnalysis.nhan !== 'AN_TOAN') {
    const newMsgTriggered = priorActiveAlerts.length === 0
      || selectedSub.scope !== 'context'
      || targetSpans.length > 0;
    if (newMsgTriggered) {
      msgAlert = {
        nhan: baseAnalysis.nhan,
        xac_suat: baseAnalysis.xac_suat,
        rui_ro: baseAnalysis.rui_ro,
        spans: targetSpans,
        prob: baseAnalysis.xac_suat?.[baseAnalysis.nhan] ?? 0,
        wasTurnZero: turns.length === 1 && lastTurn.parts.length === 1,
      };
    }
  }

  // Quét dự phòng lịch sử chưa từng phân tích (khi mở hội thoại nhiều lượt sẵn có mà chưa qua luồng tuần tự).
  // Chỉ kích hoạt khi cửa sổ đủ dài để xảy ra pha loãng (>= 4 lượt hoặc tổng >= 45 từ).
  const totalWindowWords = turns.reduce((sum, t) => sum + countWords(t.text), 0);
  if (
    baseAnalysis.nhan === 'AN_TOAN'
    && activeAlerts.length === 0
    && !isDraft
    && (turns.length >= 4 || totalWindowWords >= 45)
  ) {
    let bestHist = null;
    const scannedParts = turns.flatMap((t, rIdx) => t.parts.map((part) => ({ rIdx, part })));
    // Bỏ qua tin cuối cùng (đã kiểm tra ở Tầng 1–4)
    for (let k = scannedParts.length - 2; k >= 0; k--) {
      const { rIdx, part } = scannedParts[k];
      if (part.m.analysis || part.m.msgAlert !== undefined) continue;
      if (!part.m.text || !stripInvisibles(part.m.text).trim()) continue;
      // Kiểm tra trong cặp cục bộ nhỏ để vẫn giữ ngữ cảnh giải oan câu đùa / trích dẫn
      let sub;
      if (rIdx > 0) {
        sub = [
          makeSubTurn(rIdx - 1, turns[rIdx - 1].parts.slice(-1).map((p) => ({ part: p }))),
          makeSubTurn(rIdx, [{ part }]),
        ];
      } else if (turns.length >= 2) {
        sub = [
          makeSubTurn(0, [{ part }]),
          makeSubTurn(1, turns[1].parts.slice(0, 1).map((p) => ({ part: p }))),
        ];
      } else {
        sub = [makeSubTurn(0, [{ part }])];
      }
      const out = await runCached(sub);
      if (out.nhan !== 'AN_TOAN') {
        const pTop = out.xac_suat?.[out.nhan] ?? 0;
        if (!bestHist || isHigherSeverity(out.nhan, bestHist.out.nhan, pTop, bestHist.pTop)) {
          bestHist = { out, subTurns: sub, pTop };
        }
      }
    }
    if (bestHist) {
      const histAnalysis = packSubAnalysis(turns, bestHist.subTurns, bestHist.out, {
        scope: 'history_scan',
        totalMs,
      });
      return {
        analysis: histAnalysis,
        msgAlert: null,
        clearedIndices,
        draftHarmful,
        draftAnalysis,
      };
    }
  }

  // Tổng hợp trạng thái hội thoại cuối cùng trên cửa sổ 8 lượt
  let finalAnalysis = baseAnalysis;
  if (
    msgAlert
    && isHigherSeverity(
      msgAlert.nhan,
      finalAnalysis.nhan,
      msgAlert.prob,
      finalAnalysis.xac_suat?.[finalAnalysis.nhan] ?? 0,
    )
  ) {
    finalAnalysis = draftHarmful
      ? draftAnalysis
      : buildFromWindowAlerts(turns, msgAlert, [{ msgIdx: targetPart.i, ...msgAlert }], totalMs);
  }

  if (activeAlerts.length > 0) {
    let bestPrior = activeAlerts[0];
    for (let i = 1; i < activeAlerts.length; i++) {
      if (isHigherSeverity(activeAlerts[i].nhan, bestPrior.nhan, activeAlerts[i].prob, bestPrior.prob)) {
        bestPrior = activeAlerts[i];
      }
    }

    if (finalAnalysis.nhan === 'AN_TOAN') {
      // Cửa sổ hiện tại bị pha loãng về AN_TOAN nhưng vẫn còn tin nguy hại đang hiệu lực trong 8 lượt
      finalAnalysis = buildFromWindowAlerts(turns, bestPrior, activeAlerts, totalMs);
    } else if (isHigherSeverity(bestPrior.nhan, finalAnalysis.nhan, bestPrior.prob, finalAnalysis.xac_suat?.[finalAnalysis.nhan] ?? 0)) {
      // Giữ mức độ ưu tiên cao hơn trong cửa sổ (DE_DOA > LUA_DAO > QUAY_ROI > XUC_PHAM),
      // đồng thời giữ cả bằng chứng mới lẫn cũ.
      const combined = msgAlert
        ? [...activeAlerts, { msgIdx: targetPart.i, ...msgAlert }]
        : activeAlerts;
      finalAnalysis = buildFromWindowAlerts(turns, bestPrior, combined, totalMs);
    } else {
      finalAnalysis = mergePriorBubbleSpans(finalAnalysis, turns, activeAlerts);
    }
  }

  finalAnalysis.ms = totalMs;

  return {
    analysis: finalAnalysis,
    msgAlert,
    clearedIndices,
    draftHarmful,
    draftAnalysis,
  };
}
