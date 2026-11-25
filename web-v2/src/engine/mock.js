// Mô hình GIẢ chỉ để thiết kế UI: chuẩn hóa vài kiểu teencode rồi dò từ khóa.
// KHÔNG phải kết quả của SafeViChi. Thay bằng engine ONNX khi có mô hình thật.

// Từ điển teencode rút gọn (bản thật nằm ở bộ chuẩn hóa của nhóm).
const TEEN = { k: 'không', ko: 'không', kh: 'không', m: 'mày', t: 'tao', rep: 'trả lời', dc: 'được', đc: 'được', ns: 'nói', j: 'gì', z: 'vậy', vs: 'với', bít: 'biết', iu: 'yêu', e: 'em', a: 'anh', otp: 'OTP' };
const LOOK = { 0: 'o', 1: 'i', 3: 'e', 4: 'a', '@': 'a' };

function normWord(w) {
  let s = w.normalize('NFC');
  const core = s.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
  if (!core || /^(https?:|www\.)/i.test(core)) return s; // giữ nguyên đường link
  let c = core.replace(/(?<=\p{L})[.\-_·~*]+(?=\p{L})/gu, ''); // n.g.u -> ngu
  if (/\p{L}/u.test(c) && /[0134@]/.test(c) && !/^\d/.test(c)) c = [...c].map((ch) => LOOK[ch] || ch).join('');
  c = c.replace(/(\p{L})\1{2,}/gu, '$1'); // kéo dài
  const t = TEEN[c.toLowerCase()];
  if (t) c = t;
  return s.replace(core, c);
}

// Chuẩn hóa từng từ, nhớ từ đó nằm ở đâu trong câu gốc để tô bằng chứng đúng chỗ.
function normalizeLine(line) {
  const words = [];
  for (const m of line.matchAll(/\S+/g)) {
    // Vị trí bằng chứng bỏ dấu câu dính hai đầu ("hồn," -> "hồn").
    const lead = m[0].match(/^[^\p{L}\p{N}]*/u)[0].length;
    const trail = m[0].match(/[^\p{L}\p{N}]*$/u)[0].length;
    const start = m.index + (lead < m[0].length ? lead : 0);
    const end = m.index + m[0].length - (lead < m[0].length ? trail : 0);
    words.push({ raw: m[0], start, end, norm: normWord(m[0]) });
  }
  return { text: words.map((w) => w.norm).join(' '), words };
}

const RULES = [
  ['DE_DOA', ['liệu hồn', 'tung ảnh', 'đánh chết', 'biết tay', 'giết']],
  ['LUA_DAO', ['otp', 'trúng thưởng', 'bấm vào link', 'chuyển khoản', 'bị khóa', 'http']],
  ['QUAY_ROI', ['gửi ảnh', 'không trả lời', 'đi chơi với anh', 'sao im']],
  ['XUC_PHAM', ['ngu', 'óc chó', 'mất dạy', 'đồ điên']],
];

export const mockEngine = {
  backend: 'mock',
  async load(onProgress) {
    for (let i = 1; i <= 10; i++) { await new Promise((r) => setTimeout(r, 50)); onProgress(i, 10); }
  },
  async classify(lines) {
    const t0 = performance.now();
    const normed = lines.map(normalizeLine);
    const bang_chung = [];
    const hits = { DE_DOA: 0, LUA_DAO: 0, QUAY_ROI: 0, XUC_PHAM: 0 };
      normed.forEach(({ words }, dong) => {
      // Một từ gốc có thể thành nhiều từ (rep -> trả lời), nên dò trên từ đã tách và nhớ từ gốc.
      const toks = words.flatMap((w, wi) => w.norm.split(' ').map((sub) => ({ wi, sub: sub.toLowerCase().replace(/[^\p{L}\p{N}:/.]/gu, '') })));
      for (const [label, keys] of RULES) {
        for (const key of keys) {
          const parts = key.split(' ');
          for (let i = 0; i + parts.length <= toks.length; i++) {
            if (parts.every((p, k) => (p === 'http' ? toks[i + k].sub.startsWith('http') : toks[i + k].sub === p))) {
              const a = words[toks[i].wi];
              const b = words[toks[i + parts.length - 1].wi];
              bang_chung.push({ dong, start: a.start, end: b.end, text: lines[dong].slice(a.start, b.end), nhan: label });
              hits[label]++;
            }
          }
        }
      }
    });
    const order = ['DE_DOA', 'LUA_DAO', 'QUAY_ROI', 'XUC_PHAM'];
    const raw = { AN_TOAN: 1, XUC_PHAM: 0.2, QUAY_ROI: 0.2, DE_DOA: 0.2, LUA_DAO: 0.2 };
    for (const l of order) raw[l] += hits[l] * 2.5 + (hits[l] ? 0.01 * (4 - order.indexOf(l)) : 0);
    const nhan = order.reduce((best, l) => (hits[l] && (!best || raw[l] > raw[best]) ? l : best), null) || 'AN_TOAN';
    if (nhan !== 'AN_TOAN') raw.AN_TOAN = 0.3;
    raw[nhan] += 1.5;
    const sum = Object.values(raw).reduce((x, y) => x + y, 0);
    const xac_suat = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / sum]));
    await new Promise((r) => setTimeout(r, 15 + Math.random() * 25)); // giả lập thời gian chạy
    return {
      nhan, xac_suat,
      bang_chung: nhan === 'AN_TOAN' ? [] : bang_chung,
      chuan_hoa: normed.map((n) => n.text),
      ms_model: performance.now() - t0,
    };
  },
};
