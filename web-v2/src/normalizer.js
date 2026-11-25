// Bộ chuẩn hóa chống né lọc, viết lại từ src/shared/normalization/normalizer.py (bản 24/11).
// Kết quả phải giống hệt Python: kiểm bằng `node scripts/so_khop_chuan_hoa.mjs`.
//
// Khác Python ở một điểm: mỗi ký tự đầu ra nhớ khoảng [start, end) trong câu GỐC người dùng nhập,
// để bằng chứng mô hình tìm trên câu đã chuẩn hóa được tô lại đúng chỗ trên câu gốc.
//
// Lưu ý: bước teencode của Python tách theo khoảng trắng rồi nối bằng " " nên gộp cả "\n".
// Vì vậy hàm normalizeLines chuẩn hóa TỪNG DÒNG riêng.

const LOOKALIKE_MAP = { 4: 'a', 3: 'e', 1: 'i', 0: 'o', 5: 's', 9: 'g', 8: 'b', '@': 'a' };
const VIET = /[a-zA-ZÀ-ỹ]/u;
const IS_ALPHA = /^\p{L}$/u;
const IS_DIGIT = /^\p{Nd}$/u;
const IS_WORD = /^[\p{L}\p{N}_]$/u;
const IS_SPACE = /^\s$/u;
const COMBINING = /\p{M}/gu;

let teencode = {};
export function setTeencodeDict(dict) { teencode = dict || {}; }
export async function loadTeencodeDict(url = '/teencode_dict.json') {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Không tải được từ điển teencode.');
  setTeencodeDict(await res.json());
}

// Một "đơn vị" = một ký tự đầu ra + khoảng ký tự gốc nó đến từ đó.
const unit = (ch, start, end) => ({ ch, start, end });
const text = (units) => units.map((u) => u.ch).join('');

// Bước 1: NFC. Chuẩn hóa từng cụm (ký tự gốc + dấu kết hợp) để giữ được vị trí.
function nfc(raw) {
  const out = [];
  const re = /\P{M}\p{M}*|\p{M}+/gsu;
  let m;
  while ((m = re.exec(raw))) {
    const start = m.index;
    const end = start + m[0].length;
    for (const ch of m[0].normalize('NFC')) out.push(unit(ch, start, end));
  }
  return out;
}

// Chia chuỗi đơn vị thành các đoạn liên tiếp "khoảng trắng" / "không khoảng trắng".
function tokens(units) {
  const parts = [];
  for (const u of units) {
    const space = IS_SPACE.test(u.ch);
    const last = parts[parts.length - 1];
    if (last && last.space === space) last.units.push(u);
    else parts.push({ space, units: [u] });
  }
  return parts;
}

// Bước 2: ký tự đồng dạng (4 -> a, 0 -> o...) trong token chủ yếu là chữ.
function mapLookalike(units) {
  const out = [];
  for (const part of tokens(units)) {
    const tok = part.units;
    if (part.space || IS_DIGIT.test(tok[0].ch)) { out.push(...tok); continue; }
    const alpha = tok.filter((u) => IS_ALPHA.test(u.ch)).length;
    const conv = tok.filter((u) => u.ch in LOOKALIKE_MAP).length;
    if (conv === 0 || alpha <= conv) { out.push(...tok); continue; }
    for (const u of tok) out.push(u.ch in LOOKALIKE_MAP ? { ...u, ch: LOOKALIKE_MAP[u.ch] } : u);
  }
  return out;
}

// Bước 3: xóa . - _ · ~ nằm giữa hai chữ cái (n.g.u -> ngu, giữ 3.14).
function removeSeparators(units) {
  return units.filter((u, i) => !(
    '.-_·~'.includes(u.ch) && i > 0 && i < units.length - 1
    && VIET.test(units[i - 1].ch) && VIET.test(units[i + 1].ch)
  ));
}

const stripBase = (ch) => ch.replace(/đ/g, 'd').replace(/Đ/g, 'D').normalize('NFD').replace(COMBINING, '');

// Bước 4: rút gọn chữ lặp >= 3 lần, rồi bỏ chữ gốc thừa sau chữ có dấu (quáa -> quá).
function collapseElongation(units) {
  const a = [];
  for (let i = 0; i < units.length;) {
    let j = i + 1;
    if (VIET.test(units[i].ch)) while (j < units.length && units[j].ch === units[i].ch) j++;
    if (j - i >= 3) { a.push(unit(units[i].ch, units[i].start, units[j - 1].end)); i = j; }
    else { a.push(units[i]); i++; }
  }
  if (a.length < 2) return a;
  const out = [a[0]];
  for (let i = 1; i < a.length; i++) {
    const prev = a[i - 1].ch;
    const curr = a[i].ch;
    if (prev !== curr && IS_ALPHA.test(curr) && IS_ALPHA.test(prev)
        && stripBase(prev) === stripBase(curr) && prev !== stripBase(prev)) {
      const kept = out[out.length - 1];
      out[out.length - 1] = { ...kept, end: Math.max(kept.end, a[i].end) };
      continue;
    }
    out.push(a[i]);
  }
  return out;
}

const isUpper = (c) => c !== c.toLowerCase() && c === c.toUpperCase();
const isLower = (c) => c !== c.toUpperCase() && c === c.toLowerCase();

// Bước 5: tra teencode từng từ; nối các từ bằng một dấu cách như Python.
function restoreTeencode(units) {
  if (!Object.keys(teencode).length) return units;
  const words = tokens(units).filter((p) => !p.space).map((p) => p.units);
  const out = [];
  words.forEach((w, wi) => {
    if (wi > 0) { const prev = out[out.length - 1]; out.push(unit(' ', prev.end, w[0].start)); }
    let lead = 0;
    while (lead < w.length && !IS_WORD.test(w[lead].ch)) lead++;
    let trailStart = w.length;
    while (trailStart > lead && !IS_WORD.test(w[trailStart - 1].ch)) trailStart--;
    const core = w.slice(lead, trailStart);
    const key = text(core).toLowerCase();
    if (!core.length || !(key in teencode)) { out.push(...w); return; }
    let rep = teencode[key];
    if (rep && isUpper(core[0].ch) && isLower(rep[0])) rep = rep[0].toUpperCase() + rep.slice(1);
    const start = core[0].start;
    const end = core[core.length - 1].end;
    out.push(...w.slice(0, lead), ...[...rep].map((ch) => unit(ch, start, end)), ...w.slice(trailStart));
  });
  return out;
}

/** Chuẩn hóa một dòng. Trả về { text, map } với map[i] = [start, end) trong `raw`. */
export function normalize(raw) {
  let u = nfc(raw);
  u = mapLookalike(u);
  u = removeSeparators(u);
  u = collapseElongation(u);
  u = restoreTeencode(u);
  return { text: text(u), map: u.map(({ start, end }) => [start, end]) };
}

/** Đổi khoảng [s, e) trên câu đã chuẩn hóa về khoảng trên câu gốc. */
export function toOriginalSpan(map, s, e) {
  if (s >= e || !map.length) return null;
  return [map[s][0], map[Math.min(e, map.length) - 1][1]];
}

export const normalizeLines = (lines) => lines.map(normalize);
