/**
 * Bộ chuẩn hóa tiếng Việt chống né lọc (chuyển thể từ normalizer.py sang JavaScript).
 *
 * Quy trình xử lý:
 * 1. Chuẩn hóa Unicode NFC
 * 2. Thay thế ký tự đồng dạng (số/ký hiệu thay cho chữ cái)
 * 3. Xóa ký tự phân cách xen giữa chữ cái (.-_~·)
 * 4. Rút gọn ký tự lặp kéo dài (>= 3 lần liên tiếp)
 * 5. Tra từ điển khôi phục teencode
 *
 * @module normalizer
 */

let TEENCODE_MAP = {};

/**
 * Tải từ điển teencode từ file JSON.
 * Cần gọi hàm này trước khi thực hiện chuẩn hóa có tra từ điển.
 */
export async function init(dictUrl = '../teencode_dict.json') {
  try {
    const resp = await fetch(dictUrl);
    if (resp.ok) {
      TEENCODE_MAP = await resp.json();
    } else {
      console.warn('[normalizer] Không tải được từ điển teencode, bỏ qua bước tra từ điển.');
    }
  } catch (e) {
    console.warn('[normalizer] Lỗi khi tải từ điển:', e.message);
  }
}

const VIET_LETTER = /[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]/;

function isVietnameseLetter(ch) {
  return VIET_LETTER.test(ch);
}

function normalizeUnicode(text) {
  return text.normalize('NFC');
}

// Bảng ánh xạ ký tự đồng dạng phổ biến
const LOOKALIKE_MAP = {
  '4': 'a', '3': 'e', '1': 'i', '0': 'o',
  '5': 's', '9': 'g', '8': 'b', '@': 'a',
};

function mapLookalikeToken(token) {
  // Giữ nguyên các token bắt đầu bằng số (ví dụ: 40km, 100k, 5G)
  if (token.length > 0 && /\d/.test(token[0])) {
    return token;
  }

  let numAlpha = 0;
  let numConvertible = 0;
  for (const ch of token) {
    if (isVietnameseLetter(ch)) numAlpha++;
    if (ch in LOOKALIKE_MAP) numConvertible++;
  }

  // Chỉ thay thế khi token chủ yếu là chữ cái
  if (numConvertible === 0 || numAlpha <= numConvertible) {
    return token;
  }

  return Array.from(token)
    .map(ch => (ch in LOOKALIKE_MAP ? LOOKALIKE_MAP[ch] : ch))
    .join('');
}

function mapLookalikeChars(text) {
  return text.replace(/\S+/g, match => mapLookalikeToken(match));
}

// Chỉ xóa ký tự phân cách khi nó nằm giữa hai chữ cái tiếng Việt (ví dụ: n.g.u -> ngu, còn 3.14 giữ nguyên)
const SEPARATOR_PATTERN = /(?<=[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])[.\-_·~](?=[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])/g;

function removeInsertedSeparators(text) {
  return text.replace(SEPARATOR_PATTERN, '');
}

// Rút gọn ký tự chữ cái lặp >= 3 lần (ví dụ: nàoooo -> nào)
const ELONGATION_PATTERN = /([a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])\1{2,}/g;

function stripBase(ch) {
  let s = ch.replace(/đ/g, 'd').replace(/Đ/g, 'D');
  const nfkd = s.normalize('NFD');
  let result = '';
  for (const c of nfkd) {
    if (!/[\u0300-\u036F]/.test(c)) {
      result += c;
    }
  }
  return result;
}

function collapseElongation(text) {
  text = text.replace(ELONGATION_PATTERN, '$1');

  // Lược bỏ ký tự gốc lặp phía sau ký tự có dấu (ví dụ: quáa -> quá)
  if (text.length >= 2) {
    const chars = Array.from(text);
    const result = [chars[0]];
    for (let i = 1; i < chars.length; i++) {
      const prev = chars[i - 1];
      const curr = chars[i];
      if (
        prev !== curr &&
        isVietnameseLetter(curr) &&
        isVietnameseLetter(prev) &&
        stripBase(prev) === stripBase(curr) &&
        prev !== stripBase(prev)
      ) {
        continue;
      }
      result.push(curr);
    }
    text = result.join('');
  }

  return text;
}

const LEADING_PUNCT_RE = /^([^\w]*)/;
const TRAILING_PUNCT_RE = /([^\w]*)$/;

function transferCase(original, replacement) {
  if (!original || !replacement) return replacement;
  if (
    original[0] === original[0].toUpperCase() &&
    original[0] !== original[0].toLowerCase() &&
    replacement[0] === replacement[0].toLowerCase()
  ) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

function restoreTeencode(text) {
  if (Object.keys(TEENCODE_MAP).length === 0) return text;

  const words = text.split(' ');
  const result = [];

  for (const w of words) {
    const leadMatch = w.match(LEADING_PUNCT_RE);
    const trailMatch = w.match(TRAILING_PUNCT_RE);
    const lead = leadMatch ? leadMatch[1] : '';
    const trail = trailMatch ? trailMatch[1] : '';
    const core = trail ? w.slice(lead.length, w.length - trail.length) : w.slice(lead.length);

    if (!core) {
      result.push(w);
      continue;
    }

    const lookup = core.toLowerCase();
    if (lookup in TEENCODE_MAP) {
      const replacement = transferCase(core, TEENCODE_MAP[lookup]);
      result.push(lead + replacement + trail);
    } else {
      result.push(w);
    }
  }

  return result.join(' ');
}

/**
 * Thực hiện toàn bộ chuỗi chuẩn hóa văn bản.
 */
export function normalize(text) {
  text = normalizeUnicode(text);
  text = mapLookalikeChars(text);
  text = removeInsertedSeparators(text);
  text = collapseElongation(text);
  text = restoreTeencode(text);
  return text;
}
