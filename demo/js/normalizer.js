/**
 * Bộ chuẩn hóa chống né lọc — Port từ Python (normalizer.py) sang JavaScript.
 *
 * Pipeline 5 bước:
 *   B1. Chuẩn hóa Unicode NFC
 *   B2. Ánh xạ lookalike (số/ký hiệu → chữ, chỉ trong context chữ cái)
 *   B3. Xóa separator chèn thêm (chỉ .-_ giữa ký tự chữ, KHÔNG xóa space)
 *   B4. Gộp ký tự lặp (>= 3 lần → 1 lần)
 *   B5. Tra từ điển teencode (JSON trích từ ViLexNorm)
 *
 * Nguyên tắc vàng: THÀ KHÔNG SỬA, CHỨ ĐỪNG SỬA SAI.
 *
 * @module normalizer
 */

// ============================================================
// Từ điển teencode — sẽ được load bằng init()
// ============================================================

let TEENCODE_MAP = {};

/**
 * Khởi tạo normalizer: load từ điển teencode từ file JSON.
 * Phải gọi hàm này 1 lần trước khi dùng normalize().
 * @param {string} dictUrl - URL tới file teencode_dict.json (mặc định cùng thư mục demo)
 */
export async function init(dictUrl = '../teencode_dict.json') {
  try {
    const resp = await fetch(dictUrl);
    if (resp.ok) {
      TEENCODE_MAP = await resp.json();
    } else {
      console.warn('[normalizer] Không tải được từ điển teencode, bỏ qua Bước 5.');
    }
  } catch (e) {
    console.warn('[normalizer] Lỗi tải từ điển:', e.message);
  }
}

// ============================================================
// Regex: kiểm tra ký tự tiếng Việt (bao gồm Unicode dấu)
// ============================================================

// Phạm vi chữ cái tiếng Việt: a-zA-Z + các ký tự có dấu À-ỹ
const VIET_LETTER = /[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]/;

function isVietnameseLetter(ch) {
  return VIET_LETTER.test(ch);
}

// ============================================================
// Bước 1: Chuẩn hóa Unicode NFC
// ============================================================

function normalizeUnicode(text) {
  return text.normalize('NFC');
}

// ============================================================
// Bước 2: Lookalike — AN TOÀN
// Chỉ chuyển số→chữ khi token PHẦN LỚN là chữ cái.
// VD: "m4y" (2 alpha, 1 digit) → "may" ✓
//     "400k" (1 alpha, 3 digit) → giữ nguyên ✓
// ============================================================

const LOOKALIKE_MAP = {
  '4': 'a', '3': 'e', '1': 'i', '0': 'o',
  '5': 's', '9': 'g', '8': 'b', '@': 'a',
};

function mapLookalikeToken(token) {
  // Guard 1: token bắt đầu bằng digit → số đo (40km) → giữ nguyên
  if (token.length > 0 && /\d/.test(token[0])) {
    return token;
  }

  let numAlpha = 0;
  let numConvertible = 0;
  for (const ch of token) {
    if (isVietnameseLetter(ch)) numAlpha++;
    if (ch in LOOKALIKE_MAP) numConvertible++;
  }

  // Guard 2: không có gì chuyển, hoặc token chủ yếu là số → bỏ qua
  if (numConvertible === 0 || numAlpha <= numConvertible) {
    return token;
  }

  return Array.from(token)
    .map(ch => (ch in LOOKALIKE_MAP ? LOOKALIKE_MAP[ch] : ch))
    .join('');
}

function mapLookalikeChars(text) {
  // Tách theo khoảng trắng, xử lý từng token (giống \S+ trong Python)
  return text.replace(/\S+/g, match => mapLookalikeToken(match));
}

// ============================================================
// Bước 3: Separator removal — BẢO THỦ
// Chỉ xóa dấu . - _ · ~ nằm giữa 2 KÝ TỰ CHỮ CÁI.
// VD: "n.g.u" → "ngu" ✓ , "3.14" → giữ nguyên ✓
// ============================================================

// Regex: ký tự tiếng Việt, theo sau là separator, theo sau là ký tự tiếng Việt
// Dùng lookbehind/lookahead để chỉ xóa separator
const SEPARATOR_PATTERN = /(?<=[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])[.\-_·~](?=[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])/g;

function removeInsertedSeparators(text) {
  return text.replace(SEPARATOR_PATTERN, '');
}

// ============================================================
// Bước 4: Collapse elongation — BẢO THỦ
// Gộp khi >= 3 ký tự GIỐNG NHAU liên tiếp: "nàoooo" → "nào"
// ============================================================

// Chỉ gộp KÝ TỰ CHỮ CÁI lặp >= 3 lần. Không gộp số (100000) hay dấu câu (...).
const ELONGATION_PATTERN = /([a-zA-Z\u00C0-\u024F\u1E00-\u1EFF])\1{2,}/g;

/**
 * Bỏ dấu thanh/phụ để lấy ký tự gốc. VD: á → a, ồ → o, đ → d.
 */
function stripBase(ch) {
  let s = ch.replace(/đ/g, 'd').replace(/Đ/g, 'D');
  // NFD tách dấu thành combining characters, sau đó loại bỏ chúng
  const nfkd = s.normalize('NFD');
  let result = '';
  for (const c of nfkd) {
    // Combining characters nằm trong range U+0300 - U+036F
    if (!/[\u0300-\u036F]/.test(c)) {
      result += c;
    }
  }
  return result;
}

function collapseElongation(text) {
  // 4a: collapse ký tự giống hệt >= 3 lần → 1
  text = text.replace(ELONGATION_PATTERN, '$1');

  // 4b: strip trailing base letter sau ký tự có dấu
  // VD: "quáa" → "quá" (chữ 'a' là base của 'á')
  if (text.length >= 2) {
    const chars = Array.from(text);
    const result = [chars[0]];
    for (let i = 1; i < chars.length; i++) {
      const prev = chars[i - 1];
      const curr = chars[i];
      // Nếu curr là base letter của prev (và prev có dấu) → bỏ curr
      if (
        prev !== curr &&
        isVietnameseLetter(curr) &&
        isVietnameseLetter(prev) &&
        stripBase(prev) === stripBase(curr) &&
        prev !== stripBase(prev) // prev phải CÓ dấu
      ) {
        continue;
      }
      result.push(curr);
    }
    text = result.join('');
  }

  return text;
}

// ============================================================
// Bước 5: Tra từ điển teencode — AN TOÀN
// Chỉ thay thế khi từ CHÍNH XÁC khớp (case-insensitive).
// ============================================================

// Regex tách dấu câu dính đầu/cuối từ
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
    // Tách dấu câu đầu/cuối
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

// ============================================================
// Pipeline chính
// ============================================================

/**
 * Chạy toàn bộ 5 bước chuẩn hóa.
 * Thứ tự: B1 NFC → B2 lookalike → B3 separator → B4 elongation → B5 dictionary
 * @param {string} text - Văn bản đầu vào
 * @returns {string} Văn bản đã chuẩn hóa
 */
export function normalize(text) {
  text = normalizeUnicode(text);         // B1
  text = mapLookalikeChars(text);        // B2
  text = removeInsertedSeparators(text);  // B3
  text = collapseElongation(text);       // B4
  text = restoreTeencode(text);          // B5
  return text;
}
