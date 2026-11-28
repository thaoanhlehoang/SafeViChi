// Che / ẩn danh hóa thông tin cá nhân (PII) trên thiết bị – SafeViChi v3.
//
// Chạy hoàn toàn trong trình duyệt (hoặc Node), thuần JS: không đụng DOM, không gọi mạng, không ghi
// localStorage. Mục đích: trước khi người dùng gửi phản hồi mức 2 (kèm đoạn tin), thay các thông tin cá nhân
// bằng nhãn giữ chỗ để không byte nào chứa PII rời máy. Đây là CHE / ẨN DANH HÓA một chiều, KHÔNG phải mã hóa:
// không có khóa, không có bảng ánh xạ, không khôi phục được bản gốc.
//
// Nhãn giữ chỗ theo mục 3.3 của docs/v3/KE_HOACH_CHANG_2.md:
//   số điện thoại → [SĐT]      email → [EMAIL]          số tài khoản / thẻ → [STK]
//   OTP / mã xác minh → [MÃ]   CCCD / CMND → [GIẤY_TỜ]  địa chỉ → [ĐỊA_CHỈ]
//   tên người → [TÊN_1], [TÊN_2]… (cùng tên → cùng số, chỉ trong một lần gọi)
//   đường link → [LINK:ten-mien] : giữ tên miền (cần để nhận ra LUA_DAO), bỏ đường dẫn và tham số, ví dụ
//     "https://vietcombank-xacminh.com/login?id=123" → "[LINK:vietcombank-xacminh.com]".
//     Link chỉ có tên miền (không đường dẫn, không tham số) được giữ nguyên vì không có gì để che.
//   đoạn người dùng tự che thêm ở bước xem trước → [ĐÃ_CHE]
//
// Giữ nguyên số tiền ("500k", "2 triệu", "1.500.000đ", "3tr", "15000000 vnd"): cần để nhận ra lừa đảo.
//
// Giới hạn đã biết (vì vậy màn xem trước + che thêm là lớp phòng thủ cuối cùng):
//   - số viết bằng chữ ("không chín một hai…"), tên viết thường hoặc viết hoa toàn bộ, tên không đứng sau
//     từ xưng hô / giới thiệu, địa chỉ không có số nhà đều có thể bị sót;
//   - đầu vào nên ở dạng Unicode NFC (dạng gõ phím thông thường); vị trí start/end tính theo đơn vị UTF-16
//     của chuỗi gốc.

/** Nhãn giữ chỗ cho từng loại. Đổi ở đây nếu cần đổi cách hiển thị. */
export const NHAN_CHE = Object.freeze({
  SDT: '[SĐT]',
  EMAIL: '[EMAIL]',
  STK: '[STK]',
  MA: '[MÃ]',
  GIAY_TO: '[GIẤY_TỜ]',
  DIA_CHI: '[ĐỊA_CHỈ]',
  THU_CONG: '[ĐÃ_CHE]',
  // TEN và LINK có phần biến đổi: [TÊN_n], [LINK:ten-mien]
});

// ---------------------------------------------------------------------------
// Chuẩn hóa phụ: bỏ dấu từng ký tự, GIỮ NGUYÊN độ dài để vị trí khớp với văn bản gốc.

function boDauGiuDoDai(text) {
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === 'đ') { out += 'd'; continue; }
    if (c === 'Đ') { out += 'D'; continue; }
    const d = c.normalize('NFD').replace(/[̀-ͯ]/g, '');
    out += d.length === 1 ? d : c;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Từ khóa ngữ cảnh (so trên bản bỏ dấu, chữ thường).

const BANK = 'vietcombank|vcb|techcombank|tcb|bidv|agribank|vietinbank|acb|mbbank|mb|tpbank|vpbank|' +
  'sacombank|hdbank|shb|ocb|vib|scb|eximbank|seabank|msb|momo|zalopay|viettelpay';

const TU_KHOA_SO = [
  { loai: 'MA', re: /(?:^|[^a-z0-9])(otp|ma|code|pin|ma xac minh|ma xac nhan)(?=[^a-z0-9]|$)/g },
  { loai: 'GIAY_TO', re: /(?:^|[^a-z0-9])(cccd|cmnd|cmt|can cuoc|chung minh|cong dan)(?=[^a-z0-9]|$)/g },
  { loai: 'STK', re: new RegExp(`(?:^|[^a-z0-9])(stk|so tk|tk|tai khoan|ngan hang|atm|bank|${BANK})(?=[^a-z0-9]|$)`, 'g') },
  { loai: 'SDT', re: /(?:^|[^a-z0-9])(sdt|sđt|so dt|dt|dien thoai|phone|zalo|hotline|lh|lien he|goi|sim)(?=[^a-z0-9]|$)/g },
];

// Khoảng độ dài (số chữ số) hợp lệ khi có từ khóa đứng trước.
const DO_DAI_THEO_TU_KHOA = {
  MA: (n) => n >= 4 && n <= 8,
  GIAY_TO: (n) => n === 9 || n === 12,
  STK: (n) => n >= 6 && n <= 19,
  SDT: (n) => n >= 9 && n <= 12,
};

// Đơn vị tiền: số đứng ngay trước các đơn vị này là số tiền, phải giữ nguyên.
const RE_TIEN = /(?<![0-9a-z])\$?\d+(?:[.,]\d+)*[ ]?(?:k|tr|trieu|nghin|ngan|d|vnd|dong|usd|ty|ti|cu|xu|lit|%)(?![a-z0-9])/g;

// Chuỗi chữ số, cho phép một dấu cách / chấm / gạch giữa các chữ số ("0912 345 678", "0987.654.321").
const RE_CHUOI_SO = /(?<![0-9])\+?\d(?:[ .\-]?\d)*/g;

// ---------------------------------------------------------------------------
// Địa chỉ (heuristic): số nhà + (tên đường viết hoa) + các thành phần đường/phố/ngõ/hẻm/phường/xã/quận/huyện…
// So trên bản bỏ dấu nhưng GIỮ HOA/THƯỜNG để nhận ra tên riêng.

function hoaChuDau(words) {
  return words.map((w) => `[${w[0]}${w[0].toUpperCase()}]${w.slice(1)}`).join('|');
}
const TU_DIA_CHI = hoaChuDau([
  'duong', 'pho', 'ngo', 'ngach', 'hem', 'kiet', 'phuong', 'xa', 'quan', 'huyen', 'thi xa', 'thi tran',
  'tinh', 'thanh pho', 'thon', 'ap', 'khu pho', 'to dan pho',
]) + '|TP|Tp|tp|P|p|Q|q';
const CAP = '[A-Z][a-z]*';
const CUM_HOA = `${CAP}(?:[ ]${CAP})*`;
const TEN_THANH_PHAN = `(?:${CAP}|\\d{1,3}[A-Za-z]?)(?:[ ](?:${CAP}|\\d{1,3}[A-Za-z]?))*`;
const THANH_PHAN = `(?:(?:${TU_DIA_CHI})(?:\\.[ ]?|[ ]+)${TEN_THANH_PHAN})`;
const SO_NHA = '(?:(?:[Ss]o|[Ss]n)[ ]+)?\\d{1,4}[A-Za-z]?(?:\\/\\d{1,4}[A-Za-z]?)*';
const RE_DIA_CHI = new RegExp(
  `(?<![A-Za-z0-9\\/])${SO_NHA}[ ]*,?[ ]*(?:${THANH_PHAN}|${CUM_HOA}[ ]*,?[ ]*${THANH_PHAN})` +
  `(?:[ ]*,?[ ]*(?:${THANH_PHAN}|${CUM_HOA}))*`,
  'g',
);

// ---------------------------------------------------------------------------
// Tên người: cụm từ viết hoa đứng sau từ xưng hô / giới thiệu.

const TU_GOI = new Set([
  'anh', 'chi', 'em', 'ban', 'co', 'chu', 'thang', 'con', 'ong', 'ba', 'bac', 'cau', 'di', 'me', 'bo',
  'a', 'c', 'e', 'ad', 'sep', 'thay',
]);
const RE_TRUOC_TEN = /(?:^|[^a-z0-9])(ten la|ten|ten toi la|toi la|minh la|em la|tao la|tui la|to la|gioi thieu|anh|chi|em|ban|co|chu|thang|con|ong|ba|bac|cau|di|me|bo|a|c|e|ad|sep|thay)[ ]+$/;
// Từ viết hoa thường gặp sau đại từ nhưng không phải tên (thán từ, từ chửi, tên ngân hàng / dịch vụ…).
const KHONG_PHAI_TEN = new Set([
  'oi', 'a', 'ah', 'nhe', 'nha', 'ok', 'oke', 'alo', 'da', 'vang', 'u', 'uh', 'thi', 'la', 'va', 'voi',
  'ngu', 'dien', 'khung', 'oc', 'cho', 'lon', 'dot', 'hen', 'beo', 'xau', 'kho', 'mat', 'bi',
  // Không đưa vào đây các từ cũng là tên phổ biến (Minh, Mây, Hồ, Chung…).
  'em', 'anh', 'chi', 'toi', 'ban', 'tao', 'no', 'cac', 'moi', 'nguoi',
  'khong', 'co', 'duoc', 'roi', 'luon', 'nay', 'kia', 'do', 'day', 'gi', 'sao', 'nao',
  ...BANK.split('|'), 'zalo', 'facebook', 'fb', 'shopee', 'lazada', 'tiktok', 'grab', 'otp',
]);

// ---------------------------------------------------------------------------
// Link.

const RE_LINK = /(?:https?:\/\/[^\s]+|www\.[^\s]+|(?<![\w@.\-])(?:[a-z0-9\-]+\.)+[a-z]{2,}\/[^\s]*)/gi;
const RE_EMAIL = /(?<![\w.+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}/g;

// ---------------------------------------------------------------------------

function chongLan(spans, s, e) {
  return spans.some((x) => s < x.end && e > x.start);
}

/**
 * Che / ẩn danh hóa thông tin cá nhân trong một đoạn tin.
 *
 * @param {string} text Văn bản gốc (có thể nhiều dòng, ngăn bằng "\n"; "\n" luôn được giữ nguyên).
 * @param {{cheThem?: Array<{start:number,end:number}>}} [opts]
 *   cheThem: các đoạn (vị trí theo văn bản gốc) người dùng tự bôi đen để che thêm ở bước xem trước.
 *   Chỉ được che thêm: không có cách nào bỏ che một đoạn tự động đã che.
 * @returns {{text: string, cac_doan: Array<{loai: string, start: number, end: number}>}}
 *   text: bản đã che; cac_doan: vị trí các đoạn đã che trên văn bản gốc (không chứa nội dung gốc).
 */
export function cheThongTin(text, { cheThem = [] } = {}) {
  if (typeof text !== 'string' || text.length === 0) return { text: text ?? '', cac_doan: [] };

  const fold = boDauGiuDoDai(text); // bỏ dấu, giữ hoa/thường
  const low = fold.toLowerCase(); // bỏ dấu, chữ thường
  const spans = []; // {loai, start, end, nhan?, khoa?}
  const them = (loai, start, end, extra = {}) => {
    if (end <= start || chongLan(spans, start, end)) return false;
    spans.push({ loai, start, end, ...extra });
    return true;
  };

  // 1. Email (trước link để "@ten-mien" không bị coi là link).
  for (const m of fold.matchAll(RE_EMAIL)) them('EMAIL', m.index, m.index + m[0].length);

  // 2. Link: giữ tên miền, che đường dẫn và tham số.
  for (const m of text.matchAll(RE_LINK)) {
    let url = m[0].replace(/[.,;:!?)\]}"'»”]+$/, '');
    const start = m.index;
    const end = start + url.length;
    const host = url.replace(/^https?:\/\//i, '').split(/[/?#]/)[0];
    const conLai = url.slice(url.indexOf(host) + host.length);
    if (!host || conLai.replace(/^\/$/, '') === '') continue; // chỉ có tên miền: không có gì để che
    them('LINK', start, end, { nhan: `[LINK:${host.toLowerCase()}]` });
  }

  // 3. Số tiền: đánh dấu vùng bảo vệ, không che.
  const tien = [];
  for (const m of low.matchAll(RE_TIEN)) tien.push({ start: m.index, end: m.index + m[0].length });

  // 4. Chuỗi số: phân loại theo từ khóa đứng trước (cùng dòng), rồi theo hình dạng.
  let hetTruoc = 0;
  for (const m of fold.matchAll(RE_CHUOI_SO)) {
    const start = m.index;
    const end = start + m[0].length;
    const digits = m[0].replace(/\D/g, '');
    const n = digits.length;
    const truocDo = hetTruoc;
    hetTruoc = end;
    if (chongLan(tien, start, end) || chongLan(spans, start, end)) continue;

    // Cửa sổ ngữ cảnh: tối đa 30 ký tự trước số, cùng dòng, sau chuỗi số trước đó.
    const dauDong = low.lastIndexOf('\n', start - 1) + 1;
    const ws = Math.max(dauDong, truocDo, start - 30);
    const ctx = low.slice(ws, start);
    let tuKhoa = null;
    let viTri = -1;
    for (const { loai, re } of TU_KHOA_SO) {
      re.lastIndex = 0;
      for (const k of ctx.matchAll(re)) {
        const ketThuc = k.index + k[0].length;
        if (ketThuc > viTri) { viTri = ketThuc; tuKhoa = loai; }
      }
    }

    let loai = null;
    if (tuKhoa && DO_DAI_THEO_TU_KHOA[tuKhoa](n)) loai = tuKhoa;
    else if ((digits[0] === '0' && (n === 10 || n === 11)) || (m[0][0] === '+' && digits.startsWith('84') && n === 11)) loai = 'SDT';
    else if (digits.startsWith('84') && n === 11) loai = 'SDT';
    else if (n === 12 || n === 9) loai = 'GIAY_TO';
    else if (n >= 10 && n <= 19) loai = 'STK';
    if (loai) them(loai, start, end);
  }

  // 5. Địa chỉ.
  for (const m of fold.matchAll(RE_DIA_CHI)) {
    const s = m[0].replace(/[ ,]+$/, '');
    them('DIA_CHI', m.index, m.index + s.length);
  }

  // 6. Tên người.
  const ten = []; // {start, end, khoa}
  const themTen = (start, end) => {
    if (chongLan(spans, start, end) || chongLan(ten, start, end)) return;
    ten.push({ start, end, khoa: low.slice(start, end) });
  };
  // 6a. @nhắc tên
  for (const m of fold.matchAll(/(?<![A-Za-z0-9.])@([A-Za-z][A-Za-z0-9_]*)/g)) {
    if (!KHONG_PHAI_TEN.has(m[1].toLowerCase())) themTen(m.index + 1, m.index + m[0].length);
  }
  // 6b. Cụm viết hoa sau từ xưng hô / giới thiệu.
  for (const m of fold.matchAll(/(?<![A-Za-z0-9@])[A-Z][a-z]+(?:[ ][A-Z][a-z]+){0,4}/g)) {
    const words = [];
    let pos = m.index;
    for (const w of m[0].split(' ')) { words.push({ w, start: pos, end: pos + w.length }); pos += w.length + 1; }
    const dauDong = low.lastIndexOf('\n', m.index - 1) + 1;
    let duocGoi = RE_TRUOC_TEN.test(low.slice(Math.max(dauDong, m.index - 20), m.index));
    // Đầu câu: "Chị Lan ơi", "Anh Tuấn" – bỏ các từ xưng hô viết hoa ở đầu cụm.
    while (words.length && TU_GOI.has(words[0].w.toLowerCase())) { words.shift(); duocGoi = true; }
    if (!duocGoi) continue;
    const cum = [];
    for (const x of words) {
      if (KHONG_PHAI_TEN.has(x.w.toLowerCase())) break;
      cum.push(x);
    }
    if (cum.length) themTen(cum[0].start, cum[cum.length - 1].end);
  }
  // 6c. Cùng một tên xuất hiện lại ở chỗ khác (không có từ xưng hô) cũng được che.
  const daBiet = [...new Set(ten.map((t) => t.khoa))];
  for (const khoa of daBiet) {
    const re = new RegExp(`(?<![a-z0-9])${khoa.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![a-z0-9])`, 'g');
    for (const m of low.matchAll(re)) {
      if (/[A-Z]/.test(fold[m.index])) themTen(m.index, m.index + m[0].length);
    }
  }
  // Đánh số theo thứ tự xuất hiện; tên một từ trùng từ cuối của tên đầy đủ đã gặp → cùng số.
  // Bảng đánh số chỉ là biến cục bộ, không trả về, không lưu.
  ten.sort((a, b) => a.start - b.start);
  const soCuaKhoa = new Map();
  let dem = 0;
  for (const t of ten) {
    let so = soCuaKhoa.get(t.khoa);
    if (so === undefined && !t.khoa.includes(' ')) {
      for (const [k, v] of soCuaKhoa) if (k.split(' ').pop() === t.khoa) { so = v; break; }
    }
    if (so === undefined) { so = ++dem; soCuaKhoa.set(t.khoa, so); }
    spans.push({ loai: 'TEN', start: t.start, end: t.end, nhan: `[TÊN_${so}]` });
  }
  soCuaKhoa.clear();

  // 7. Che thêm thủ công: cắt theo dòng (không nuốt "\n"), gộp với các đoạn đã che nếu chồng lấn.
  //    Không bao giờ bỏ che: đoạn tự động chỉ có thể bị gộp vào đoạn lớn hơn.
  let ketQua = spans.slice();
  for (const d of Array.isArray(cheThem) ? cheThem : []) {
    let s = Math.max(0, Math.floor(Number(d?.start)));
    const e = Math.min(text.length, Math.floor(Number(d?.end)));
    if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) continue;
    while (s < e) {
      const nl = text.indexOf('\n', s);
      const ee = nl === -1 || nl >= e ? e : nl;
      if (ee > s) ketQua = gopThuCong(ketQua, s, ee);
      s = ee + 1;
    }
  }

  // 8. Ghép văn bản đã che.
  ketQua.sort((a, b) => a.start - b.start);
  let out = '';
  let i = 0;
  for (const sp of ketQua) {
    out += text.slice(i, sp.start) + (sp.nhan ?? NHAN_CHE[sp.loai]);
    i = sp.end;
  }
  out += text.slice(i);
  return { text: out, cac_doan: ketQua.map(({ loai, start, end }) => ({ loai, start, end })) };
}

function gopThuCong(spans, s, e) {
  const bao = spans.find((x) => x.start <= s && x.end >= e);
  if (bao) return spans; // nằm gọn trong đoạn đã che: không cần làm gì
  let start = s;
  let end = e;
  const giu = [];
  for (const x of spans) {
    if (x.start < end && x.end > start) { start = Math.min(start, x.start); end = Math.max(end, x.end); }
    else giu.push(x);
  }
  // Lặp lại vì vùng gộp có thể vừa chạm thêm đoạn khác.
  const conChong = giu.filter((x) => x.start < end && x.end > start);
  if (conChong.length) return gopThuCong(giu, start, end);
  giu.push({ loai: 'THU_CONG', start, end });
  return giu;
}
