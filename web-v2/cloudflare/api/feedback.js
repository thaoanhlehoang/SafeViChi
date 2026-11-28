import { validateFeedback } from '../../src/feedback/contract.js';
import { ApiError } from './http.js';

export function getVietnamDate(now = Date.now()) {
  return new Date(now + 7 * 3600_000).toISOString().slice(0, 10);
}

async function hashRateLimitKey(secret, date, deviceId) {
  if (typeof secret !== 'string' || secret.length < 32) throw new ApiError(503, 'chua_cau_hinh');
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey('raw', encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const mac = await crypto.subtle.sign('HMAC', key, encoder.encode(date + ':' + deviceId));
  return Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, '0')).join('');
}

export async function saveFeedback(db, rateSecret, data) {
  const { level, submissionId, deviceId, feedback } = await validateFeedback(data);
  const date = getVietnamDate();
  const deviceKey = await hashRateLimitKey(rateSecret, date, deviceId);
  const isLevel2 = level === 'muc_2';
  const serializedFeedback = JSON.stringify(feedback);
  // changes() đọc số hàng từ lệnh INSERT ngay trước nó, trong cùng giao dịch D1.
  const [, inserted, , found] = await db.batch([
    db.prepare('DELETE FROM gioi_han WHERE ngay < ?').bind(date),
    db.prepare(`
      INSERT INTO phan_hoi (id, loai, ngay, du_lieu, trang_thai)
      SELECT ?, ?, ?, ?, ?
      WHERE coalesce((SELECT so_lan FROM gioi_han WHERE ngay = ? AND ma = ? AND loai = ?), 0) < ?
      ON CONFLICT (id) DO NOTHING RETURNING id
    `).bind(submissionId, level, date, serializedFeedback, isLevel2 ? 'cho_duyet' : null, date, deviceKey, level, isLevel2 ? 5 : 20),
    db.prepare(`
      INSERT INTO gioi_han (ngay, ma, loai, so_lan) SELECT ?, ?, ?, 1 WHERE changes() = 1
      ON CONFLICT (ngay, ma, loai) DO UPDATE SET so_lan = so_lan + 1
    `).bind(date, deviceKey, level),
    db.prepare('SELECT loai, du_lieu FROM phan_hoi WHERE id = ?').bind(submissionId),
  ]);
  const row = found.results[0];
  if (!row) throw new ApiError(429, 'qua_gioi_han');
  if (row.loai !== level || row.du_lieu !== serializedFeedback) throw new ApiError(409, 'trung_ma_lan_gui');
  return { id: submissionId, loai: level, da_luu: true, gui_lai: inserted.results.length === 0 };
}

const GROUP_FIELDS = {
  nhan: ['nhan_mo_hinh', 'nhan_de_xuat'],
  ly_do: ['ma_bao_cao', 'ma_ly_do', 'ma_chi_tiet'],
  khoang_diem: ['khoang_diem'],
  phien_ban: ['phien_ban_mo_hinh', 'phien_ban_nguong'],
};

function isValidDate(date) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return false;
  const millis = Date.parse(date);
  return Number.isFinite(millis) && new Date(millis).toISOString().slice(0, 10) === date;
}

export async function getFeedbackStats(db, url) {
  const { searchParams } = url;
  const groupBy = searchParams.get('theo') ?? 'nhan';
  const endDate = searchParams.get('den') ?? getVietnamDate();
  const startDate = searchParams.get('tu') ?? getVietnamDate(Date.now() - 6 * 86400_000);
  if (!Object.hasOwn(GROUP_FIELDS, groupBy) || !isValidDate(startDate) || !isValidDate(endDate) ||
      startDate > endDate || Date.parse(endDate) - Date.parse(startDate) > 90 * 86400_000) {
    throw new ApiError(400, 'bo_loc_khong_hop_le');
  }
  const fields = GROUP_FIELDS[groupBy];
  const select = fields.map((field) => "json_extract(du_lieu, '$." + field + "') AS " + field).join(', ');
  const [count, data] = await db.batch([
    db.prepare("SELECT count(*) AS n FROM phan_hoi WHERE loai = 'muc_1' AND ngay BETWEEN ? AND ?").bind(startDate, endDate),
    db.prepare(`
      SELECT ${select}, count(*) AS so_luot FROM phan_hoi
      WHERE loai = 'muc_1' AND ngay BETWEEN ? AND ?
      GROUP BY ${fields.join(', ')} ORDER BY so_luot DESC, ${fields.join(', ')} LIMIT 251
    `).bind(startDate, endDate),
  ]);
  return {
    theo: groupBy, tu: startDate, den: endDate, tong: count.results[0].n,
    con_nhom: data.results.length > 250, nhom: data.results.slice(0, 250),
  };
}
