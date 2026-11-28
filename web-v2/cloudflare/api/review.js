import { assertValid, assertAllowedFields, LABEL_CODES, UUID_PATTERN } from '../../src/feedback/contract.js';
import { ApiError } from './http.js';

// Quyết định 28/11/2026: một người duyệt, một lượt duyệt là chốt. Mẫu cũ còn ở cho_nguoi_thu_hai / bat_dong
// được gộp vào tab chờ duyệt và chốt ở lượt duyệt kế tiếp.
export const QUEUE_TABS = ['cho_duyet', 'da_duyet'];

// Mẫu thuộc một tab với người đang xem: tab đang xử lý chỉ gồm mẫu họ chưa duyệt; tab đã duyệt gồm mọi mẫu.
const IN_TAB = `p.loai = 'muc_2' AND (p.trang_thai = 'da_duyet') = (?1 = 'da_duyet')
  AND (?1 = 'da_duyet' OR NOT EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = p.id AND nguoi = ?2))`;

export async function listReviewQueue(db, actor, params) {
  const tab = params.get('trang_thai') ?? 'cho_duyet';
  assertValid(QUEUE_TABS.includes(tab), 'trang_thai');
  const date = params.get('sau_ngay') ?? '';
  const id = params.get('sau_id') ?? '';
  assertValid((date === '' && id === '') || (/^\d{4}-\d{2}-\d{2}$/.test(date) && UUID_PATTERN.test(id)), 'trang');
  // Chỉ trả thông tin danh sách (mã lựa chọn, không có đoạn hội thoại hay mô tả tự do).
  // Nội dung mẫu đi qua endpoint mở mẫu có nhật ký.
  const [list, counts] = await db.batch([
    db.prepare(`
      SELECT * FROM (
        SELECT p.id, p.ngay, p.trang_thai,
          json_extract(p.du_lieu, '$.nhan_mo_hinh') AS nhan_mo_hinh,
          json_extract(p.du_lieu, '$.nhan_de_xuat') AS nhan_de_xuat,
          json_extract(p.du_lieu, '$.ma_bao_cao') AS ma_bao_cao,
          json_extract(p.du_lieu, '$.ma_ly_do') AS ma_ly_do,
          json_extract(p.du_lieu, '$.ma_chi_tiet') AS ma_chi_tiet
        FROM phan_hoi p WHERE ${IN_TAB}
      ) WHERE (ngay, id) > (?3, ?4)
      ORDER BY ngay, id LIMIT 26
    `).bind(tab, actor, date, id),
    db.prepare(`
      SELECT CASE p.trang_thai WHEN 'da_duyet' THEN 'da_duyet' ELSE 'cho_duyet' END AS trang_thai, count(*) AS n
      FROM phan_hoi p
      WHERE p.loai = 'muc_2' AND (p.trang_thai = 'da_duyet'
        OR NOT EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = p.id AND nguoi = ?))
      GROUP BY 1
    `).bind(actor),
  ]);
  const dem = Object.fromEntries(QUEUE_TABS.map((key) => [key, 0]));
  for (const row of counts.results) dem[row.trang_thai] = row.n;
  return { trang_thai: tab, dem, hang_doi: list.results.slice(0, 25), con_mau: list.results.length > 25 };
}

export async function openReviewSample(db, actor, id) {
  const selection = id
    ? "SELECT id FROM phan_hoi WHERE loai = 'muc_2' AND id = ?"
    : `
        SELECT p.id FROM phan_hoi p
        WHERE p.loai = 'muc_2' AND p.trang_thai <> 'da_duyet'
          AND NOT EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = p.id AND nguoi = ?)
        ORDER BY p.ngay, p.id LIMIT 1
      `;
  const [, query] = await db.batch([
    db.prepare(`
      INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong)
      SELECT ?, 'mo_mau', id FROM (${selection})
    `).bind(actor, id ?? actor),
    db.prepare(`
      SELECT p.id, p.ngay, p.trang_thai, p.du_lieu, d.nhan, d.bang_chung,
        EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = p.id AND nguoi = ?) AS da_gui_luot_duyet
      FROM phan_hoi p LEFT JOIN luot_duyet d ON d.id = p.quyet_dinh_id
      WHERE p.id = (${selection})
    `).bind(actor, id ?? actor),
  ]);
  const row = query.results[0];
  if (!row) {
    if (id) throw new ApiError(404, 'khong_tim_thay_mau');
    return { mau: null };
  }
  const decision = row.nhan === null ? null : {
    nhan: row.nhan, bang_chung: JSON.parse(row.bang_chung), cach_gan: 'nguoi', nguon: 'phan_hoi',
  };
  return { mau: { id: row.id, ngay: row.ngay, trang_thai: row.trang_thai,
    ...JSON.parse(row.du_lieu), quyet_dinh: decision, da_gui_luot_duyet: Boolean(row.da_gui_luot_duyet) } };
}

function extractEvidence(redactedConversation, spans) {
  const lines = redactedConversation.split('\n');
  return spans.map((span) => {
    assertAllowedFields(span, ['dong', 'start', 'end']);
    const { dong: lineIndex, start, end } = span;
    const line = lines[lineIndex];
    assertValid(Number.isInteger(lineIndex) && typeof line === 'string' &&
      Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start && end <= line.length &&
      !/[\uDC00-\uDFFF]/u.test(line.charAt(start)) && !/[\uDC00-\uDFFF]/u.test(line.charAt(end)), 'bang_chung');
    const text = line.slice(start, end);
    assertValid(text.trim().length > 0, 'bang_chung');
    return { dong: lineIndex, start, end, text };
  });
}

export async function saveReview(db, actor, id, data) {
  assertAllowedFields(data, ['nhan', 'bang_chung', 'nham_nhom', 'ghi_chu']);
  const { nhan: label, bang_chung: evidenceSpans = [], nham_nhom: targetsGroup = false, ghi_chu: rawNote = null } = data;
  assertValid(rawNote === null || (typeof rawNote === 'string' && Array.from(rawNote.trim()).length <= 200), 'ghi_chu');
  const note = rawNote?.trim() || null;
  assertValid(LABEL_CODES.includes(label), 'nhan');
  assertValid(typeof targetsGroup === 'boolean' && (!targetsGroup || label === 'XUC_PHAM'), 'nham_nhom');
  assertValid(Array.isArray(evidenceSpans) && evidenceSpans.length <= 16 &&
    (label !== 'AN_TOAN' || evidenceSpans.length === 0), 'bang_chung');
  const sample = await db.prepare("SELECT du_lieu FROM phan_hoi WHERE id = ? AND loai = 'muc_2'").bind(id).first();
  if (!sample) throw new ApiError(404, 'khong_tim_thay_mau');
  const feedback = JSON.parse(sample.du_lieu);
  const evidence = extractEvidence(feedback.doan_da_che, evidenceSpans);

  const [saved, current] = await db.batch([
    db.prepare(`
      INSERT INTO luot_duyet (id, mau_id, nguoi, nhan, nham_nhom, can_duyet_hai, bang_chung, luc, ghi_chu)
      SELECT ?, id, ?, ?, ?, 0, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?
      FROM phan_hoi WHERE id = ? AND loai = 'muc_2' AND trang_thai <> 'da_duyet'
      ON CONFLICT (mau_id, nguoi) DO NOTHING RETURNING id
    `).bind(crypto.randomUUID(), actor, label, Number(targetsGroup), JSON.stringify(evidence), note, id),
    db.prepare('SELECT trang_thai FROM phan_hoi WHERE id = ?').bind(id),
  ]);
  if (!saved.results.length) throw new ApiError(409, 'khong_the_duyet_lai');
  return { id, trang_thai: current.results[0].trang_thai };
}

export async function getAuditLog(db, subjectId) {
  const query = db.prepare(`
    SELECT luc, nguoi, hanh_dong, doi_tuong, chi_tiet FROM nhat_ky
    ${subjectId ? 'WHERE doi_tuong = ?' : ''} ORDER BY id DESC LIMIT 100
  `);
  const { results } = await (subjectId ? query.bind(subjectId) : query).all();
  return { nhat_ky: results };
}

// Người duyệt che thêm thông tin cá nhân còn sót trong bản đã che. Chỉ làm được khi mẫu chưa có lượt duyệt nào,
// để vị trí bằng chứng của các lượt duyệt không bị lệch. Ghi nhật ký "che_them".
export async function maskMore(db, actor, id, data) {
  assertAllowedFields(data, ['dong', 'start', 'end']);
  const sample = await db.prepare("SELECT du_lieu FROM phan_hoi WHERE id = ? AND loai = 'muc_2'").bind(id).first();
  if (!sample) throw new ApiError(404, 'khong_tim_thay_mau');
  const feedback = JSON.parse(sample.du_lieu);
  const [span] = extractEvidence(feedback.doan_da_che, [data]);
  const lines = feedback.doan_da_che.split('\n');
  const line = lines[span.dong];
  lines[span.dong] = line.slice(0, span.start) + '[ĐÃ_CHE]' + line.slice(span.end);
  const updated = JSON.stringify({ ...feedback, doan_da_che: lines.join('\n') });
  const [changed] = await db.batch([
    db.prepare(`
      UPDATE phan_hoi SET du_lieu = ? WHERE id = ? AND du_lieu = ? AND trang_thai = 'cho_duyet'
        AND NOT EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = phan_hoi.id)
      RETURNING id
    `).bind(updated, id, sample.du_lieu),
    db.prepare(`
      INSERT INTO nhat_ky (nguoi, hanh_dong, doi_tuong, chi_tiet)
      SELECT ?, 'che_them', ?, ? WHERE changes() = 1
    `).bind(actor, id, 'lượt ' + (span.dong + 1)),
  ]);
  if (!changed.results.length) throw new ApiError(409, 'khong_the_che_them');
  return { id, doan_da_che: lines.join('\n') };
}
