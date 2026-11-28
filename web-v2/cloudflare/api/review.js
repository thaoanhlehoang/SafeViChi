import { assertValid, assertAllowedFields, LABEL_CODES, UUID_PATTERN } from '../../src/feedback/contract.js';
import { ApiError } from './http.js';

export async function listReviewQueue(db, actor, params) {
  const date = params.get('sau_ngay') ?? '';
  const id = params.get('sau_id') ?? '';
  assertValid((date === '' && id === '') || (/^\d{4}-\d{2}-\d{2}$/.test(date) && UUID_PATTERN.test(id)), 'trang');
  // Chỉ trả thông tin danh sách. Nội dung mẫu đi qua endpoint mở mẫu có nhật ký.
  const { results } = await db.prepare(`
    SELECT p.id, p.ngay, p.trang_thai,
      json_extract(p.du_lieu, '$.nhan_mo_hinh') AS nhan_mo_hinh,
      json_extract(p.du_lieu, '$.nhan_de_xuat') AS nhan_de_xuat
    FROM phan_hoi p
    WHERE p.loai = 'muc_2' AND p.trang_thai <> 'da_duyet' AND (p.ngay, p.id) > (?, ?)
      AND NOT EXISTS (SELECT 1 FROM luot_duyet WHERE mau_id = p.id AND nguoi = ?)
    ORDER BY p.ngay, p.id LIMIT 26
  `).bind(date, id, actor).all();
  return { hang_doi: results.slice(0, 25), con_mau: results.length > 25 };
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
  assertAllowedFields(data, ['nhan', 'bang_chung', 'nham_nhom']);
  const { nhan: label, bang_chung: evidenceSpans = [], nham_nhom: targetsGroup = false } = data;
  assertValid(LABEL_CODES.includes(label), 'nhan');
  assertValid(typeof targetsGroup === 'boolean' && (!targetsGroup || label === 'XUC_PHAM'), 'nham_nhom');
  assertValid(Array.isArray(evidenceSpans) && evidenceSpans.length <= 16 &&
    (label !== 'AN_TOAN' || evidenceSpans.length === 0), 'bang_chung');
  const sample = await db.prepare("SELECT du_lieu FROM phan_hoi WHERE id = ? AND loai = 'muc_2'").bind(id).first();
  if (!sample) throw new ApiError(404, 'khong_tim_thay_mau');
  const feedback = JSON.parse(sample.du_lieu);
  const evidence = extractEvidence(feedback.doan_da_che, evidenceSpans);

  const requiresSecondReviewer = targetsGroup || [label, feedback.nhan_mo_hinh, feedback.nhan_de_xuat].includes('DE_DOA') ||
    feedback.ma_chi_tiet === 'miet_thi_nhom';
  const [saved, current] = await db.batch([
    db.prepare(`
      INSERT INTO luot_duyet (id, mau_id, nguoi, nhan, nham_nhom, can_duyet_hai, bang_chung)
      SELECT ?, id, ?, ?, ?, CASE WHEN ? OR EXISTS (
        SELECT 1 FROM luot_duyet WHERE mau_id = phan_hoi.id AND can_duyet_hai = 1
      ) THEN 1 ELSE 0 END, ?
      FROM phan_hoi WHERE id = ? AND loai = 'muc_2' AND trang_thai <> 'da_duyet'
      ON CONFLICT (mau_id, nguoi) DO NOTHING RETURNING id
    `).bind(crypto.randomUUID(), actor, label, Number(targetsGroup), Number(requiresSecondReviewer), JSON.stringify(evidence), id),
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
