import { redactPersonalInfo } from '../../src/feedback/redaction.js';
import { isShortText, assertValid, assertAllowedFields, isValidCode } from '../../src/feedback/contract.js';
import { BASELINE_THRESHOLD_VERSION, isValidThresholds } from '../../src/feedback/thresholds.js';
import { ApiError } from './http.js';

function validateMetrics(metrics) {
  assertAllowedFields(metrics, ['macro_f1', 'fpr_an_toan']);
  assertValid([metrics.macro_f1, metrics.fpr_an_toan].every((n) => Number.isFinite(n) && n >= 0 && n <= 1), 'so_do');
}

export async function createProposal(db, actor, data) {
  assertAllowedFields(data, [
    'phien_ban', 'phien_ban_mo_hinh', 'phien_ban_goc', 'nguong', 'ly_do', 'validation',
  ]);
  const { phien_ban: version, phien_ban_mo_hinh: modelVersion, phien_ban_goc: baseVersion, nguong: thresholds, ly_do: reason, validation } = data;
  assertValid(isValidCode(version) && version !== BASELINE_THRESHOLD_VERSION, 'phien_ban');
  assertValid(isValidCode(modelVersion), 'phien_ban_mo_hinh');
  assertValid(isValidCode(baseVersion) && baseVersion !== version, 'phien_ban_goc');
  assertValid(isValidThresholds(thresholds), 'nguong');
  assertValid(isShortText(reason, 480) && redactPersonalInfo(reason) === reason, 'ly_do');
  assertAllowedFields(validation, ['tap', 'so_mau', 'truoc', 'sau']);
  assertValid(isValidCode(validation.tap) && Number.isSafeInteger(validation.so_mau) && validation.so_mau > 0, 'validation');
  validateMetrics(validation.truoc);
  validateMetrics(validation.sau);
  const id = crypto.randomUUID();
  const { results } = await db.prepare(`
    INSERT INTO de_xuat (id, phien_ban, phien_ban_mo_hinh, nguoi_de_xuat, du_lieu)
    VALUES (?, ?, ?, ?, ?) ON CONFLICT (phien_ban_mo_hinh, phien_ban) DO NOTHING RETURNING id
  `).bind(id, version, modelVersion, actor, JSON.stringify({ phien_ban_goc: baseVersion, nguong: thresholds, ly_do: reason, validation })).all();
  if (!results.length) throw new ApiError(409, 'trung_phien_ban');
  return { id, trang_thai: 'cho_phe_duyet' };
}

export async function getProposal(db, id) {
  const row = await db.prepare('SELECT * FROM de_xuat WHERE id = ?').bind(id).first();
  if (!row) throw new ApiError(404, 'khong_tim_thay_de_xuat');
  const { du_lieu: storedData, ...fields } = row;
  return { ...fields, ...JSON.parse(storedData), trang_thai: row.nguoi_phe_duyet ? 'da_duyet' : 'cho_phe_duyet' };
}

export async function listPendingProposals(db) {
  const { results } = await db.prepare(`
    SELECT id, phien_ban, phien_ban_mo_hinh, nguoi_de_xuat FROM de_xuat
    WHERE nguoi_phe_duyet IS NULL ORDER BY rowid DESC LIMIT 100
  `).all();
  return { de_xuat: results };
}

export async function approveProposal(db, actor, id, data) {
  assertAllowedFields(data, ['ly_do']);
  assertValid(isShortText(data.ly_do, 480) && redactPersonalInfo(data.ly_do) === data.ly_do, 'ly_do');
  const { results } = await db.prepare(`
    UPDATE de_xuat AS dx SET nguoi_phe_duyet = ?, ly_do_phe_duyet = ?
    WHERE id = ? AND nguoi_phe_duyet IS NULL AND json_extract(du_lieu, '$.phien_ban_goc') = coalesce((
      SELECT d.phien_ban FROM nguong_hieu_luc n JOIN de_xuat d ON d.id = n.de_xuat_id
      WHERE n.phien_ban_mo_hinh = dx.phien_ban_mo_hinh
    ), ?)
    RETURNING id
  `).bind(actor, data.ly_do, id, BASELINE_THRESHOLD_VERSION).all();
  if (!results.length) {
    await getProposal(db, id);
    throw new ApiError(409, 'de_xuat_khong_con_hop_le');
  }
  return { id, trang_thai: 'da_duyet' };
}

export async function getActiveThreshold(db, model) {
  assertValid(isValidCode(model), 'phien_ban_mo_hinh');
  const row = await db.prepare(`
    SELECT d.phien_ban, d.phien_ban_mo_hinh, json_extract(d.du_lieu, '$.nguong') AS nguong
    FROM nguong_hieu_luc n JOIN de_xuat d ON d.id = n.de_xuat_id WHERE n.phien_ban_mo_hinh = ?
  `).bind(model).first();
  return { cau_hinh: row ? { ...row, nguong: JSON.parse(row.nguong) } : null };
}
