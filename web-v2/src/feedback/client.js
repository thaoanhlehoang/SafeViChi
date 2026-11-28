import { redactPersonalInfo } from './redaction.js';
import { assertValid, validateFeedback, UUID_PATTERN, CONSENT_VERSION, sha256 } from './contract.js';
import { recordFalsePositive } from './warning-preferences.js';

const DEVICE_ID_KEY = 'safevichi.ma_thiet_bi';
const LEVEL_2_SETTING_KEY = 'safevichi.cho_phep_muc_2';
const PREVIEW_SNAPSHOTS = new WeakSet();

export function getDeviceId(storage = localStorage) {
  let deviceId = storage.getItem(DEVICE_ID_KEY);
  if (!UUID_PATTERN.test(deviceId ?? '')) {
    deviceId = crypto.randomUUID();
    storage.setItem(DEVICE_ID_KEY, deviceId);
  }
  return deviceId;
}

export function isLevel2Enabled(storage = localStorage) {
  return storage.getItem(LEVEL_2_SETTING_KEY) !== 'tat';
}

export function setLevel2Enabled(enabled, storage = localStorage) {
  assertValid(typeof enabled === 'boolean', 'cho_phep_muc_2');
  storage.setItem(LEVEL_2_SETTING_KEY, enabled ? 'bat' : 'tat');
}

function makePreviewSnapshot(text) {
  assertValid(Array.from(text).length <= 4800, 'doan_da_che');
  const preview = Object.freeze({ text });
  PREVIEW_SNAPSHOTS.add(preview);
  return preview;
}

export function createPreview(turns, { contactNames = [] } = {}) {
  assertValid(Array.isArray(turns) && turns.length >= 1 && turns.length <= 8 &&
    turns.every((line) => typeof line === 'string' && line.trim() && !/[\r\n]/u.test(line)), 'luot');
  return makePreviewSnapshot(redactPersonalInfo(turns.join('\n'), { contactNames }));
}

export function redactPreviewSelection(preview, lineIndex, start, end) {
  assertValid(PREVIEW_SNAPSHOTS.has(preview), 'ban_xem_truoc');
  const lines = preview.text.split('\n');
  const line = lines[lineIndex];
  const splitsCharacter = (i) => /[\uDC00-\uDFFF]/u.test(line.charAt(i));
  assertValid(Number.isInteger(lineIndex) && typeof line === 'string' &&
    Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end <= line.length &&
    start < end && !splitsCharacter(start) && !splitsCharacter(end), 'vi_tri_che');
  // Vị trí được tính trên bản xem trước đã chuẩn hóa.
  const offset = lines.slice(0, lineIndex).reduce((count, turn) => count + turn.length + 1, 0);
  return makePreviewSnapshot(redactPersonalInfo(preview.text, {
    cheThem: [{ start: offset + start, end: offset + end }],
  }));
}

export function getScoreBucket(probability) {
  assertValid(Number.isFinite(probability) && probability >= 0 && probability <= 1, 'khoang_diem');
  return Math.min(9, Math.floor(probability * 10));
}

async function createFeedbackPayload(data, level, submissionId, storage, additionalFields = {}) {
  const {
    ma_bao_cao: reportCode, nhan_mo_hinh: modelLabel, nhan_de_xuat: suggestedLabel,
    ma_ly_do: reasonCode, ma_chi_tiet: detailCode, ly_do_khac: otherReason,
    co_to_sang: hasHighlights, khoang_diem: scoreBucket,
    phien_ban_mo_hinh: modelVersion, phien_ban_nguong: thresholdVersion,
  } = data;
  assertValid(otherReason == null ||
    (typeof otherReason === 'string' && Array.from(otherReason.trim()).length <= 200), 'ly_do_khac');
  const payload = {
    loai: level, ma_lan_gui: submissionId, ma_thiet_bi: getDeviceId(storage),
    ma_bao_cao: reportCode, nhan_mo_hinh: modelLabel, nhan_de_xuat: suggestedLabel,
    ma_ly_do: reasonCode, ma_chi_tiet: detailCode,
    ly_do_khac: otherReason == null ? null : redactPersonalInfo(otherReason.trim()),
    co_to_sang: hasHighlights, khoang_diem: scoreBucket,
    phien_ban_mo_hinh: modelVersion, phien_ban_nguong: thresholdVersion,
    ...additionalFields,
  };
  await validateFeedback(payload);
  return Object.freeze(payload);
}

export function createLevel1Payload(data, { submissionId = crypto.randomUUID(), storage = localStorage } = {}) {
  return createFeedbackPayload(data, 'muc_1', submissionId, storage);
}

export async function createLevel2Payload(data, preview, {
  consent = false, submissionId = crypto.randomUUID(), storage = localStorage,
} = {}) {
  assertValid(isLevel2Enabled(storage) && consent === true && PREVIEW_SNAPSHOTS.has(preview), 'dong_y');
  return createFeedbackPayload(data, 'muc_2', submissionId, storage, {
    doan_da_che: preview.text,
    dong_y: Object.freeze({ phien_ban: CONSENT_VERSION, ban_xem_truoc_sha256: await sha256(preview.text) }),
  });
}

export async function submitFeedback(payload, { storage = localStorage, fetcher = fetch } = {}) {
  assertValid(payload.loai !== 'muc_2' || isLevel2Enabled(storage), 'dong_y');
  await validateFeedback(payload);
  const response = await fetcher('/api/phan-hoi', {
    method: 'POST', credentials: 'omit',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw Object.assign(new Error(result.loi), { status: response.status, ...result });
  if (result.da_luu && !result.gui_lai && payload.loai === 'muc_1' && payload.ma_bao_cao === 'tin_nhan_an_toan') {
    recordFalsePositive(payload.nhan_mo_hinh, storage);
  }
  return result;
}
