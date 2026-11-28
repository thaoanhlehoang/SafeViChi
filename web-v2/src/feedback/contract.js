import { LABELS } from '../labels.js';
import { redactPersonalInfo } from './redaction.js';

export const LABEL_CODES = Object.keys(LABELS);
export const HARMFUL_LABEL_CODES = LABEL_CODES.filter((label) => label !== 'AN_TOAN');
// Đổi khi câu chữ thông báo đồng ý đổi (v2: bước cuối đề xuất gửi kèm đoạn hội thoại, cam kết quyền riêng tư).
export const CONSENT_VERSION = 'muc-2-v2';
export const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export const LABEL_INDICATORS = {
  XUC_PHAM: ['lang_ma_ca_nhan', 'miet_thi_nhom'],
  QUAY_ROI: ['ga_gam', 'binh_pham_co_the', 'bam_riet_kiem_soat', 'beu_rieu'],
  DE_DOA: ['than_the_tai_san', 'danh_du', 'cuong_ep_tong_tien'],
  LUA_DAO: ['mao_danh', 'chuyen_tien', 'otp_mat_khau', 'lien_ket_ung_dung'],
};
const EVIDENCE_LOCATIONS = ['cung_luot', 'luot_truoc', 'nhieu_luot'];
export const REPORT_OPTIONS = {
  tin_nhan_an_toan: {
    dua_noi_than_mat: ['cach_noi_chuyen', 'tra_loi_nguoi_nhan', 'luot_truoc'],
    tu_long_dia_phuong: ['nghia_quen_thuoc', 'cach_dung', 'luot_truoc'],
    trich_dan_ke_lai: ['dau_hieu_trich_dan', 'noi_ro_dang_ke', 'luot_truoc'],
    ngu_canh_binh_thuong: ['cong_viec_hoc_tap', 'giao_dich_hop_le', 'tro_chuyen_hang_ngay'],
    khac: [],
  },
  sai_loai: { sai_y_ngu_canh: [], uu_tien_dau_hieu_khac: [], khac: [] },
  bo_sot: {
    dau_hieu_truc_tiep: [],
    can_luot_truoc: ['ham_y_nguy_hai', 'hanh_vi_lap_lai', 'nguoi_nhan_tu_choi'],
    viet_bien_dang: ['bo_dau', 'viet_tat_teencode', 'chen_ky_tu', 'viet_dinh', 'ky_tu_dong_dang'],
    khac: [],
  },
  loi_to_sang: {
    to_nham: ['doan_vo_hai', 'lien_quan_mot_phan', 'lien_quan_cho_khac'],
    to_thieu: EVIDENCE_LOCATIONS,
    chua_to: EVIDENCE_LOCATIONS,
    khac: [],
  },
  khac_chua_chac: {
    chua_chac: ['chua_chac_nguy_hai', 'phan_van_giua_nhan', 'can_ngu_canh'],
    van_de_khac: ['canh_bao_kho_hieu', 'canh_bao_qua_thuong_xuyen', 'muc_canh_bao', 'khac'],
  },
};

export class ValidationError extends Error {
  constructor(field) {
    super(field);
    this.field = field;
  }
}

export function assertValid(condition, field) {
  if (!condition) throw new ValidationError(field);
}

export function assertAllowedFields(data, fields) {
  assertValid(data && typeof data === 'object' && !Array.isArray(data), 'du_lieu');
  assertValid(Object.keys(data).every((field) => fields.includes(field)), 'du_lieu');
}

export function isShortText(value, max) {
  return typeof value === 'string' && value.trim() === value &&
    value.length > 0 && Array.from(value).length <= max && !/[\u0000-\u001f\u007f\u2028\u2029\p{Cs}]/u.test(value);
}

export function isValidCode(value) {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.@-]{0,79}$/.test(value);
}

export async function sha256(text) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function validateFeedback(data) {
  assertAllowedFields(data, [
    'loai', 'ma_lan_gui', 'ma_thiet_bi', 'ma_bao_cao', 'nhan_mo_hinh', 'nhan_de_xuat',
    'ma_ly_do', 'ma_chi_tiet', 'ly_do_khac', 'co_to_sang', 'khoang_diem',
    'phien_ban_mo_hinh', 'phien_ban_nguong', 'doan_da_che', 'dong_y',
  ]);
  const {
    loai: level, ma_lan_gui: submissionId, ma_thiet_bi: deviceId,
    ma_bao_cao: reportCode, nhan_mo_hinh: modelLabel, nhan_de_xuat: suggestedLabel = null,
    ma_ly_do: reasonCode = null, ma_chi_tiet: detailCode = null, ly_do_khac: otherReason = null,
    co_to_sang: hasHighlights, khoang_diem: scoreBucket,
    phien_ban_mo_hinh: modelVersion, phien_ban_nguong: thresholdVersion,
    doan_da_che: redactedConversation, dong_y: consentDetails,
  } = data;

  assertValid(level === 'muc_1' || level === 'muc_2', 'loai');
  assertValid(typeof submissionId === 'string' && UUID_PATTERN.test(submissionId), 'ma_lan_gui');
  assertValid(typeof deviceId === 'string' && UUID_PATTERN.test(deviceId), 'ma_thiet_bi');
  assertValid(LABEL_CODES.includes(modelLabel), 'nhan_mo_hinh');
  assertValid(suggestedLabel === null || LABEL_CODES.includes(suggestedLabel), 'nhan_de_xuat');
  const isSafe = modelLabel === 'AN_TOAN';
  assertValid(typeof hasHighlights === 'boolean' && !(isSafe && hasHighlights), 'co_to_sang');
  assertValid(Number.isInteger(scoreBucket) && scoreBucket >= 0 && scoreBucket <= 9, 'khoang_diem');
  assertValid(isValidCode(modelVersion), 'phien_ban_mo_hinh');
  assertValid(isValidCode(thresholdVersion), 'phien_ban_nguong');
  assertValid(typeof reportCode === 'string' && Object.hasOwn(REPORT_OPTIONS, reportCode), 'ma_bao_cao');
  const reasonOptions = REPORT_OPTIONS[reportCode];
  assertValid(reasonCode === null || (typeof reasonCode === 'string' && Object.hasOwn(reasonOptions, reasonCode)), 'ma_ly_do');
  let detailOptions = reasonOptions[reasonCode] ?? [];

  switch (reportCode) {
    case 'tin_nhan_an_toan':
      assertValid(!isSafe, 'ma_bao_cao');
      assertValid(suggestedLabel === 'AN_TOAN', 'nhan_de_xuat');
      assertValid(reasonCode !== null, 'ma_ly_do');
      break;
    case 'sai_loai':
      assertValid(!isSafe, 'ma_bao_cao');
      assertValid(HARMFUL_LABEL_CODES.includes(suggestedLabel) && suggestedLabel !== modelLabel, 'nhan_de_xuat');
      if (reasonCode !== 'khac') detailOptions = LABEL_INDICATORS[suggestedLabel];
      break;
    case 'bo_sot':
      assertValid(isSafe, 'ma_bao_cao');
      assertValid(HARMFUL_LABEL_CODES.includes(suggestedLabel), 'nhan_de_xuat');
      if (reasonCode === 'dau_hieu_truc_tiep') detailOptions = LABEL_INDICATORS[suggestedLabel];
      break;
    case 'loi_to_sang':
      assertValid(!isSafe, 'ma_bao_cao');
      assertValid(suggestedLabel === modelLabel, 'nhan_de_xuat');
      assertValid(reasonCode === 'khac' ||
        (hasHighlights ? ['to_nham', 'to_thieu'].includes(reasonCode) : reasonCode === 'chua_to'), 'ma_ly_do');
      break;
    case 'khac_chua_chac':
      assertValid(reasonCode !== null, 'ma_ly_do');
      if (reasonCode === 'van_de_khac') assertValid(detailCode !== null, 'ma_chi_tiet');
      break;
  }
  assertValid(detailCode === null || detailOptions.includes(detailCode), 'ma_chi_tiet');
  const requiresDescription = reasonCode === 'khac' || detailCode === 'khac';
  assertValid(requiresDescription ? isShortText(otherReason, 200) && redactPersonalInfo(otherReason) === otherReason : otherReason === null,
    'ly_do_khac');
  const feedback = {
    ma_bao_cao: reportCode, nhan_mo_hinh: modelLabel, nhan_de_xuat: suggestedLabel,
    ma_ly_do: reasonCode, ma_chi_tiet: detailCode, ly_do_khac: otherReason,
    co_to_sang: hasHighlights, khoang_diem: scoreBucket,
    phien_ban_mo_hinh: modelVersion, phien_ban_nguong: thresholdVersion,
  };

  if (level === 'muc_1') {
    assertValid(!Object.hasOwn(data, 'doan_da_che') && !Object.hasOwn(data, 'dong_y'), 'loai');
  } else {
    assertValid(typeof redactedConversation === 'string' && Array.from(redactedConversation).length <= 4800 &&
      !/[\u0000-\u0008\u000b-\u001f\u007f\u2028\u2029\p{Cs}]/u.test(redactedConversation), 'doan_da_che');
    const lines = redactedConversation.split('\n');
    assertValid(lines.length <= 8 && lines.every((line) => line.trim()) &&
      redactPersonalInfo(redactedConversation) === redactedConversation, 'doan_da_che');
    assertAllowedFields(consentDetails, ['phien_ban', 'ban_xem_truoc_sha256']);
    assertValid(consentDetails.phien_ban === CONSENT_VERSION &&
      consentDetails.ban_xem_truoc_sha256 === await sha256(redactedConversation), 'dong_y');
    Object.assign(feedback, { doan_da_che: redactedConversation, phien_ban_dong_y: consentDetails.phien_ban });
  }
  return { level, submissionId, deviceId, feedback };
}
