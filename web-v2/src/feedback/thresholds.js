import { HARMFUL_LABEL_CODES, isValidCode } from './contract.js';

export const BASELINE_THRESHOLD_VERSION = 'baseline';

export function isValidThresholds(thresholds) {
  return Boolean(thresholds && typeof thresholds === 'object' && !Array.isArray(thresholds) &&
    Object.keys(thresholds).length === HARMFUL_LABEL_CODES.length &&
    HARMFUL_LABEL_CODES.every((key) => Number.isFinite(thresholds[key]) && thresholds[key] >= 0 && thresholds[key] <= 1));
}

export function applyThreshold(label, p, thresholds) {
  return thresholds && Object.hasOwn(thresholds, label) && p < thresholds[label] ? 'AN_TOAN' : label;
}

// Chỉ đọc cấu hình khi nạp mô hình. Mỗi lần suy luận vẫn chạy hoàn toàn trên thiết bị.
export async function fetchActiveThresholds(modelVersion, fetcher = fetch) {
  try {
    const url = '/api/nguong?phien_ban_mo_hinh=' + encodeURIComponent(modelVersion);
    const response = await fetcher(url, { credentials: 'omit', signal: AbortSignal.timeout(2500) });
    if (!response.ok) return null;
    const { cau_hinh: configuration } = await response.json();
    if (configuration?.phien_ban_mo_hinh === modelVersion &&
        isValidCode(configuration.phien_ban) && isValidThresholds(configuration.nguong)) return configuration;
  } catch {
    // Khi web tĩnh hoặc ngoại tuyến, dùng bộ giải mã gốc.
  }
  return null;
}
