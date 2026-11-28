const KEY = 'safevichi.bao_sai_gan_day';
const WINDOW_MS = 7 * 86400_000;
const WARNING_PREFERENCES = {
  XUC_PHAM: { blur: false },
  QUAY_ROI: { suggestBlock: false },
};

function readWarningHistory(storage) {
  try {
    return JSON.parse(storage.getItem(KEY)) ?? {};
  } catch {
    return {};
  }
}

function withinLastSevenDays(list, now) {
  return Array.isArray(list) ? list.filter((t) => Number.isFinite(t) && t > now - WINDOW_MS && t <= now) : [];
}

export function recordFalsePositive(label, storage = localStorage, now = Date.now()) {
  if (!Object.hasOwn(WARNING_PREFERENCES, label)) return;
  const data = readWarningHistory(storage);
  const times = [...withinLastSevenDays(data[label], now), now].slice(-3);
  try {
    storage.setItem(KEY, JSON.stringify({ ...data, [label]: times }));
  } catch {
    // Lỗi cài đặt cục bộ không làm thay đổi kết quả gửi phản hồi.
  }
}

export function getWarningPreference(label, storage = localStorage, now = Date.now()) {
  if (!Object.hasOwn(WARNING_PREFERENCES, label) || withinLastSevenDays(readWarningHistory(storage)[label], now).length < 3) return null;
  return { ...WARNING_PREFERENCES[label] };
}

export function resetWarnings(storage = localStorage) {
  storage.removeItem(KEY);
}
