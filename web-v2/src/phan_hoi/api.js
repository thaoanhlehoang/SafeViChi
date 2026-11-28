import { createLevel1Payload, createLevel2Payload, submitFeedback } from '../feedback/client.js';

let phienBan = { mo_hinh: 'dualorc-mmbert-int8@v2', nguong: 'baseline' };

export function datPhienBan({ mo_hinh, nguong }) {
  if (mo_hinh && nguong) phienBan = { mo_hinh, nguong };
}

export function layPhienBan() {
  return { ...phienBan };
}

export async function laBanDemo() {
  const response = await fetch('/api/noi-bo/che-do', { credentials: 'omit' });
  return response.ok && (await response.json()).demo === true;
}

export async function guiPhanHoi(goi, { banXemTruoc, dongY = false } = {}) {
  const submissionId = goi.ma_lan_gui;
  const payload = banXemTruoc
    ? await createLevel2Payload(goi, banXemTruoc, { submissionId, consent: dongY })
    : await createLevel1Payload(goi, { submissionId });
  const ketQua = await submitFeedback(payload);
  if (ketQua.da_luu !== true) throw new Error('Máy chủ chưa xác nhận lưu phản hồi.');
  return ketQua;
}
