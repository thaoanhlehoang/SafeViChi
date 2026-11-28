import { cheThongTin } from '../phan_hoi/che_thong_tin.js';

function chuanHoa(text) {
  return text.normalize('NFKC').replace(/[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]/gu, '');
}

// Dùng chung cho bản xem trước, mô tả tự do và xác thực API.
export function redactPersonalInfo(text, { contactNames = [], cheThem = [] } = {}) {
  return cheThongTin(chuanHoa(text), {
    tenLienHe: contactNames.filter((name) => typeof name === 'string').map(chuanHoa),
    giuTenMien: false,
    cheThem,
  }).text;
}
