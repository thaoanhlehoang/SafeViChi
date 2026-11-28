// Điểm nối phản hồi mức 1 với phần logic/DB (người A) và hàm che thông tin (người B).
// Giao diện chỉ gọi hai hàm `guiPhanHoiMuc1` và `cheThongTin`; phần còn lại thay ở đây.
//
// Gói mức 1 (đúng danh sách trường ở CLAUDE.md và docs/v3/DE_XUAT_TRIEN_KHAI_CHANG_2.md, Mục 2):
// {
//   loai: 'muc_1',
//   ma_lan_gui: string,          // ngẫu nhiên, giữ nguyên khi người dùng bấm gửi lại sau lỗi
//   ma_bao_cao: string,          // xem LOAI_BAO_CAO trong ma.js
//   nhan_mo_hinh: 'AN_TOAN' | 'XUC_PHAM' | 'QUAY_ROI' | 'DE_DOA' | 'LUA_DAO',
//   nhan_de_xuat: cùng tập nhãn | null,
//   ma_ly_do: string | null,
//   ma_chi_tiet: string | null,
//   ly_do_khac: string | null,   // chỉ khi chọn "Khác", ≤ 200 ký tự, đã qua cheThongTin
//   co_to_sang: boolean,
//   khoang_diem: number,         // xác suất của nhãn mô hình, làm tròn 0,1
//   phien_ban_mo_hinh: string | null,
//   phien_ban_nguong: string | null,
// }
// Không có tin nhắn, đoạn tô sáng, mã tài khoản, số điện thoại hay IP trong gói.

// Mặc định gửi tới API cùng origin mà người A sẽ gắn vào serve.py.
// Khi chưa có API (404) biểu mẫu báo lỗi và giữ nguyên câu trả lời, đúng quy tắc ở Mục 1.
async function guiMacDinh(goi) {
  const res = await fetch('/api/phan-hoi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(goi),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json().catch(() => ({}));
}

let hamGui = guiMacDinh;

/** Thay cách gửi (vd. thêm mã thiết bị vào header cho bộ đếm tần suất). fn(goi) phải trả Promise. */
export function datHamGuiMuc1(fn) {
  hamGui = typeof fn === 'function' ? fn : guiMacDinh;
}

/** Gửi gói mức 1. Resolve chỉ khi máy chủ xác nhận đã lưu; reject khi lỗi. */
export function guiPhanHoiMuc1(goi) {
  return hamGui(goi);
}

// TẠM: che vài dạng dễ nhận ra để không lọt thông tin thô khi thử. Người B thay bằng hàm che
// dùng chung cho mức 1 và mức 2 qua datHamChe(fn).
function cheTam(s) {
  return s
    .replace(/[\w.+-]+@[\w-]+(\.[\w-]+)+/g, '[EMAIL]')
    .replace(/https?:\/\/([^\s/]+)\S*/gi, (_, host) => `[LINK:${host}]`)
    .replace(/(\+?84|0)(\s?\d){8,10}\b/g, '[SĐT]')
    .replace(/\b\d{6,}\b/g, '[MÃ]');
}

let hamChe = cheTam;

export function datHamChe(fn) {
  hamChe = typeof fn === 'function' ? fn : cheTam;
}

export function cheThongTin(s) {
  return hamChe(s);
}

// Phiên bản hiển thị trong gói. App sẽ lấy phiên bản ngưỡng đã phê duyệt khi khởi tạo phiên demo.
let phienBan = { mo_hinh: 'v2', nguong: null };

export function datPhienBan({ mo_hinh, nguong } = {}) {
  if (mo_hinh !== undefined) phienBan.mo_hinh = mo_hinh;
  if (nguong !== undefined) phienBan.nguong = nguong;
}

export function layPhienBan() {
  return { ...phienBan };
}
