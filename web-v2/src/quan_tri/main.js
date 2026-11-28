import '../fonts.css';
import './quan_tri.css';
import { el, thongBao } from './giao_dien.js';
import { hienHangDoi } from './kiem_duyet.js';
import { hienNhatKy, hienThongKe } from './bao_cao.js';
import { hienNguong } from './nguong.js';

const $ = (id) => document.getElementById(id);
const sections = [
  ['Hàng đợi kiểm duyệt', 'Mở từng mẫu đã che, sửa nhãn và chọn bằng chứng trên chính bản đã che.', hienHangDoi, 'nguoi_duyet'],
  ['Thống kê mức 1', 'Xem số lượt phản hồi theo nhãn, lý do, khoảng điểm hoặc phiên bản.', hienThongKe],
  ['Ngưỡng mô hình', 'Đối chiếu số đo validation và quản lý đề xuất thay đổi ngưỡng.', hienNguong],
  ['Nhật ký', 'Tra hoạt động kiểm duyệt và phê duyệt theo mã mẫu hoặc mã đề xuất.', hienNhatKy, 'quan_tri'],
];

const errors = {
  can_xac_thuc: 'Mã truy cập không hợp lệ hoặc đã được thu hồi.',
  chon_vai_tro_demo: 'Hãy chọn lại vai trò demo.',
  thieu_quyen: 'Vai trò hiện tại không có quyền thực hiện thao tác này.',
  chua_cau_hinh: 'API phản hồi chưa được cấu hình.',
  chua_cau_hinh_noi_bo: 'Máy chủ chưa cấu hình mã truy cập nội bộ.',
  khong_tim_thay_mau: 'Không tìm thấy mẫu mức 2 này.',
  khong_tim_thay_de_xuat: 'Không tìm thấy đề xuất này.',
  khong_the_duyet_lai: 'Mẫu đã được duyệt hoặc bạn đã gửi lượt duyệt cho mẫu này.',
  de_xuat_khong_con_hop_le: 'Đề xuất đã được duyệt hoặc phiên bản ngưỡng gốc đã thay đổi.',
  trung_phien_ban: 'Phiên bản ngưỡng đã được dùng cho mô hình này.',
  bo_loc_khong_hop_le: 'Hãy kiểm tra ngày bắt đầu, ngày kết thúc và giới hạn 90 ngày.',
};
const fields = {
  ly_do: 'lý do', nhan: 'nhãn', bang_chung: 'bằng chứng', nguong: 'ngưỡng',
  so_do: 'số đo validation', validation: 'thông tin tập validation',
  phien_ban: 'phiên bản mới', phien_ban_goc: 'phiên bản gốc', phien_ban_mo_hinh: 'phiên bản mô hình',
  trang: 'trang danh sách',
};

let token = '';
let actor;
let page;
let demoRoles = [];

async function request(path, { body, signal, credential = token } = {}) {
  const headers = {};
  if (demoRoles.length && actor) headers['X-SafeViChi-Demo-Role'] = actor.id;
  else if (credential) headers.Authorization = 'Bearer ' + credential;
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  let res;
  try {
    res = await fetch('/api/' + path, {
      headers, credentials: 'omit', signal,
      method: body === undefined ? 'GET' : 'POST',
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new Error('Không kết nối được máy chủ phản hồi.');
  }
  const data = await res.json().catch(() => null);
  if (!data) throw new Error('Không tìm thấy API phản hồi. Kiểm tra máy chủ Wrangler hoặc binding của bản triển khai.');
  if (!res.ok) {
    const message = errors[data.loi] ?? (data.loi === 'du_lieu_khong_hop_le'
      ? 'Dữ liệu không hợp lệ ở ' + (fields[data.truong] ?? 'một trường bắt buộc') + '.'
      : 'Máy chủ chưa hoàn tất yêu cầu.');
    if (res.status === 401 && actor && credential === token) signOut(message);
    throw new Error(message);
  }
  return data;
}

function signOut(message = '') {
  page?.abort();
  token = '';
  actor = undefined;
  $('screen').replaceChildren();
  $('navigation').replaceChildren();
  $('actor-id').textContent = '';
  $('actor-role').textContent = '';
  $('actor-mark').textContent = '';
  $('screen-title').textContent = '';
  $('screen-description').textContent = '';
  $('workspace').hidden = true;
  $('sign-out').hidden = true;
  $('demo-switch').hidden = true;
  $('entry').hidden = false;
  thongBao($('sign-in-status'), message, !!message);
  if (demoRoles.length) $('demo-roles').querySelector('button')?.focus();
  else $('token').focus();
}

function vaoVaiTroDemo(id) {
  const role = demoRoles.find((option) => option.id === id);
  if (!role) return;
  actor = role;
  moKhuVuc();
}

function moKhuVuc() {
  const menu = $('navigation');
  const views = sections.filter(([, , , role]) => !role || role === actor.vai_tro);
  const buttons = views.map(([title, description, render]) => {
    const button = el('button', 'nav-item', title);
    button.type = 'button';
    button.onclick = () => {
      page?.abort();
      const controller = new AbortController();
      page = controller;
      for (const item of menu.children) item.removeAttribute('aria-current');
      button.setAttribute('aria-current', 'page');
      $('screen-title').textContent = title;
      $('screen-description').textContent = description;
      $('screen-title').focus();
      const view = el('div');
      $('screen').replaceChildren(view);
      const api = (path, body) => request(path, { body, signal: controller.signal });
      render(view, api, actor.vai_tro);
    };
    return button;
  });
  const demo = demoRoles.length > 0;
  const displayName = demo ? actor.ten : actor.id;
  $('actor-id').textContent = displayName;
  $('actor-role').textContent = demo ? 'Chế độ trình diễn' : actor.vai_tro === 'quan_tri' ? 'Quản trị viên' : 'Người duyệt';
  $('actor-mark').textContent = displayName[0].toUpperCase();
  menu.replaceChildren(...buttons);
  $('entry').hidden = true;
  $('workspace').hidden = false;
  $('sign-out').hidden = demo;
  $('demo-switch').hidden = !demo;
  if (demo) $('demo-role').value = actor.id;
  buttons[0].click();
}

$('sign-out').onclick = () => signOut();
$('demo-role').onchange = (event) => vaoVaiTroDemo(event.currentTarget.value);
$('sign-in').onsubmit = async (event) => {
  event.preventDefault();
  const field = $('token');
  const submit = event.currentTarget.querySelector('[type=submit]');
  if (submit.disabled) return;
  const credential = field.value.trim();
  field.value = '';
  submit.disabled = field.disabled = true;
  const status = $('sign-in-status');
  thongBao(status, 'Đang xác thực…');
  try {
    actor = await request('noi-bo/phien', { credential });
    token = credential;
    thongBao(status, '');
    moKhuVuc();
  } catch (err) {
    thongBao(status, err.message, true);
  } finally {
    submit.disabled = field.disabled = false;
    if (!actor) field.focus();
  }
};

async function taiCheDoTruyCap() {
  const retry = $('retry-entry');
  const status = $('sign-in-status');
  retry.hidden = true;
  thongBao(status, 'Đang tải chế độ truy cập…');
  try {
    const mode = await request('noi-bo/che-do');
    demoRoles = mode.demo ? mode.vai_tro : [];
    const demo = demoRoles.length > 0;
    $('sign-in').hidden = demo;
    $('demo-entry').hidden = !demo;
    if (demo) {
      $('demo-roles').replaceChildren(...demoRoles.map(({ id, ten }, index) => {
        const button = el('button', index === 0 ? 'button primary' : 'button', ten);
        button.type = 'button';
        button.onclick = () => vaoVaiTroDemo(id);
        return button;
      }));
      $('demo-role').replaceChildren(...demoRoles.map(({ id, ten }) => Object.assign(el('option', null, ten), { value: id })));
    }
    thongBao(status, '');
  } catch (err) {
    thongBao(status, err.message, true);
    retry.hidden = false;
  }
}

$('retry-entry').onclick = taiCheDoTruyCap;
taiCheDoTruyCap();
