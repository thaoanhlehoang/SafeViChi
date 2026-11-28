// Khu vực kiểm duyệt và quản trị, dựng được ở hai nơi:
// - trang riêng quan-tri.html (dự phòng),
// - tab "Kiểm duyệt & quản trị" nằm cạnh tab nhắn tin trong index.html.
// Mọi truy vấn DOM đi qua `root` để hai nơi dùng chung một mã.
import { el, thongBao } from './giao_dien.js';
import { hienHangDoi } from './kiem_duyet.js';
import { hienNhatKy, hienThongKe } from './bao_cao.js';
import { hienNguong } from './nguong.js';

const sections = [
  ['Thống kê phản hồi', '', hienThongKe],
  ['Hàng đợi kiểm duyệt', 'Mở từng mẫu đã che, sửa nhãn và chọn bằng chứng trên chính bản đã che.', hienHangDoi, 'nguoi_duyet'],
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
  trang: 'trang danh sách', ghi_chu: 'ghi chú', luot_duyet_id: 'lượt duyệt được chọn', trang_thai: 'trạng thái',
};

const MARKUP = `
  <div class="qt-entry entry" data-k="entry">
    <section class="sign-in" aria-labelledby="qt-entry-title">
      <div class="entry-intro">
        <h1 id="qt-entry-title">Kiểm duyệt<br>và quản trị</h1>
        <p>Xem xét phản hồi đã che thông tin, thống kê mức 1 và thay đổi ngưỡng mô hình.</p>
      </div>
      <div class="entry-controls">
        <form class="entry-form" data-k="sign-in" hidden>
          <h2>Đăng nhập nội bộ</h2>
          <label class="field">Mã truy cập
            <input data-k="token" type="password" required autocomplete="off" spellcheck="false" maxlength="128">
          </label>
          <button class="button primary" type="submit">Vào khu vực nội bộ</button>
        </form>
        <p class="notice" role="status" data-k="sign-in-status">Đang tải chế độ truy cập…</p>
        <button class="button" data-k="retry-entry" type="button" hidden>Thử lại</button>
      </div>
    </section>
  </div>

  <div class="workspace" data-k="workspace" hidden>
    <aside class="workspace-nav">
      <header class="nav-head"><h2>Kiểm duyệt</h2></header>
      <button class="button quiet" data-k="sign-out" type="button" hidden>Đăng xuất</button>
      <p class="nav-label" aria-hidden="true">Mục</p>
      <nav class="qt-nav" data-k="navigation" aria-label="Khu vực nội bộ"></nav>
    </aside>
    <section class="workspace-content" aria-labelledby="qt-screen-title">
      <header class="screen-header">
        <h1 id="qt-screen-title" data-k="screen-title" tabindex="-1"></h1>
        <p data-k="screen-description"></p>
      </header>
      <div class="screen" data-k="screen"></div>
    </section>
  </div>`;

/**
 * Dựng khu vực kiểm duyệt và quản trị vào `root` (phần tử hoặc ShadowRoot).
 * Trả về `{ focus }` để nơi nhúng chuyển tiêu điểm khi mở tab.
 */
export function dungKhuVucQuanTri(root) {
  const holder = document.createElement('div');
  holder.className = 'qt-root';
  holder.innerHTML = MARKUP;
  root.append(holder);
  const $ = (key) => holder.querySelector(`[data-k="${key}"]`);

  let token = '';
  let actor;
  let page;
  let demoRoles = [];
  let started = false;

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
    for (const key of ['screen', 'navigation']) $(key).replaceChildren();
    for (const key of ['screen-title', 'screen-description']) $(key).textContent = '';
    $('workspace').hidden = true;
    $('sign-out').hidden = true;
    $('entry').hidden = false;
    thongBao($('sign-in-status'), message, !!message);
    // Chế độ trình diễn không có màn chọn vai trò: hiện nút "Thử lại" để vào lại vai trò mặc định.
    if (demoRoles.length) {
      $('retry-entry').hidden = false;
      $('retry-entry').focus();
    } else $('token').focus();
  }

  function vaoVaiTroDemo(id) {
    const role = demoRoles.find((option) => option.id === id);
    if (!role) return;
    actor = role;
    moKhuVuc();
  }

  function moKhuVuc() {
    const menu = $('navigation');
    // Người kiểm duyệt của bản demo làm được mọi mục; đường token thật vẫn lọc theo vai trò.
    const views = sections.filter(([, , , role]) => !role || actor.vai_tro === 'kiem_duyet' || role === actor.vai_tro);
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
    menu.replaceChildren(...buttons);
    $('entry').hidden = true;
    $('workspace').hidden = false;
    $('sign-out').hidden = demo;
    buttons[0].click();
  }

  $('sign-out').onclick = () => signOut();
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
      thongBao(status, '');
      if (demo) {
        // Bản demo chỉ có một người kiểm duyệt: vào thẳng khu vực, không có bước chọn vai trò.
        vaoVaiTroDemo(demoRoles[0].id);
      }
    } catch (err) {
      thongBao(status, err.message, true);
      retry.hidden = false;
    }
  }

  $('retry-entry').onclick = taiCheDoTruyCap;

  return {
    /** Gọi khi khu vực hiện ra lần đầu: chỉ lúc này mới hỏi máy chủ chế độ truy cập. */
    batDau() {
      if (started) return;
      started = true;
      taiCheDoTruyCap();
    },
    /** Đưa tiêu điểm vào chỗ hợp lý nhất khi mở tab. */
    focus() {
      if (!$('workspace').hidden) $('screen-title').focus();
      else if (!$('retry-entry').hidden) $('retry-entry').focus();
      else if (!$('sign-in').hidden) $('token').focus();
    },
  };
}
