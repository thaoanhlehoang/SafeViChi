import { UUID_PATTERN } from '../feedback/contract.js';
import { tenLuaChon } from '../phan_hoi/ma.js';
import { bang, dauThe, el, gioVietNam, input, ngayVietNam, nhac, nhan, nut, select, thongBao, truong } from './giao_dien.js';

const groupLabels = {
  nhan: ['Nhãn AI', 'Nhãn người dùng đề xuất'],
  ly_do: ['Loại báo cáo', 'Lý do chính', 'Chi tiết'],
  khoang_diem: ['Khoảng xác suất'],
  phien_ban: ['Phiên bản mô hình', 'Phiên bản ngưỡng'],
};

function statsCells(row, group) {
  switch (group) {
    case 'nhan': return [nhan(row.nhan_mo_hinh), nhan(row.nhan_de_xuat)];
    case 'ly_do': return [row.ma_bao_cao, row.ma_ly_do, row.ma_chi_tiet].map(tenLuaChon);
    case 'khoang_diem': return [row.khoang_diem * 10 + '–' + (row.khoang_diem + 1) * 10 + '%'];
    case 'phien_ban': return [row.phien_ban_mo_hinh, row.phien_ban_nguong];
    default: return [];
  }
}

export function hienThongKe(root, api) {
  const card = el('section', 'card');
  const controls = el('form', 'filters');
  const by = select([['nhan', 'Nhãn'], ['ly_do', 'Lý do'], ['khoang_diem', 'Khoảng xác suất'], ['phien_ban', 'Phiên bản']]);
  const from = input({ type: 'date' });
  const to = input({ type: 'date' });
  const submit = nut('Xem thống kê', 'primary');
  submit.type = 'submit';
  controls.append(truong('Nhóm theo', by), truong('Từ ngày', from), truong('Đến ngày', to), submit);
  const status = nhac();
  const output = el('div', 'stack');
  card.append(controls, status, output);
  root.append(card);

  async function load() {
    if (submit.disabled) return;
    submit.disabled = true;
    thongBao(status, 'Đang tải thống kê…');
    output.replaceChildren();
    const params = new URLSearchParams({ theo: by.value });
    if (from.value) params.set('tu', from.value);
    if (to.value) params.set('den', to.value);
    try {
      const stats = await api('thong-ke?' + params);
      from.value = stats.tu;
      to.value = stats.den;
      const summary = el('div', 'stack');
      summary.append(el('p', 'metric', stats.tong.toLocaleString('vi-VN') + ' lượt'),
        el('p', 'muted', ngayVietNam(stats.tu) + ' – ' + ngayVietNam(stats.den)
          + ' · Số lượt người dùng báo cáo.'));
      const table = stats.nhom.length
        ? bang([...groupLabels[stats.theo], 'Số lượt'],
          stats.nhom.map((row) => [...statsCells(row, stats.theo), row.so_luot.toLocaleString('vi-VN')]))
        : el('p', 'empty', 'Chưa có phản hồi trong khoảng ngày này.');
      output.replaceChildren(summary, table);
      if (stats.con_nhom) output.append(el('p', 'muted', 'Bảng hiển thị 250 nhóm có số lượt cao nhất.'));
      thongBao(status, '');
    } catch (err) {
      thongBao(status, err.message, true);
    } finally {
      submit.disabled = false;
    }
  }
  controls.onsubmit = (event) => { event.preventDefault(); load(); };
  load();
}

const auditLabels = {
  mo_mau: 'Mở mẫu',
  luu_luot_duyet: 'Lưu lượt duyệt',
  tao_de_xuat: 'Tạo đề xuất',
  phe_duyet_nguong: 'Phê duyệt ngưỡng',
};

export function hienNhatKy(root, api) {
  const card = el('section', 'card');
  const filter = el('form', 'filters');
  const subject = input({ type: 'text', placeholder: 'Để trống để xem gần nhất', maxLength: 36, spellcheck: false });
  const search = nut('Xem nhật ký', 'primary');
  search.type = 'submit';
  filter.append(truong('Mã mẫu hoặc mã đề xuất', subject), search);
  const status = nhac();
  const output = el('div', 'stack');
  card.append(dauThe('Hoạt động gần đây'), filter, status, output);
  root.append(card);
  async function load() {
    if (search.disabled) return;
    const id = subject.value.trim().toLowerCase();
    if (id && !UUID_PATTERN.test(id)) {
      thongBao(status, 'Mã cần tra phải có định dạng UUID.', true);
      return;
    }
    search.disabled = true;
    thongBao(status, 'Đang tải nhật ký…');
    output.replaceChildren();
    try {
      const { nhat_ky: events } = await api(id ? 'nhat-ky/' + id : 'nhat-ky');
      output.replaceChildren(events.length ? bang(
        ['Thời gian', 'Người thực hiện', 'Hành động', 'Mã đối tượng', 'Chi tiết'],
        events.map((event) => [
          gioVietNam(event.luc), event.nguoi, auditLabels[event.hanh_dong] ?? event.hanh_dong,
          el('code', 'code', event.doi_tuong), event.chi_tiet,
        ]),
      ) : el('p', 'empty', 'Chưa có hoạt động phù hợp.'));
      if (events.length === 100) output.append(el('p', 'muted', 'Đang hiển thị 100 sự kiện gần nhất.'));
      thongBao(status, '');
    } catch (err) {
      thongBao(status, err.message, true);
    } finally {
      search.disabled = false;
    }
  }
  filter.onsubmit = (event) => { event.preventDefault(); load(); };
  load();
}
