import { UUID_PATTERN } from '../feedback/contract.js';
import { LOAI_BAO_CAO, tenLuaChon } from '../phan_hoi/ma.js';
import { LABELS } from '../labels.js';
import { bang, dauThe, el, gioVietNam, input, ngayVietNam, nhac, nhan, nut, select, thongBao, truong } from './giao_dien.js';

// Thống kê phản hồi: đếm chung mọi lần gửi, có hay không kèm đoạn hội thoại.
function phanTram(n, tong) {
  return tong ? Math.round((100 * n) / tong) + '%' : '0%';
}

function chipNhan(ma) {
  const info = LABELS[ma];
  const chip = el('span', 'badge tag', info?.ten ?? 'Chưa chắc');
  if (info) chip.style.setProperty('--c', info.mau);
  return chip;
}

function oSo(ten, so, phu) {
  const tile = el('div', 'qt-o-so');
  tile.append(el('span', 'qt-o-so-ten', ten), el('strong', 'metric', so));
  if (phu) tile.append(el('span', 'qt-o-so-phu', phu));
  return tile;
}

// Thanh ngang như thanh xác suất ở cột phân tích của màn chat.
function thanh(nhanTrai, n, max, mau) {
  const row = el('div', 'qt-thanh');
  const bar = el('span', 'qt-thanh-ve');
  const fill = el('span', 'qt-thanh-day');
  fill.style.setProperty('--w', max ? String(n / max) : '0');
  if (mau) fill.style.setProperty('--c', mau);
  bar.append(fill);
  const left = el('span', 'qt-thanh-ten');
  left.append(nhanTrai);
  row.append(left, bar, el('span', 'qt-thanh-so', n.toLocaleString('vi-VN')));
  return row;
}

function theThongKe(tieuDe, phu, ...noiDung) {
  const card = el('section', 'card qt-the-tk');
  card.append(el('h3', null, tieuDe));
  if (phu) card.append(el('p', 'muted', phu));
  card.append(...noiDung);
  return card;
}

const RANGES = [['7', '7 ngày'], ['30', '30 ngày'], ['tuy_chon', 'Tùy chọn']];

export function hienThongKe(root, api) {
  root.className = 'stack';
  const filter = el('form', 'qt-loc');
  const seg = el('div', 'qt-tabs');
  let range = '7';
  const from = input({ type: 'date' });
  const to = input({ type: 'date' });
  const custom = el('div', 'qt-loc-ngay');
  custom.append(truong('Từ ngày', from), truong('Đến ngày', to));
  const apply = nut('Xem', 'primary');
  apply.type = 'submit';
  custom.append(apply);
  custom.hidden = true;
  const segButtons = RANGES.map(([key, ten]) => {
    const b = el('button', 'qt-tab', ten);
    b.type = 'button';
    b.onclick = () => {
      range = key;
      for (const other of segButtons) other.setAttribute('aria-pressed', String(other === b));
      custom.hidden = key !== 'tuy_chon';
      if (key !== 'tuy_chon') load();
    };
    b.setAttribute('aria-pressed', String(key === range));
    seg.append(b);
    return b;
  });
  filter.append(seg, custom);
  filter.onsubmit = (event) => { event.preventDefault(); load(); };
  const status = nhac();
  const output = el('div', 'stack');
  root.append(filter, status, output);

  function ngayCach(days) {
    return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Ho_Chi_Minh' })
      .format(new Date(Date.now() - days * 86400_000));
  }

  let busy = false;
  async function load() {
    if (busy) return;
    busy = true;
    thongBao(status, 'Đang tải thống kê…');
    const params = new URLSearchParams({ theo: 'nhan' });
    if (range === 'tuy_chon') {
      if (from.value) params.set('tu', from.value);
      if (to.value) params.set('den', to.value);
    } else {
      params.set('tu', ngayCach(Number(range) - 1));
      params.set('den', ngayCach(0));
    }
    try {
      const st = await api('thong-ke?' + params);
      from.value = st.tu;
      to.value = st.den;
      veThongKe(st);
      thongBao(status, '');
    } catch (err) {
      thongBao(status, err.message, true);
    } finally {
      busy = false;
    }
  }

  function veThongKe(st) {
    const kem = st.theo_loai.muc_2;
    const tiles = el('div', 'qt-hang-o');
    tiles.append(
      oSo('Tổng phản hồi', st.tong.toLocaleString('vi-VN'), ngayVietNam(st.tu) + ' – ' + ngayVietNam(st.den)),
      oSo('Có kèm đoạn hội thoại', kem.toLocaleString('vi-VN'), phanTram(kem, st.tong) + ' tổng phản hồi'),
      oSo('Đã duyệt', st.kem_doan.da_duyet.toLocaleString('vi-VN'), 'trong số phản hồi kèm đoạn'),
      oSo('Chờ duyệt', st.kem_doan.chua_duyet.toLocaleString('vi-VN'), 'trong số phản hồi kèm đoạn'),
    );
    if (!st.tong) {
      output.replaceChildren(tiles, el('p', 'empty', 'Chưa có phản hồi trong khoảng ngày này.'));
      return;
    }

    // (1) Theo loại báo cáo: đủ 5 nhánh, kể cả nhánh 0 lượt.
    const reportMax = Math.max(1, ...st.theo_bao_cao.map((r) => r.n));
    const reports = LOAI_BAO_CAO.map(({ ma, ten }) =>
      thanh(ten, st.theo_bao_cao.find((r) => r.ma_bao_cao === ma)?.n ?? 0, reportMax, '#24552c'));

    // (2) AI gắn → người dùng đề xuất.
    const pairs = el('ol', 'qt-cap');
    for (const row of st.cap_nhan) {
      const item = el('li');
      const left = el('span', 'qt-cap-nhan');
      left.append(chipNhan(row.nhan_mo_hinh), el('span', 'qt-mui-ten', '→'), chipNhan(row.nhan_de_xuat));
      item.append(left, el('span', 'qt-thanh-so', row.n.toLocaleString('vi-VN')));
      pairs.append(item);
    }

    // (3) Lý do phổ biến theo từng nhánh báo cáo.
    const reasons = el('div', 'qt-ly-do-nhom');
    for (const { ma, ten } of LOAI_BAO_CAO) {
      const rows = st.ly_do.filter((r) => r.ma_bao_cao === ma).slice(0, 4);
      if (!rows.length) continue;
      const block = el('div', 'qt-ly-do-khoi');
      const max = Math.max(...rows.map((r) => r.n));
      block.append(el('h4', null, ten), ...rows.map((r) => thanh(tenLuaChon(r.ma_ly_do), r.n, max, '#7d887d')));
      reasons.append(block);
    }
    if (!reasons.children.length) reasons.append(el('p', 'muted', 'Chưa có lý do nào được chọn.'));

    const grid = el('div', 'qt-luoi-tk');
    grid.append(
      theThongKe('Theo loại báo cáo', null, ...reports),
      theThongKe('AI gắn → Người dùng đề xuất', 'Các cặp nhãn gặp nhiều nhất.', pairs),
      theThongKe('Lý do phổ biến', 'Tối đa 4 lý do mỗi loại báo cáo.', reasons),
    );
    output.replaceChildren(tiles, grid);
  }

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
