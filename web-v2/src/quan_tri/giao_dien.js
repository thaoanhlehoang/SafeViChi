import { LABELS } from '../labels.js';

export const TRANG_THAI = {
  cho_duyet: 'Chờ duyệt',
  cho_nguoi_thu_hai: 'Chờ người duyệt thứ hai',
  bat_dong: 'Chưa thống nhất',
  da_duyet: 'Đã duyệt',
  cho_phe_duyet: 'Chờ phê duyệt',
};

export function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

export function nut(text, cls, handle) {
  const button = el('button', 'button ' + (cls ?? ''), text);
  button.type = 'button';
  if (handle) button.onclick = handle;
  return button;
}

export function trangThai(value) {
  const badge = el('span', 'badge', TRANG_THAI[value] ?? value);
  badge.dataset.status = value;
  return badge;
}

export function nhan(value) {
  const badge = el('span', 'badge tag', LABELS[value]?.ten ?? 'Chưa có đề xuất');
  badge.dataset.label = LABELS[value] ? value : '';
  // Màu nhãn lấy từ labels.js như màn chat; đặt bằng CSSOM nên vẫn hợp CSP style-src 'self'.
  if (LABELS[value]) badge.style.setProperty('--c', LABELS[value].mau);
  return badge;
}

export function truong(text, control) {
  const label = el('label', 'field', text);
  label.append(control);
  return label;
}

export function input(options = {}) {
  return Object.assign(el('input'), options);
}

export function select(options) {
  const control = el('select');
  for (const [value, text] of options) control.append(Object.assign(el('option', null, text), { value }));
  return control;
}

export function nhac() {
  const node = el('p', 'notice');
  node.setAttribute('role', 'status');
  return node;
}

export function thongBao(node, text, error = false) {
  node.textContent = text;
  node.classList.toggle('is-error', error);
}

export function dauThe(title, action) {
  const head = el('header', 'card-head');
  head.append(el('h2', null, title));
  if (action) head.append(action);
  return head;
}

export function bang(headers, rows) {
  const wrap = el('div', 'table-wrap');
  const table = el('table');
  if (headers.length >= 4) {
    wrap.classList.add('table-wide');
    wrap.tabIndex = 0;
    wrap.setAttribute('role', 'region');
    wrap.setAttribute('aria-label', 'Bảng có thể cuộn ngang');
  }
  const header = el('tr');
  for (const title of headers) header.append(Object.assign(el('th', null, title), { scope: 'col' }));
  const head = el('thead');
  head.append(header);
  const body = el('tbody');
  for (const cells of rows) {
    const row = el('tr');
    for (const value of cells) {
      const cell = el('td');
      cell.append(value ?? '—');
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
  wrap.append(table);
  return wrap;
}

export function focusChiTiet(card) {
  const heading = card.querySelector('h2');
  heading.tabIndex = -1;
  heading.focus();
}

const dateTime = new Intl.DateTimeFormat('vi-VN', {
  dateStyle: 'short', timeStyle: 'short', timeZone: 'Asia/Ho_Chi_Minh',
});

export function gioVietNam(timestamp) {
  return dateTime.format(new Date(timestamp));
}

export function ngayVietNam(date) {
  return date.split('-').reverse().join('/');
}
