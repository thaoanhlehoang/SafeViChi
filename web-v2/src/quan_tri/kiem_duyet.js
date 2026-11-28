import { LABELS } from '../labels.js';
import { UUID_PATTERN } from '../feedback/contract.js';
import { tenLuaChon } from '../phan_hoi/ma.js';
import { dauThe, el, focusChiTiet, input, ngayVietNam, nhac, nhan, nut, select, thongBao, trangThai, truong, TRANG_THAI } from './giao_dien.js';

function bienTap(mau, lines, card, turns, luot, api, onSaved) {
  const form = el('form', 'stack');
  const choice = select([['', 'Chọn nhãn'], ...Object.entries(LABELS).map(([key, label]) => [key, label.ten])]);
  choice.required = true;
  const group = input({ type: 'checkbox' });
  const groupField = el('label', 'check');
  groupField.append(group, ' Xúc phạm nhắm vào nhóm hoặc vùng miền');
  groupField.hidden = true;
  const sources = el('ul', 'evidence');
  const status = nhac();
  const submit = nut('Lưu lượt duyệt', 'primary');
  submit.type = 'submit';
  let saving = false;
  let spans = [];
  const sourcesHint = el('p', 'muted', 'Chưa có bằng chứng.');
  const addButtons = lines.map((text, index) => {
    const button = nut('Thêm bằng chứng', 'quiet', () => {
      const { selectionStart: start, selectionEnd: end } = text;
      if (!text.value.slice(start, end).trim()) {
        thongBao(status, 'Chọn cụm từ trong lượt ' + (index + 1) + ' trước khi thêm.', true);
        text.focus();
        return;
      }
      if (spans.length >= 16) {
        thongBao(status, 'Một lượt duyệt có thể lưu tối đa 16 cụm bằng chứng.', true);
        return;
      }
      if (spans.some((span) => span.dong === index && span.start === start && span.end === end)) return;
      spans.push({ dong: index, start, end });
      thongBao(status, '');
      hienBangChung();
    });
    button.disabled = true;
    button.setAttribute('aria-label', 'Thêm bằng chứng từ lượt ' + (index + 1));
    luot[index].append(button);
    return button;
  });

  function hienBangChung() {
    sources.replaceChildren(...spans.map(({ dong, start, end }, i) => {
      const item = el('li');
      item.append(el('span', null, lines[dong].value.slice(start, end)), nut('Bỏ', 'quiet', () => {
        spans.splice(i, 1);
        hienBangChung();
      }));
      return item;
    }));
    sourcesHint.hidden = spans.length > 0;
  }

  choice.onchange = () => {
    groupField.hidden = choice.value !== 'XUC_PHAM';
    if (groupField.hidden) group.checked = false;
    if (choice.value === 'AN_TOAN') {
      spans = [];
      hienBangChung();
    }
    for (const button of addButtons) button.disabled = !choice.value || choice.value === 'AN_TOAN';
  };
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (saving) return;
    saving = true;
    submit.disabled = choice.disabled = group.disabled = true;
    sources.inert = true;
    for (const button of addButtons) button.disabled = true;
    thongBao(status, 'Đang lưu lượt duyệt…');
    try {
      const result = await api('duyet/' + mau.id, { nhan: choice.value, nham_nhom: group.checked, bang_chung: spans });
      onSaved(result.trang_thai);
    } catch (err) {
      thongBao(status, err.message, true);
      saving = false;
      submit.disabled = choice.disabled = group.disabled = false;
      sources.inert = false;
      for (const button of addButtons) button.disabled = choice.value === 'AN_TOAN';
    }
  };
  const footer = el('div', 'button-row');
  footer.append(submit);
  form.append(dauThe('Lượt duyệt của bạn'), truong('Nhãn sau khi xem xét', choice), groupField,
    el('p', 'muted', 'Nếu có bằng chứng, chọn cụm nguyên văn trong một lượt rồi bấm “Thêm bằng chứng”.'),
    el('h3', null, 'Đoạn hội thoại đã che'), turns,
    sourcesHint, sources, status, footer);
  card.append(form);
}

export function hienHangDoi(root, api) {
  root.className = 'split';
  const queue = el('section', 'card');
  const detail = el('section', 'card');
  const items = el('div', 'items');
  const queueStatus = nhac();
  const more = nut('Tải thêm', 'quiet', () => loadQueue(true));
  more.hidden = true;
  const reload = nut('Làm mới', 'quiet', () => loadQueue());
  queue.append(dauThe('Mẫu chờ xử lý', reload), items, more, queueStatus);
  detail.append(el('p', 'empty', 'Chọn một mẫu để mở bản đã che.'));

  const find = el('form', 'stack');
  const sampleId = input({ type: 'text', required: true, maxLength: 36, spellcheck: false });
  const open = nut('Mở mẫu', 'quiet');
  open.type = 'submit';
  find.append(truong('Mã mẫu đã biết', sampleId), open);
  find.onsubmit = (event) => {
    event.preventDefault();
    const id = sampleId.value.trim().toLowerCase();
    if (!UUID_PATTERN.test(id)) {
      thongBao(queueStatus, 'Mã mẫu phải có định dạng UUID.', true);
      return;
    }
    openSample(id);
  };
  queue.append(find);
  root.append(queue, detail);

  let cursor;
  let selected;
  let detailRequest = 0;
  function markSelected() {
    for (const button of items.children) button.setAttribute('aria-pressed', String(button.dataset.id === selected));
  }

  async function loadQueue(append = false) {
    if (reload.disabled) return;
    reload.disabled = more.disabled = true;
    thongBao(queueStatus, 'Đang tải hàng đợi…');
    try {
      const query = append && cursor
        ? '?' + new URLSearchParams({ sau_ngay: cursor.ngay, sau_id: cursor.id }) : '';
      const result = await api('duyet' + query);
      if (!append) items.replaceChildren();
      if (!items.children.length && !result.hang_doi.length) items.append(el('p', 'empty', 'Bạn đã xem hết mẫu đang chờ.'));
      for (const row of result.hang_doi) {
        const item = el('button', 'item');
        item.type = 'button';
        item.dataset.id = row.id;
        item.onclick = () => openSample(row.id);
        const meta = el('span', 'item-meta');
        meta.append(nhan(row.nhan_mo_hinh), trangThai(row.trang_thai));
        item.append(el('span', 'item-main', 'Mẫu ' + row.id.slice(0, 8)), meta,
          el('span', 'item-meta', ngayVietNam(row.ngay)));
        items.append(item);
      }
      markSelected();
      cursor = result.hang_doi.at(-1) ?? cursor;
      more.hidden = !result.con_mau;
      thongBao(queueStatus, '');
    } catch (err) {
      thongBao(queueStatus, err.message, true);
    } finally {
      reload.disabled = more.disabled = false;
    }
  }

  async function openSample(id, message = '') {
    const attempt = ++detailRequest;
    selected = id;
    markSelected();
    detail.replaceChildren(el('p', 'notice', 'Đang mở mẫu…'));
    try {
      const { mau } = await api('duyet/' + id);
      if (attempt !== detailRequest) return;
      renderSample(mau, message);
      focusChiTiet(detail);
    } catch (err) {
      if (attempt === detailRequest) detail.replaceChildren(el('p', 'notice is-error', err.message));
    }
  }

  function renderSample(mau, message) {
    const header = dauThe('Mẫu phản hồi mức 2', trangThai(mau.trang_thai));
    const meta = el('div', 'meta');
    meta.append('AI gợi ý', nhan(mau.nhan_mo_hinh), 'Người dùng đề xuất', nhan(mau.nhan_de_xuat));
    const report = [mau.ma_bao_cao, mau.ma_ly_do, mau.ma_chi_tiet].filter(Boolean).map(tenLuaChon).join(' · ');
    detail.replaceChildren(header, el('code', 'code muted', mau.id), meta, el('p', 'muted', report));
    if (message) detail.append(el('p', 'notice', message));
    if (mau.ly_do_khac) detail.append(el('p', null, mau.ly_do_khac));
    const turns = el('div', 'stack');
    const headings = [];
    const lines = mau.doan_da_che.split('\n').map((line, i) => {
      const block = el('div', 'turn');
      const head = el('div', 'turn-head');
      const label = el('label', null, 'Lượt ' + (i + 1));
      const text = Object.assign(el('textarea'), {
        id: 'sample-turn-' + i, readOnly: true, value: line,
        rows: Math.max(2, Math.min(6, Math.ceil(line.length / 66))),
      });
      label.htmlFor = text.id;
      head.append(label);
      headings.push(head);
      block.append(head, text);
      turns.append(block);
      return text;
    });
    if (mau.quyet_dinh || mau.da_gui_luot_duyet) detail.append(el('h3', null, 'Đoạn hội thoại đã che'), turns);
    if (mau.quyet_dinh) {
      const decision = el('div', 'stack');
      decision.append(el('h3', null, 'Quyết định sau duyệt'), nhan(mau.quyet_dinh.nhan));
      if (mau.quyet_dinh.bang_chung.length) {
        const list = el('ul', 'evidence');
        list.append(...mau.quyet_dinh.bang_chung.map(({ text }) => el('li', null, text)));
        decision.append(list);
      }
      detail.append(decision);
    } else if (mau.da_gui_luot_duyet) {
      detail.append(el('p', 'notice', 'Bạn đã gửi lượt duyệt cho mẫu này.'));
    } else {
      bienTap(mau, lines, detail, turns, headings, api, (status) => {
        if (selected === mau.id) openSample(mau.id, 'Đã lưu lượt duyệt. ' + TRANG_THAI[status] + '.');
        loadQueue();
      });
    }
  }

  loadQueue();
}
