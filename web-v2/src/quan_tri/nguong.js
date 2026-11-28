import { LABELS } from '../labels.js';
import { HARMFUL_LABEL_CODES, isValidCode } from '../feedback/contract.js';
import { BASELINE_THRESHOLD_VERSION } from '../feedback/thresholds.js';
import { bang, dauThe, el, focusChiTiet, input, nhac, nut, thongBao, trangThai, truong } from './giao_dien.js';

const percent = new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 2 });
const numberInput = () => input({ type: 'number', min: '0', max: '1', step: 'any', required: true });
const reasonInput = () => input({ type: 'text', required: true, maxLength: 480 });

function metricsField(title) {
  const fieldset = el('fieldset');
  const row = el('div', 'field-grid');
  const f1 = numberInput();
  const fpr = numberInput();
  row.append(truong('Macro-F1', f1), truong('FPR trên An toàn', fpr));
  fieldset.append(el('legend', null, title), row);
  return { fieldset, values: () => ({ macro_f1: f1.valueAsNumber, fpr_an_toan: fpr.valueAsNumber }) };
}

function formDeXuat(api, onCreated) {
  const form = el('form', 'stack');
  const identity = el('div', 'field-grid');
  const model = input({ type: 'text', required: true, maxLength: 80, spellcheck: false, placeholder: 'dualorc-mmbert-int8@…' });
  const base = input({ type: 'text', required: true, maxLength: 80, spellcheck: false, value: BASELINE_THRESHOLD_VERSION });
  const version = input({ type: 'text', required: true, maxLength: 80, spellcheck: false });
  identity.append(truong('Phiên bản mô hình', model), truong('Ngưỡng gốc', base), truong('Phiên bản ngưỡng mới', version));
  const status = nhac();
  const check = nut('Tra ngưỡng hiện tại', 'quiet', async () => {
    const modelVersion = model.value.trim();
    if (!isValidCode(modelVersion)) {
      thongBao(status, 'Hãy điền mã phiên bản mô hình hợp lệ.', true);
      model.focus();
      return;
    }
    check.disabled = true;
    try {
      const params = new URLSearchParams({ phien_ban_mo_hinh: modelVersion });
      const { cau_hinh: current } = await api('nguong?' + params);
      if (model.value.trim() !== modelVersion) return;
      base.value = current?.phien_ban ?? BASELINE_THRESHOLD_VERSION;
      thongBao(status, 'Ngưỡng đang dùng: ' + base.value + '.');
    } catch (err) {
      thongBao(status, err.message, true);
    } finally {
      check.disabled = false;
    }
  });
  const checkRow = el('div', 'button-row');
  checkRow.append(check);

  const thresholds = el('fieldset');
  const thresholdGrid = el('div', 'field-grid');
  const scores = HARMFUL_LABEL_CODES.map((code) => {
    const control = numberInput();
    thresholdGrid.append(truong(LABELS[code].ten, control));
    return [code, control];
  });
  thresholds.append(el('legend', null, 'Ngưỡng mới (0–1)'), thresholdGrid);
  const validation = el('fieldset');
  const dataRow = el('div', 'field-grid');
  const dataset = input({ type: 'text', required: true, maxLength: 80, spellcheck: false });
  const count = input({ type: 'number', min: '1', step: '1', required: true });
  const before = metricsField('Trước');
  const after = metricsField('Sau');
  const metrics = el('div', 'field-grid');
  dataRow.append(truong('Mã tập validation', dataset), truong('Số mẫu', count));
  metrics.append(before.fieldset, after.fieldset);
  validation.className = 'stack';
  validation.append(el('legend', null, 'Đánh giá trên cùng tập validation (0–1)'), dataRow, metrics);
  const reason = reasonInput();
  const save = nut('Tạo đề xuất', 'primary');
  save.type = 'submit';
  form.append(identity, checkRow, thresholds, validation, truong('Lý do đề xuất', reason), status, save);
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (save.disabled) return;
    save.disabled = true;
    thongBao(status, 'Đang tạo đề xuất…');
    try {
      const result = await api('de-xuat', {
        phien_ban: version.value.trim(),
        phien_ban_mo_hinh: model.value.trim(),
        phien_ban_goc: base.value.trim(),
        nguong: Object.fromEntries(scores.map(([code, control]) => [code, control.valueAsNumber])),
        validation: { tap: dataset.value.trim(), so_mau: count.valueAsNumber, truoc: before.values(), sau: after.values() },
        ly_do: reason.value.trim(),
      });
      form.reset();
      thongBao(status, 'Đã tạo đề xuất. Thay đổi đang chờ quản trị viên phê duyệt.');
      onCreated(result.id);
    } catch (err) {
      thongBao(status, err.message, true);
    } finally {
      save.disabled = false;
    }
  };
  return form;
}

export function hienNguong(root, api, role) {
  root.className = 'stack';
  const split = el('div', 'split');
  const list = el('section', 'card');
  const items = el('div', 'items');
  const listStatus = nhac();
  const detail = el('section', 'card');
  detail.append(el('p', 'empty', 'Chọn một đề xuất để xem ngưỡng và số đo validation.'));
  const refresh = nut('Làm mới', 'quiet', () => loadList());
  list.append(dauThe('Đề xuất đang chờ', refresh), items, listStatus);
  split.append(list, detail);
  const create = el('details', 'card');
  const summary = el('summary', 'item-main', 'Đề xuất ngưỡng mới');
  create.append(summary, formDeXuat(api, (id) => {
    create.open = false;
    loadList();
    openProposal(id, 'Đã tạo đề xuất. Cấu hình mới sẽ có hiệu lực sau phê duyệt.');
  }));
  root.append(split, create);
  let activeId;
  let requestId = 0;

  async function loadList() {
    refresh.disabled = true;
    thongBao(listStatus, 'Đang tải đề xuất…');
    try {
      const { de_xuat: proposals } = await api('de-xuat');
      items.replaceChildren(...proposals.map((proposal) => {
        const button = el('button', 'item');
        button.type = 'button';
        button.dataset.id = proposal.id;
        button.setAttribute('aria-pressed', String(activeId === proposal.id));
        button.append(el('span', 'item-main', proposal.phien_ban), el('span', 'code muted', proposal.phien_ban_mo_hinh));
        button.onclick = () => openProposal(proposal.id);
        return button;
      }));
      if (!proposals.length) items.append(el('p', 'empty', 'Không có đề xuất đang chờ.'));
      thongBao(listStatus, '');
    } catch (err) {
      thongBao(listStatus, err.message, true);
    } finally {
      refresh.disabled = false;
    }
  }

  async function openProposal(id, message = '') {
    activeId = id;
    const index = ++requestId;
    for (const button of items.children) if (button.dataset.id) {
      button.setAttribute('aria-pressed', String(button.dataset.id === id));
    }
    detail.replaceChildren(el('p', 'notice', 'Đang tải đề xuất…'));
    try {
      const proposal = await api('de-xuat/' + id);
      if (index !== requestId) return;
      renderProposal(proposal, message);
      focusChiTiet(detail);
    } catch (err) {
      if (index === requestId) detail.replaceChildren(el('p', 'notice is-error', err.message));
    }
  }

  function renderProposal(proposal, message) {
    const { validation, nguong } = proposal;
    detail.replaceChildren(
      dauThe(proposal.phien_ban, trangThai(proposal.trang_thai)),
      el('code', 'code muted', proposal.id),
      el('p', 'muted', 'Mô hình ' + proposal.phien_ban_mo_hinh + ' · Ngưỡng gốc ' + proposal.phien_ban_goc),
      el('p', 'muted', 'Người đề xuất: ' + proposal.nguoi_de_xuat),
      el('p', null, proposal.ly_do),
      el('h3', null, 'Kết quả validation'),
      el('p', 'muted', validation.tap + ' · ' + validation.so_mau.toLocaleString('vi-VN') + ' mẫu'),
      bang(['Chỉ số', 'Trước', 'Sau'], [
        ['Macro-F1', percent.format(validation.truoc.macro_f1), percent.format(validation.sau.macro_f1)],
        ['FPR trên An toàn', percent.format(validation.truoc.fpr_an_toan), percent.format(validation.sau.fpr_an_toan)],
      ]),
      bang(['Nhãn', 'Ngưỡng đề xuất'], HARMFUL_LABEL_CODES.map((code) => [LABELS[code].ten, percent.format(nguong[code])])),
    );
    if (message) detail.append(el('p', 'notice', message));
    if (proposal.nguoi_phe_duyet) {
      detail.append(el('p', 'muted', 'Đã phê duyệt bởi ' + proposal.nguoi_phe_duyet + '.'),
        el('p', null, proposal.ly_do_phe_duyet));
    } else if (role === 'quan_tri') {
      const form = el('form', 'stack');
      const reason = reasonInput();
      const approve = nut('Phê duyệt và áp dụng', 'primary');
      approve.type = 'submit';
      const status = nhac();
      form.append(truong('Lý do phê duyệt', reason), approve, status);
      form.onsubmit = async (event) => {
        event.preventDefault();
        if (approve.disabled) return;
        approve.disabled = true;
        thongBao(status, 'Đang phê duyệt…');
        try {
          await api('de-xuat/' + proposal.id + '/phe-duyet', { ly_do: reason.value.trim() });
          if (activeId === proposal.id) openProposal(proposal.id, 'Đã phê duyệt và cập nhật ngưỡng hiệu lực.');
          loadList();
        } catch (err) {
          thongBao(status, err.message, true);
          approve.disabled = false;
        }
      };
      detail.append(form);
    }
  }

  loadList();
}
