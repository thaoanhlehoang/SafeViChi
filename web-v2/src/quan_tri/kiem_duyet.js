// Hàng đợi kiểm duyệt và màn một mẫu.
// - Hàng đợi: tab Chờ duyệt / Đã duyệt, xếp theo ngày gửi, mẫu cũ nhất trước.
// - Một mẫu: 2 cột như màn chat. Trái là đoạn đã che dạng bong bóng; phải là ① kết quả AI,
//   ② người dùng báo gì, ③ quyết định của người duyệt.
// - Một người duyệt, một lượt duyệt là chốt (quyết định 28/11/2026).
import { LABELS } from '../labels.js';
import { UUID_PATTERN } from '../feedback/contract.js';
import { tenLuaChon } from '../phan_hoi/ma.js';
import { el, input, nhac, nhan, nut, thongBao, trangThai, truong, TRANG_THAI } from './giao_dien.js';

const TABS = [
  ['cho_duyet', 'Chờ duyệt'],
  ['da_duyet', 'Đã duyệt'],
];
const GHI_CHU_TOI_DA = 200;

function thoiGianCho(ngay) {
  const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Ho_Chi_Minh' }).format(new Date());
  const days = Math.round((Date.parse(today) - Date.parse(ngay)) / 86400_000);
  if (days <= 0) return 'Gửi hôm nay';
  return 'Chờ ' + days + ' ngày';
}

function khoangDiem(bucket) {
  if (!Number.isInteger(bucket)) return '—';
  return bucket * 10 + '–' + (bucket === 9 ? 100 : bucket * 10 + 10) + '%';
}

// "AI: Đe dọa → Người dùng: An toàn · Đùa hoặc nói thân mật"
function tomTat(row) {
  const wrap = el('span', 'qt-tom-tat');
  wrap.append(nhan(row.nhan_mo_hinh), el('span', 'qt-mui-ten', '→'), nhan(row.nhan_de_xuat));
  const ly = [row.ma_bao_cao, row.ma_ly_do].filter(Boolean).map(tenLuaChon).join(' · ');
  if (ly) wrap.append(el('span', 'qt-ly-do', ly));
  return wrap;
}

/** Đoạn hội thoại đã che dạng bong bóng; mỗi lượt là textarea chỉ đọc để chọn được vùng chữ. */
export function veHoiThoai(doan, { boSung } = {}) {
  const list = el('ol', 'qt-hoi-thoai');
  const texts = doan.split('\n').map((line, i) => {
    const item = el('li', 'qt-luot' + (i % 2 ? ' qt-luot-phai' : ''));
    const head = el('div', 'qt-luot-dau');
    const label = el('label', null, 'Lượt ' + (i + 1));
    const text = Object.assign(el('textarea', 'qt-bong-bong'), {
      id: 'qt-luot-' + i, readOnly: true, value: line,
      rows: Math.max(1, Math.min(6, Math.ceil(line.length / 46))),
    });
    label.htmlFor = text.id;
    head.append(label);
    boSung?.(head, i, text);
    item.append(head, text);
    list.append(item);
    return text;
  });
  return { list, texts };
}

function nutNhan(onChon) {
  const group = el('div', 'qt-nut-nhan');
  group.setAttribute('role', 'radiogroup');
  group.setAttribute('aria-label', 'Nhãn sau khi xem xét');
  const buttons = Object.entries(LABELS).map(([ma, { ten, mau }]) => {
    const button = el('button', 'qt-nhan-chon', ten);
    button.type = 'button';
    button.dataset.label = ma;
    button.style.setProperty('--c', mau);
    button.setAttribute('role', 'radio');
    button.setAttribute('aria-checked', 'false');
    button.onclick = () => {
      for (const other of buttons) other.setAttribute('aria-checked', String(other === button));
      onChon(ma);
    };
    group.append(button);
    return button;
  });
  return { group, buttons };
}

export function hienHangDoi(root, api) {
  root.className = 'qt-kiem-duyet';
  let tab = 'cho_duyet';
  let rows = [];
  let cursor = null;
  let loading = false;

  // ---------- Danh sách ----------
  const listView = el('div', 'stack');
  const tabBar = el('div', 'qt-tabs');
  tabBar.setAttribute('role', 'tablist');
  tabBar.setAttribute('aria-label', 'Trạng thái mẫu');
  const tabButtons = TABS.map(([key, ten]) => {
    const button = el('button', 'qt-tab');
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.dataset.tab = key;
    button.append(el('span', null, ten), el('span', 'qt-dem', '0'));
    button.onclick = () => { tab = key; taiDanhSach(); };
    tabBar.append(button);
    return button;
  });
  const reload = nut('Làm mới', 'quiet', () => taiDanhSach());
  const toolbar = el('div', 'qt-thanh-cong-cu');
  toolbar.append(tabBar, reload);
  const items = el('div', 'qt-danh-sach');
  const more = nut('Tải thêm', 'quiet', () => taiDanhSach(true));
  more.hidden = true;
  const listStatus = nhac();
  const find = el('form', 'qt-tim');
  const sampleId = input({ type: 'text', required: true, maxLength: 36, spellcheck: false, placeholder: 'Mã mẫu (UUID)' });
  const open = nut('Mở mẫu', 'quiet');
  open.type = 'submit';
  find.append(truong('Mở theo mã mẫu', sampleId), open);
  find.onsubmit = (event) => {
    event.preventDefault();
    const id = sampleId.value.trim().toLowerCase();
    if (!UUID_PATTERN.test(id)) return thongBao(listStatus, 'Mã mẫu phải có định dạng UUID.', true);
    moMau(id);
  };
  listView.append(toolbar, items, more, listStatus, find);

  // ---------- Một mẫu ----------
  const sampleView = el('div', 'qt-mau');
  sampleView.hidden = true;
  root.append(listView, sampleView);

  async function taiDanhSach(append = false) {
    if (loading) return;
    loading = true;
    reload.disabled = more.disabled = true;
    for (const button of tabButtons) button.setAttribute('aria-selected', String(button.dataset.tab === tab));
    thongBao(listStatus, 'Đang tải hàng đợi…');
    try {
      const params = new URLSearchParams({ trang_thai: tab });
      if (append && cursor) {
        params.set('sau_ngay', cursor.ngay);
        params.set('sau_id', cursor.id);
      }
      const result = await api('duyet?' + params);
      for (const button of tabButtons) button.querySelector('.qt-dem').textContent = result.dem[button.dataset.tab] ?? 0;
      rows = append ? rows.concat(result.hang_doi) : result.hang_doi;
      cursor = rows.at(-1) ?? null;
      items.replaceChildren(...rows.map(veDong));
      if (!rows.length) {
        items.append(el('p', 'empty', tab === 'da_duyet' ? 'Chưa có mẫu nào được duyệt.' : 'Không còn mẫu nào trong mục này.'));
      }
      more.hidden = !result.con_mau;
      thongBao(listStatus, '');
    } catch (err) {
      thongBao(listStatus, err.message, true);
    } finally {
      loading = false;
      reload.disabled = more.disabled = false;
    }
  }

  function veDong(row) {
    const item = el('button', 'qt-dong');
    item.type = 'button';
    item.onclick = () => moMau(row.id);
    const top = el('span', 'qt-dong-dau');
    top.append(el('strong', null, 'Mẫu ' + row.id.slice(0, 8)));
    const meta = el('span', 'qt-dong-meta');
    meta.append(trangThai(row.trang_thai === 'da_duyet' ? 'da_duyet' : 'cho_duyet'), el('span', null, thoiGianCho(row.ngay)));
    item.append(top, tomTat(row), meta);
    return item;
  }

  function veLai() {
    sampleView.hidden = true;
    listView.hidden = false;
  }

  function mauTiepTheo(id) {
    const index = rows.findIndex((row) => row.id === id);
    return rows[index + 1]?.id ?? null;
  }

  async function moMau(id, message = '') {
    listView.hidden = true;
    sampleView.hidden = false;
    sampleView.replaceChildren(el('p', 'notice', 'Đang mở mẫu…'));
    try {
      const { mau } = await api('duyet/' + id);
      veMau(mau, message);
    } catch (err) {
      sampleView.replaceChildren(nut('← Hàng đợi', 'quiet', veLai), el('p', 'notice is-error', err.message));
    }
  }

  function veMau(mau, message) {
    const back = nut('← Hàng đợi', 'quiet', () => { veLai(); taiDanhSach(); });
    const head = el('header', 'qt-mau-dau');
    const title = el('h2', null, 'Mẫu ' + mau.id.slice(0, 8));
    title.tabIndex = -1;
    head.append(back, title, trangThai(mau.trang_thai === 'da_duyet' ? 'da_duyet' : 'cho_duyet'), el('code', 'code', mau.id));
    const notice = nhac();
    if (message) thongBao(notice, message);

    const daXong = Boolean(mau.quyet_dinh || mau.da_gui_luot_duyet);
    const coTheChe = !daXong && mau.trang_thai === 'cho_duyet';

    // ----- Cột trái: đoạn hội thoại -----
    const left = el('section', 'qt-cot-trai');
    left.setAttribute('aria-label', 'Đoạn hội thoại');
    const leftStatus = nhac();
    const addButtons = [];
    const { list, texts } = veHoiThoai(mau.doan_da_che, {
      boSung: daXong ? null : (headRow, i, text) => {
        const add = nut('Thêm bằng chứng', 'quiet', () => themBangChung(i, text));
        add.disabled = true;
        add.setAttribute('aria-label', 'Thêm bằng chứng từ lượt ' + (i + 1));
        addButtons.push(add);
        headRow.append(add);
        if (coTheChe) {
          const che = nut('Che phần đã chọn', 'quiet', () => cheThem(i, text));
          che.setAttribute('aria-label', 'Che phần đã chọn ở lượt ' + (i + 1));
          headRow.append(che);
        }
      },
    });
    left.append(el('h3', null, 'Đoạn hội thoại'),
      list, leftStatus);

    // ----- Cột phải -----
    const right = el('div', 'qt-cot-phai');
    const ai = el('section', 'card');
    ai.append(el('h3', null, '① Kết quả AI'));
    const aiMeta = el('div', 'meta');
    aiMeta.append(nhan(mau.nhan_mo_hinh), el('span', null, 'Độ tin cậy ' + khoangDiem(mau.khoang_diem)),
      el('span', null, mau.co_to_sang ? 'Có tô sáng bằng chứng' : 'Không tô sáng'));
    ai.append(aiMeta, el('p', 'muted', 'Bằng chứng nguyên văn của AI không được gửi lên máy chủ.'));

    const user = el('section', 'card');
    user.append(el('h3', null, '② Người dùng báo'));
    const userMeta = el('div', 'meta');
    userMeta.append('Nhãn đề xuất', nhan(mau.nhan_de_xuat));
    user.append(userMeta);
    const report = el('dl', 'qt-bao-cao');
    for (const [ten, ma] of [['Loại báo cáo', mau.ma_bao_cao], ['Lý do', mau.ma_ly_do], ['Chi tiết', mau.ma_chi_tiet]]) {
      if (ma) report.append(el('dt', null, ten), el('dd', null, tenLuaChon(ma)));
    }
    if (mau.ly_do_khac) report.append(el('dt', null, 'Mô tả'), el('dd', null, mau.ly_do_khac));
    user.append(report);

    const decision = el('section', 'card');
    decision.append(el('h3', null, '③ Quyết định của bạn'));
    right.append(ai, user, decision);

    const body = el('div', 'qt-mau-than');
    body.append(left, right);
    sampleView.replaceChildren(head, notice, body);
    title.focus();

    if (mau.quyet_dinh) {
      decision.append(el('p', 'muted', 'Mẫu đã được chốt.'), nhan(mau.quyet_dinh.nhan));
      if (mau.quyet_dinh.bang_chung.length) {
        const ev = el('ul', 'evidence');
        ev.append(...mau.quyet_dinh.bang_chung.map(({ text }) => el('li', null, text)));
        decision.append(ev);
      }
      return;
    }
    if (mau.da_gui_luot_duyet) {
      decision.append(el('p', 'notice', 'Bạn đã gửi lượt duyệt cho mẫu này.'));
      return;
    }

    let label = '';
    let spans = [];
    let saving = false;
    const { group: labelGroup } = nutNhan((ma) => {
      label = ma;
      groupField.hidden = ma !== 'XUC_PHAM';
      if (groupField.hidden) groupBox.checked = false;
      if (ma === 'AN_TOAN') { spans = []; veBangChung(); }
      for (const button of addButtons) button.disabled = ma === 'AN_TOAN';
      capNhat();
    });
    const groupBox = input({ type: 'checkbox' });
    const groupField = el('label', 'check');
    groupField.append(groupBox, ' Nhắm vào một nhóm hoặc vùng miền');
    groupField.hidden = true;
    const evidence = el('ul', 'evidence');
    const evidenceHint = el('p', 'muted', 'Chọn nhãn, rồi quét chọn cụm nguyên văn trong một lượt bên trái và bấm “Thêm bằng chứng”.');
    const note = Object.assign(el('textarea'), { rows: 2, maxLength: GHI_CHU_TOI_DA * 2 });
    const noteCount = el('span', 'qt-dem-chu', '0/' + GHI_CHU_TOI_DA);
    note.oninput = () => {
      const n = Array.from(note.value.trim()).length;
      noteCount.textContent = n + '/' + GHI_CHU_TOI_DA;
      noteCount.classList.toggle('is-error', n > GHI_CHU_TOI_DA);
      capNhat();
    };
    const noteField = truong('Ghi chú (không bắt buộc)', note);
    noteField.append(noteCount);
    const status = nhac();
    const save = nut('Lưu và mở mẫu tiếp', 'primary', luu);
    const skip = nut('Bỏ qua', 'quiet', () => {
      const next = mauTiepTheo(mau.id);
      if (next) moMau(next);
      else { veLai(); taiDanhSach(); }
    });
    const actions = el('div', 'button-row');
    actions.append(save, skip);
    decision.append(el('p', 'qt-nhan-hoi', 'Nhãn sau khi xem xét'), labelGroup, groupField,
      el('p', 'qt-nhan-hoi', 'Bằng chứng'), evidenceHint, evidence, noteField, status, actions);

    function capNhat() {
      const n = Array.from(note.value.trim()).length;
      save.disabled = saving || !label || n > GHI_CHU_TOI_DA;
    }

    function veBangChung() {
      evidence.replaceChildren(...spans.map((span, i) => {
        const item = el('li');
        item.append(el('span', null, texts[span.dong].value.slice(span.start, span.end)),
          nut('Bỏ', 'quiet', () => { spans.splice(i, 1); veBangChung(); }));
        return item;
      }));
      evidenceHint.hidden = spans.length > 0;
    }

    function themBangChung(i, text) {
      const { selectionStart: start, selectionEnd: end } = text;
      if (!text.value.slice(start, end).trim()) {
        thongBao(leftStatus, 'Chọn cụm từ trong lượt ' + (i + 1) + ' trước khi thêm.', true);
        text.focus();
        return;
      }
      if (spans.length >= 16) return thongBao(leftStatus, 'Một lượt duyệt lưu tối đa 16 cụm bằng chứng.', true);
      if (spans.some((s) => s.dong === i && s.start === start && s.end === end)) return;
      spans.push({ dong: i, start, end });
      thongBao(leftStatus, '');
      veBangChung();
    }

    async function cheThem(i, text) {
      const { selectionStart: start, selectionEnd: end } = text;
      if (start === end) {
        thongBao(leftStatus, 'Chọn phần cần che trong lượt ' + (i + 1) + '.', true);
        text.focus();
        return;
      }
      thongBao(leftStatus, 'Đang che…');
      try {
        await api('duyet/' + mau.id + '/che-them', { dong: i, start, end });
        await moMau(mau.id, 'Đã che thêm ở lượt ' + (i + 1) + '. Thao tác đã được ghi nhật ký.');
      } catch (err) {
        thongBao(leftStatus, err.message, true);
      }
    }

    async function luu() {
      if (save.disabled) return;
      saving = true;
      capNhat();
      thongBao(status, 'Đang lưu lượt duyệt…');
      try {
        const result = await api('duyet/' + mau.id, {
          nhan: label, nham_nhom: groupBox.checked, bang_chung: spans,
          ghi_chu: note.value.trim() || null,
        });
        const next = mauTiepTheo(mau.id);
        const msg = 'Đã lưu mẫu ' + mau.id.slice(0, 8) + ': ' + TRANG_THAI[result.trang_thai] + '.';
        if (next) moMau(next, msg);
        else { veLai(); await taiDanhSach(); thongBao(listStatus, msg); }
      } catch (err) {
        thongBao(status, err.message, true);
        saving = false;
        capNhat();
      }
    }
    capNhat();
  }

  taiDanhSach();
}

