// Biểu mẫu phản hồi dùng chung. Người dùng trả lời các câu hỏi trước; bước cuối luôn đề xuất gửi kèm
// đoạn hội thoại đã che (có cam kết quyền riêng tư) và chỉ gửi kèm khi người dùng bấm đồng ý.
// Quy tắc theo docs/v3/DE_XUAT_TRIEN_KHAI_CHANG_2.md, Mục 1:
// - mỗi câu chọn một đáp án; câu phụ được bỏ qua, trừ "Vấn đề nằm ở đâu?";
// - đổi nhánh, nhãn hoặc lý do thì xóa các đáp án phụ đã mất điều kiện hiển thị;
// - "Mô tả ngắn" chỉ hiện và bắt buộc khi chọn "Khác", tối đa 200 ký tự sau khi bỏ khoảng trắng;
// - chỉ báo "Đã gửi" sau khi máy chủ xác nhận; lỗi thì giữ câu trả lời để thử lại.
// Mọi chữ đi qua textContent, không dùng innerHTML.

import './phan_hoi.css';
import {
  NHAN_NGUY_HAI, TEN_NHAN, DAU_HIEU, LOAI_BAO_CAO,
  LY_DO_AN_TOAN, LY_DO_SAI_LOAI, LY_DO_BO_SOT, LY_DO_TO_SANG, LY_DO_KHAC,
  CHU, MO_TA_TOI_DA,
} from './ma.js';
import { guiPhanHoi, layPhienBan } from './api.js';
import { createPreview, getScoreBucket } from '../feedback/client.js';

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

// Biểu tượng khiên có dấu tích (nét viền), dựng bằng DOM để giữ CSP.
function iconKhien() {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  for (const d of ['M12 3.5l7 2.8v5.2c0 4.3-2.9 7.6-7 9-4.1-1.4-7-4.7-7-9V6.3z', 'M9 12l2.2 2.2L15.5 10']) {
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    svg.append(path);
  }
  return svg;
}

const nhanOpt = (ma) => ({ ma, ten: TEN_NHAN[ma] });

const tim = (options, ma) => options.find((o) => o.ma === ma);

// Danh sách câu hỏi đang hiển thị, tính từ trạng thái hiện tại.
function cauHoi(st, ctx) {
  const qs = [];
  const add = (key, hoi, batBuoc, options) => {
    qs.push({ key, hoi, batBuoc, options });
    return tim(options, st[key]);
  };
  const addPhu = (key, phu) => phu && add(key, phu.hoi, !!phu.batBuoc, phu.options);

  switch (st.bao_cao) {
    case 'tin_nhan_an_toan': {
      const ld = add('ly_do', 'Vì sao bạn cho rằng tin nhắn này an toàn?', true, LY_DO_AN_TOAN);
      addPhu('chi_tiet', ld?.phu);
      break;
    }
    case 'sai_loai': {
      const nhan = add('nhan', 'Theo bạn, nhãn nào phù hợp hơn?', true,
        NHAN_NGUY_HAI.filter((n) => n !== ctx.nhanMoHinh).map(nhanOpt));
      const ld = add('ly_do', 'Vì sao bạn chọn nhãn khác?', false, LY_DO_SAI_LOAI);
      if (nhan && !ld?.khac) add('chi_tiet', 'Bạn nhận thấy dấu hiệu nào?', false, DAU_HIEU[nhan.ma]);
      break;
    }
    case 'bo_sot': {
      const nhan = add('nhan', 'Theo bạn, tin nhắn thuộc loại nào?', true, NHAN_NGUY_HAI.map(nhanOpt));
      if (nhan) {
        const ld = add('ly_do', 'Bạn nhận ra nội dung nguy hại bằng cách nào?', false, LY_DO_BO_SOT);
        if (ld?.phuTheoNhan) add('chi_tiet', 'Bạn nhận thấy dấu hiệu nào?', false, DAU_HIEU[nhan.ma]);
        else addPhu('chi_tiet', ld?.phu);
      }
      break;
    }
    case 'loi_to_sang': {
      const opts = LY_DO_TO_SANG.filter((o) => o.canToSang === undefined || o.canToSang === ctx.coToSang);
      const ld = add('ly_do', 'Phần tô sáng có vấn đề gì?', true, opts);
      addPhu('chi_tiet', ld?.phu);
      break;
    }
    case 'khac_chua_chac': {
      const ld = add('ly_do', 'Bạn muốn báo điều gì?', true, LY_DO_KHAC);
      addPhu('chi_tiet', ld?.phu);
      add('nhan_tuy_chon', 'Nếu đã có nhãn đề xuất, bạn chọn nhãn nào?', false,
        ['AN_TOAN', ...NHAN_NGUY_HAI, 'CHUA_CHAC'].map(nhanOpt));
      break;
    }
    default:
  }
  const coMoTa = qs.some((q) => tim(q.options, st[q.key])?.khac);
  return { qs, coMoTa };
}

// Bỏ các đáp án không còn hiển thị hoặc không còn hợp lệ.
function donTrangThai(st, ctx) {
  for (let lap = 0; lap < 4; lap += 1) {
    const { qs, coMoTa } = cauHoi(st, ctx);
    let doi = false;
    for (const key of ['nhan', 'ly_do', 'chi_tiet', 'nhan_tuy_chon']) {
      if (st[key] == null) continue;
      const q = qs.find((x) => x.key === key);
      if (!q || !tim(q.options, st[key])) { delete st[key]; doi = true; }
    }
    if (!coMoTa && st.mo_ta) { st.mo_ta = ''; doi = true; }
    if (!doi) return;
  }
}

// Đổi câu nào thì xóa đáp án phụ phía sau câu đó.
const PHU_THUOC = {
  bao_cao: ['nhan', 'ly_do', 'chi_tiet', 'nhan_tuy_chon', 'mo_ta'],
  nhan: ['chi_tiet'],
  ly_do: ['chi_tiet'],
};

function hopLe(st, ctx) {
  if (!st.bao_cao) return false;
  const { qs, coMoTa } = cauHoi(st, ctx);
  if (qs.some((q) => q.batBuoc && !st[q.key])) return false;
  if (coMoTa) {
    const len = Array.from((st.mo_ta ?? '').trim()).length;
    if (len === 0 || len > MO_TA_TOI_DA) return false;
  }
  return true;
}

function moTaToGui(st) {
  return (st.mo_ta ?? '').trim().replace(/[\t\r\n]+/g, ' ');
}

function taoGoi(st, ctx, maLanGui) {
  const { coMoTa } = cauHoi(st, ctx);
  let nhanDeXuat = null;
  if (st.bao_cao === 'tin_nhan_an_toan') nhanDeXuat = 'AN_TOAN';
  else if (st.bao_cao === 'sai_loai' || st.bao_cao === 'bo_sot') nhanDeXuat = st.nhan ?? null;
  else if (st.bao_cao === 'loi_to_sang') nhanDeXuat = ctx.nhanMoHinh;
  else if (st.bao_cao === 'khac_chua_chac') {
    nhanDeXuat = st.nhan_tuy_chon && st.nhan_tuy_chon !== 'CHUA_CHAC' ? st.nhan_tuy_chon : null;
  }
  const pb = ctx.phienBan ?? layPhienBan();
  return {
    ma_lan_gui: maLanGui,
    ma_bao_cao: st.bao_cao,
    nhan_mo_hinh: ctx.nhanMoHinh,
    nhan_de_xuat: nhanDeXuat,
    ma_ly_do: st.ly_do ?? null,
    ma_chi_tiet: st.chi_tiet ?? null,
    ly_do_khac: coMoTa ? moTaToGui(st) : null,
    co_to_sang: ctx.nhanMoHinh !== 'AN_TOAN' && !!ctx.coToSang,
    khoang_diem: getScoreBucket(ctx.xacSuat),
    phien_ban_mo_hinh: pb.mo_hinh,
    phien_ban_nguong: pb.nguong,
  };
}

function nhomLuaChon(key, hoi, batBuoc, options, chon, onChon, { lon = false } = {}) {
  const fs = el('fieldset', `ph-q${lon ? ' ph-q-loai' : ''}`);
  const lg = el('legend', 'ph-hoi', hoi);
  lg.append(el('span', batBuoc ? 'ph-bat-buoc' : 'ph-tuy-chon', batBuoc ? ' · Bắt buộc' : ' · Không bắt buộc'));
  fs.append(lg);
  const list = el('div', 'ph-opts');
  for (const o of options) {
    const id = `ph-${key}-${o.ma}`;
    const lab = el('label', 'ph-opt');
    lab.htmlFor = id;
    const r = el('input');
    r.type = 'radio';
    r.name = `ph-${key}`;
    r.id = id;
    r.value = o.ma;
    r.checked = chon === o.ma;
    r.onchange = () => onChon(key, o.ma, id);
    lab.append(r, el('span', 'ph-opt-text', o.ten));
    list.append(lab);
  }
  fs.append(list);
  return fs;
}

/**
 * Mở biểu mẫu phản hồi từ một kết quả phân tích.
 * @param {object} p
 * @param {string} p.nhanMoHinh   nhãn AI đã gán (AN_TOAN | XUC_PHAM | QUAY_ROI | DE_DOA | LUA_DAO)
 * @param {number} p.xacSuat      xác suất của nhãn đó (0..1)
 * @param {boolean} p.coToSang    AI có tô đoạn nào không
 * @param {string[]} [p.luot]    các lượt thuộc kết quả đang được báo cáo
 * @param {string[]} [p.tenLienHe]  tên chỉ dùng trên thiết bị để che thêm
 * @param {HTMLElement} [p.host]  nơi gắn hộp thoại (để dùng chung biến màu của khung chat)
 * @param {(ketQua:object)=>void} [p.onDaGui]
 */
export function moBieuMauMuc1({
  nhanMoHinh, xacSuat, coToSang, luot = [], tenLienHe = [], host = document.body, onDaGui,
} = {}) {
  const ctx = { nhanMoHinh, xacSuat, coToSang, phienBan: layPhienBan() };
  const st = { mo_ta: '' };
  let maLanGui = null;
  let kieuLanGui = null; // 'phan_hoi' | 'kem_doan': đổi lựa chọn thì tạo mã lần gửi mới
  let dangGui = false;
  let buocChiaSe = false;
  let banXemTruoc = null;
  // Bước cuối luôn hỏi mỗi lần, không nhớ lựa chọn. Chỉ bỏ qua khi không có đoạn hội thoại để gửi.
  const coDoan = luot.length > 0;

  const dlg = el('dialog', 'ph-dialog');
  dlg.setAttribute('aria-labelledby', 'ph-title');
  const form = el('form', 'ph-form');
  form.method = 'dialog';
  form.noValidate = true;

  const head = el('header', 'ph-head');
  const title = el('h3', null, CHU.tieuDe);
  title.id = 'ph-title';
  const x = el('button', 'ph-x', '×');
  x.type = 'button';
  x.setAttribute('aria-label', 'Đóng');
  x.onclick = () => dlg.close();
  head.append(title, x);

  const intro = el('p', 'ph-intro');
  const [a, ten, b] = CHU.moDau(TEN_NHAN[nhanMoHinh] || nhanMoHinh);
  const chip = el('strong', `ph-nhan ph-nhan-${nhanMoHinh}`, ten);
  intro.append(a, chip, b);

  const body = el('div', 'ph-body');
  const foot = el('footer', 'ph-foot');
  const note = el('p', 'ph-note');
  const status = el('p', 'ph-status');
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  const actions = el('div', 'ph-actions');
  const huy = el('button', 'ph-huy', CHU.huy);
  huy.type = 'button';
  huy.onclick = () => {
    if (!buocChiaSe) {
      dlg.close();
      return;
    }
    buocChiaSe = false;
    status.textContent = '';
    ve();
  };
  const tuChoi = el('button', 'ph-huy ph-tu-choi', CHU.tuChoiKem);
  tuChoi.type = 'button';
  tuChoi.onclick = () => gui(false);
  const gui1 = el('button', 'ph-gui', CHU.gui);
  gui1.type = 'submit';
  actions.append(huy, tuChoi, gui1);
  foot.append(note, status, actions);

  form.append(head, intro, body, foot);
  dlg.append(form);

  function capNhatNut() {
    const ok = hopLe(st, ctx) && !dangGui;
    gui1.disabled = !ok;
    gui1.setAttribute('aria-disabled', String(!ok));
    tuChoi.hidden = !buocChiaSe;
    tuChoi.disabled = !ok;
    gui1.textContent = buocChiaSe ? CHU.dongYKem : coDoan ? CHU.tiep : CHU.gui;
    tuChoi.textContent = CHU.tuChoiKem;
    if (dangGui) (kieuLanGui === 'phan_hoi' && buocChiaSe ? tuChoi : gui1).textContent = CHU.dangGui;
    huy.textContent = buocChiaSe ? 'Quay lại' : CHU.huy;
    huy.disabled = x.disabled = dangGui;
    note.textContent = !buocChiaSe && coDoan ? CHU.nhacBuocCuoi : '';
    note.hidden = !note.textContent;
  }

  function onChon(key, ma, focusId) {
    st[key] = ma;
    for (const k of PHU_THUOC[key] || []) {
      if (k === 'mo_ta') st.mo_ta = '';
      else delete st[k];
    }
    donTrangThai(st, ctx);
    status.textContent = '';
    status.className = 'ph-status';
    maLanGui = null;
    ve();
    document.getElementById(focusId)?.focus();
  }

  function ve() {
    const parts = [];
    const loai = LOAI_BAO_CAO.filter((o) => o.hien(nhanMoHinh));
    parts.push(nhomLuaChon('bao_cao', 'Loại báo cáo', true, loai, st.bao_cao, onChon, { lon: true }));

    const { qs, coMoTa } = cauHoi(st, ctx);
    if (qs.length) {
      const nhanh = el('div', 'ph-nhanh');
      for (const q of qs) nhanh.append(nhomLuaChon(q.key, q.hoi, q.batBuoc, q.options, st[q.key], onChon));
      if (coMoTa) {
        const wrap = el('div', 'ph-mota');
        const lab = el('label', 'ph-hoi', 'Mô tả ngắn');
        lab.htmlFor = 'ph-mo-ta';
        lab.append(el('span', 'ph-bat-buoc', ' · Bắt buộc'));
        const ta = el('textarea');
        ta.id = 'ph-mo-ta';
        ta.rows = 3;
        ta.maxLength = 2 * MO_TA_TOI_DA + 40; // chừa chỗ cho ký tự hai đơn vị UTF-16 và khoảng trắng
        ta.value = st.mo_ta;
        const meta = el('div', 'ph-mota-meta');
        const nhac = el('span', 'ph-nhac', CHU.nhacMoTa);
        const dem = el('span', 'ph-dem');
        const capNhatDem = () => {
          const n = Array.from(st.mo_ta.trim()).length;
          dem.textContent = `${n}/${MO_TA_TOI_DA}`;
          dem.classList.toggle('qua', n > MO_TA_TOI_DA);
        };
        ta.oninput = () => { maLanGui = null; st.mo_ta = ta.value; capNhatDem(); capNhatNut(); };
        capNhatDem();
        meta.append(nhac, dem);
        wrap.append(lab, ta, meta);
        nhanh.append(wrap);
      }
      parts.push(nhanh);
    }

    title.textContent = CHU.tieuDe;
    intro.hidden = false;
    body.replaceChildren(...parts);
    capNhatNut();
  }

  // Bước cuối: đề xuất gửi kèm đoạn hội thoại, kèm cam kết quyền riêng tư.
  // Theo quyết định 28/11/2026, đoạn hội thoại không hiện ra ở đây. Bản che tự động vẫn được tạo
  // trên thiết bị ngay lúc này và chính bản đó được gửi khi người dùng bấm đồng ý.
  function moBuocChiaSe() {
    banXemTruoc ??= createPreview(luot, { contactNames: tenLienHe });
    const camKet = el('section', 'ph-cam-ket');
    camKet.setAttribute('aria-labelledby', 'ph-cam-ket-title');
    const camKetTitle = el('h4', null, CHU.camKetTieuDe);
    camKetTitle.id = 'ph-cam-ket-title';
    camKetTitle.tabIndex = -1;
    const moDau = CHU.camKetMoDau.map((doan) => el('p', 'ph-cam-ket-doan', doan));
    const riengTu = el('h5', 'ph-rieng-tu');
    riengTu.append(iconKhien(), CHU.camKetRiengTu);
    const list = el('ul');
    for (const dong of CHU.camKet) list.append(el('li', null, dong));
    camKet.append(camKetTitle, ...moDau, el('hr'), riengTu, list);
    buocChiaSe = true;
    title.textContent = CHU.tieuDeBuocCuoi;
    intro.hidden = true;
    body.replaceChildren(camKet);
    body.scrollTop = 0;
    camKetTitle.focus();
    capNhatNut();
  }

  async function gui(kemDoan) {
    if (dangGui || !hopLe(st, ctx)) return;
    const kieu = kemDoan ? 'kem_doan' : 'phan_hoi';
    if (kieu !== kieuLanGui) { maLanGui = null; kieuLanGui = kieu; }
    dangGui = true;
    body.inert = true;
    status.className = 'ph-status';
    capNhatNut();
    try {
      maLanGui ??= crypto.randomUUID();
      const ketQua = await guiPhanHoi(taoGoi(st, ctx, maLanGui), {
        banXemTruoc: kemDoan ? banXemTruoc : null,
        dongY: kemDoan,
      });
      const done = el('div', 'ph-done');
      done.append(el('p', 'ph-done-text', CHU.daGui));
      const dong = el('button', 'ph-gui', 'Đóng');
      dong.type = 'button';
      dong.onclick = () => dlg.close();
      done.append(dong);
      x.disabled = false;
      dangGui = false;
      title.textContent = CHU.tieuDe;
      form.replaceChildren(head, done);
      dong.focus();
      onDaGui?.(ketQua);
    } catch (err) {
      status.textContent = err?.status === 429 ? 'Bạn đã đạt giới hạn gửi hôm nay.' : CHU.loiGui;
      status.className = 'ph-status loi';
      dangGui = false;
      body.inert = false;
      capNhatNut();
    }
  }

  form.onsubmit = (e) => {
    e.preventDefault();
    if (gui1.disabled) return;
    status.textContent = '';
    if (buocChiaSe) { gui(true); return; }
    if (!coDoan) { gui(false); return; }
    try {
      moBuocChiaSe();
    } catch {
      // Không tạo được bản xem trước thì vẫn cho gửi phản hồi không kèm đoạn hội thoại.
      status.className = 'ph-status loi';
      status.textContent = CHU.loiXemTruoc;
    }
  };

  dlg.addEventListener('cancel', (event) => { if (dangGui) event.preventDefault(); });
  dlg.addEventListener('close', () => dlg.remove());
  host.append(dlg);
  ve();
  dlg.showModal();
  return dlg;
}

// Xuất để kiểm thử hoặc để phần logic dùng lại.
export { cauHoi as _cauHoi, taoGoi as _taoGoi, hopLe as _hopLe };
