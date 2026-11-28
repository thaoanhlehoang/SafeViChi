import { LABELS } from './labels.js';
import { setupLanding } from './landing.js';
import { setupChat } from './chat.js';
import { drawClosing } from './about-art.js';
import { datPhienBan } from './phan_hoi/api.js';
import { setupAppTabs } from './quan_tri/tab_nhung.js';

const $ = (id) => document.getElementById(id);
let worker = null;
let nextId = 0;
const pending = new Map();
const statusListeners = [];
let lastStatus = null;

function setStatus(s) {
  lastStatus = s;
  statusListeners.forEach((f) => f(s));
}

function rejectAllPending(reason) {
  for (const [, p] of pending) p.reject(new Error(reason));
  pending.clear();
}

// Gọi mô hình ONNX trong worker, trả về Promise kết quả.
function classify(lines, { priority = false } = {}) {
  return new Promise((resolve, reject) => {
    if (!worker) {
      reject(new Error('Worker mô hình chưa khởi chạy.'));
      return;
    }
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'classify', id, lines, priority });
  });
}

function formatProgress(data) {
  const stage = data.stage || 'Đang tải mô hình';
  if (data.total > 1 && data.loaded < data.total) {
    const pct = Math.min(99, Math.max(0, Math.round((100 * data.loaded) / data.total)));
    return `${stage} · ${pct}%`;
  }
  return stage;
}

function onWorkerMessage({ data }) {
  if (data.type === 'progress') {
    setStatus({ kind: 'loading', text: formatProgress(data), stage: data.stage, loaded: data.loaded, total: data.total });
  } else if (data.type === 'ready') {
    datPhienBan({ mo_hinh: data.phien_ban_mo_hinh, nguong: data.phien_ban_nguong });
    setStatus({ kind: 'real', text: 'SafeViChi v2' });
  } else if (data.type === 'result') {
    pending.get(data.id)?.resolve(data);
    pending.delete(data.id);
  } else if (data.type === 'error') {
    if (data.id && pending.has(data.id)) {
      pending.get(data.id).reject(new Error(data.message));
      pending.delete(data.id);
    } else {
      setStatus({ kind: 'error', text: 'Lỗi nạp mô hình', message: data.message || 'Không tải được mô hình ONNX.' });
    }
  }
}

function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./model.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (ev) => {
    const msg = ev?.message || 'Worker ONNX gặp sự cố không mong muốn.';
    rejectAllPending(msg);
    setStatus({ kind: 'error', text: 'Lỗi worker', message: msg });
  };
  return worker;
}

function loadModel() {
  setStatus({ kind: 'loading', text: 'Đang khởi động ONNX…', stage: 'Đang khởi động…', loaded: 0, total: 0 });
  ensureWorker().postMessage({ type: 'load', dungNguongDaDuyet: true });
}

// Chỉ tải mô hình khi người dùng vào màn chat, để trang đầu mở thật nhanh.
let chatReady = false;
function startApp() {
  if (!chatReady) {
    chatReady = true;
    setupChat({
      classify,
      retryLoad: loadModel,
      onEngineStatus: (f) => {
        statusListeners.push(f);
        if (lastStatus) f(lastStatus);
      },
    });
  }
  if (!worker) loadModel();
}

// Màu nhãn cho trang tổng quan (lấy từ labels.js để đổi một chỗ là cả trang ăn theo).
const VAR = { AN_TOAN: '--at', XUC_PHAM: '--xp', QUAY_ROI: '--qr', DE_DOA: '--dd', LUA_DAO: '--ld' };
const about = $('thong-tin');
for (const [k, v] of Object.entries(VAR)) about.style.setProperty(v, LABELS[k].mau);

// Vẽ nền trang đầu trước khi dựng các minh họa phía dưới để khung hình đầu hiện ngay.
setupLanding({ onStart: startApp });

// Tab kiểm duyệt và quản trị nằm cạnh tab nhắn tin, đổi tại chỗ.
setupAppTabs();

// Sơ đồ kiến trúc: đường cong nối các khối, tính theo vị trí thật trong khung sơ đồ.
// Hiệu ứng: một luồng sáng chạy từ đầu vào tới đầu ra, khối nào được luồng chạm tới thì sáng lên, rồi lặp lại.
const EDGES = [['in', 'bert', 0], ['in', 'cnn', 0], ['bert', 'fuse-dot', 1], ['cnn', 'fuse-dot', 1], ['fuse-dot', 'cls', 2], ['fuse-dot', 'bio', 2], ['cls', 'out', 3], ['bio', 'out', 3]];
const STAGE_NODES = [['in'], ['bert', 'cnn'], ['fuse'], ['cls', 'bio'], ['out']];
const STEP = 0.95, TRAVEL = 0.75, CYCLE = STEP * 4 + 2.2; // giây
const archBox = $('arch');
const SVGNS = 'http://www.w3.org/2000/svg';
let flows = [];
function drawArch() {
  const svg = archBox.querySelector('.arch-lines');
  svg.replaceChildren();
  flows = [];
  if (getComputedStyle(svg).display === 'none') return;
  const box = archBox.getBoundingClientRect();
  const rect = (k) => {
    const r = archBox.querySelector(`[data-n="${k}"]`).getBoundingClientRect();
    return { l: r.left - box.left, r: r.right - box.left, y: r.top - box.top + r.height / 2 };
  };
  const mkp = (cls, d) => { const p = document.createElementNS(SVGNS, 'path'); p.setAttribute('class', cls); p.setAttribute('d', d); svg.append(p); return p; };
  for (const [a, b, stage] of EDGES) {
    const A = rect(a), B = rect(b);
    const x1 = A.r, x2 = B.l, mx = (x1 + x2) / 2;
    const d = `M${x1},${A.y} C${mx},${A.y} ${mx},${B.y} ${x2},${B.y}`;
    mkp('base', d);
    const flow = mkp('flow', d);
    const len = flow.getTotalLength();
    flow.style.strokeDasharray = `${len} ${len}`;
    flow.style.strokeDashoffset = len;
    const spark = document.createElementNS(SVGNS, 'circle');
    spark.setAttribute('class', 'spark');
    spark.setAttribute('r', 3.6);
    spark.style.opacity = 0;
    svg.append(spark);
    flows.push({ flow, spark, len, stage });
  }
}
const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
const stageEls = STAGE_NODES.map((ks) => ks.map((k) => archBox.querySelector(`[data-n="${k}"]`)));
let t0 = 0, raf = 0;
function frame(now) {
  const t = ((now - t0) / 1000) % CYCLE;
  const fade = t > CYCLE - 0.5 ? 1 - (t - (CYCLE - 0.5)) / 0.5 : 1;
  stageEls.forEach((els, i) => els.forEach((n) => n.classList.toggle('on', t >= i * STEP + (i ? TRAVEL : 0) && t < CYCLE - 0.35)));
  for (const f of flows) {
    const s = f.stage * STEP + 0.15;
    const p = Math.min(1, Math.max(0, (t - s) / TRAVEL));
    const e = ease(p);
    f.flow.style.strokeDashoffset = f.len * (1 - e);
    f.flow.style.opacity = fade;
    if (p > 0 && p < 1) {
      const pt = f.flow.getPointAtLength(f.len * e);
      f.spark.setAttribute('cx', pt.x);
      f.spark.setAttribute('cy', pt.y);
      f.spark.style.opacity = 1;
    } else f.spark.style.opacity = 0;
  }
  raf = requestAnimationFrame(frame);
}
function startArch() { if (!raf) { t0 = performance.now(); raf = requestAnimationFrame(frame); } }
function stopArch() { cancelAnimationFrame(raf); raf = 0; }
drawArch();
new ResizeObserver(drawArch).observe(archBox);
document.fonts?.ready.then(drawArch);
if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
  stageEls.flat().forEach((n) => n.classList.add('on'));
} else {
  new IntersectionObserver(([e]) => (e.isIntersecting ? startArch() : stopArch()), { threshold: 0.35 }).observe(archBox);
}

// Kết quả: khi bảng hiện ra, các số đếm từ 0 lên rồi hàng SafeViChi v2 sáng lên.
const resTable = document.querySelector('#res-main .tbl');
const numCells = [...resTable.querySelectorAll('td')].filter((td) => /^\d+(,\d+)?%?$/.test(td.textContent.trim()))
  .map((td) => { const t = td.textContent.trim(); return { td, t, v: parseFloat(t.replace(',', '.')), dec: (t.split(',')[1] || '').replace('%', '').length, pct: t.endsWith('%') }; });
let countRaf = 0;
function resetCount() { cancelAnimationFrame(countRaf); for (const c of numCells) c.td.textContent = c.t; resTable.classList.remove('lit'); }
function countUp() {
  resetCount();
  const t0 = performance.now(), D = 1100;
  const step = (now) => {
    const k = Math.min(1, (now - t0) / D), e = 1 - (1 - k) ** 3;
    for (const c of numCells) c.td.textContent = (c.v * e).toFixed(c.dec).replace('.', ',') + (c.pct ? '%' : '');
    if (k < 1) countRaf = requestAnimationFrame(step);
    else { for (const c of numCells) c.td.textContent = c.t; resTable.classList.add('lit'); }
  };
  countRaf = requestAnimationFrame(step);
}
if (matchMedia('(prefers-reduced-motion: reduce)').matches) resTable.classList.add('lit');
else new IntersectionObserver(([e]) => (e.isIntersecting ? countUp() : resetCount()), { threshold: 0.3 }).observe(resTable);

// Hình minh họa hoạt hình cùng nét với trang chủ.
// Khung chat mẫu ở trang cuối: bong bóng tin nhắn ôm sát dòng chữ dài nhất (CSS không tự co khi chữ xuống dòng).
function fitBubbles() {
  for (const b of document.querySelectorAll('.end-demo .ed-msg')) {
    b.style.width = '';
    const r = document.createRange();
    r.selectNodeContents(b);
    const rects = [...r.getClientRects()];
    if (!rects.length) continue;
    const cs = getComputedStyle(b);
    const pad = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight) + parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth);
    const w = Math.max(...rects.map((x) => x.right)) - Math.min(...rects.map((x) => x.left));
    b.style.width = `${Math.ceil(w + pad + 1)}px`;
  }
}
fitBubbles();
document.fonts?.ready.then(fitBubbles);
addEventListener('resize', fitBubbles);

const endArt = $('art-closing');
let endW = 0, endT = 0;
new ResizeObserver(() => {
  clearTimeout(endT);
  endT = setTimeout(() => { if (Math.abs(endArt.clientWidth - endW) > 40 || !endW) { endW = endArt.clientWidth; drawClosing(endArt); } }, 150);
}).observe(endArt);
