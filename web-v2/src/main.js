import '@fontsource-variable/plus-jakarta-sans';
import '@fontsource/be-vietnam-pro/400.css';
import '@fontsource/be-vietnam-pro/500.css';
import '@fontsource/be-vietnam-pro/600.css';
import '@fontsource/be-vietnam-pro/700.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-mono/500.css';
import '@fontsource/ibm-plex-mono/600.css';
import './style.css';
import './chat.css';
import { LABELS, LABEL_INFO } from './labels.js';
import { setupLanding } from './landing.js';
import { setupChat } from './chat.js';

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
    setStatus({ kind: 'loading', text: formatProgress(data) });
  } else if (data.type === 'ready') {
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
  setStatus({ kind: 'loading', text: 'Đang khởi động ONNX…' });
  ensureWorker().postMessage({ type: 'load' });
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
setupLanding({ onStart: startApp });

// Thẻ giới thiệu 5 nhãn ở phần thông tin.
$('label-list').replaceChildren(...Object.entries(LABELS).map(([key, info]) => {
  const li = document.createElement('li');
  li.style.setProperty('--c', info.mau);
  const h = document.createElement('h3');
  h.textContent = info.ten;
  const p = document.createElement('p');
  p.textContent = LABEL_INFO[key];
  const a = document.createElement('p');
  a.className = 'label-action';
  a.textContent = info.hanh_dong;
  li.append(h, p, a);
  return li;
}));
