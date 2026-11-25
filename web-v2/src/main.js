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
const setStatus = (s) => statusListeners.forEach((f) => f(s));

// Gọi mô hình trong worker, trả về Promise kết quả.
function classify(lines) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'classify', id, lines });
  });
}

function onWorkerMessage({ data }) {
  if (data.type === 'progress') setStatus({ kind: 'loading', text: `Đang tải ${Math.round((100 * data.loaded) / data.total)}%` });
  else if (data.type === 'ready') setStatus(data.backend === 'mock' ? { kind: 'mock', text: 'Mô hình giả' } : { kind: 'real', text: `Trên máy · ${data.backend}` });
  else if (data.type === 'result') { pending.get(data.id)?.resolve(data); pending.delete(data.id); }
  else if (data.type === 'error') {
    if (data.id && pending.has(data.id)) { pending.get(data.id).reject(new Error(data.message)); pending.delete(data.id); }
    else setStatus({ kind: 'error', text: 'Lỗi mô hình' });
  }
}

// Chỉ tải mô hình khi người dùng vào màn chat, để trang đầu mở thật nhanh.
let chatReady = false;
function startApp() {
  if (!chatReady) { chatReady = true; setupChat({ classify, onEngineStatus: (f) => statusListeners.push(f) }); }
  if (worker) return;
  worker = new Worker(new URL('./model.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.postMessage({ type: 'load' });
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
