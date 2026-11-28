// Mô hình SafeVi-DualORC-CRF (ONNX INT8) chạy trong Web Worker để UI không bị đơ.
// Mọi suy luận diễn ra hoàn toàn trên máy người dùng: worker không gửi tin nhắn đi đâu.
//
// Hợp đồng tin nhắn (UI <-> worker):
//   UI -> { type: 'load', dungNguongDaDuyet?: boolean }
//   worker -> { type: 'progress', loaded, total, stage } ... rồi { type: 'ready', backend, source, loadMs }
//   UI -> { type: 'classify', id, lines: string[], priority?: boolean }
//   worker -> { type: 'result', id, nhan, xac_suat: { NHAN: p }, rui_ro: number,
//               bang_chung: [{ dong, start, end, text }], ms }
//   worker -> { type: 'error', id?, message }

import { onnxEngine } from './engine/onnx.js';

let loaded = false;
let runtimeInfo = null;
let loadingPromise = null;
const queue = [];
let busy = false;

async function drainQueue() {
  if (busy) return;
  busy = true;
  try {
    while (queue.length > 0) {
      const job = queue.shift();
      try {
        if (!loaded) throw new Error('Mô hình ONNX chưa tải xong.');
        const t0 = performance.now();
        const out = await onnxEngine.classify(job.lines);
        self.postMessage({ type: 'result', id: job.id, ...out, ms: performance.now() - t0 });
      } catch (err) {
        self.postMessage({ type: 'error', id: job.id, message: String(err?.message || err) });
      }
    }
  } finally {
    busy = false;
  }
}

async function handleLoad(useApprovedThresholds) {
  if (loaded) {
    self.postMessage({ type: 'ready', ...runtimeInfo });
    return;
  }
  if (!loadingPromise) {
    loadingPromise = onnxEngine
      .load((loadedBytes, totalBytes, stage) => {
        self.postMessage({ type: 'progress', loaded: loadedBytes, total: totalBytes, stage });
      }, { useApprovedThresholds })
      .then((info) => {
        runtimeInfo = info;
        loaded = true;
        return info;
      })
      .finally(() => {
        loadingPromise = null;
      });
  }
  try {
    const info = await loadingPromise;
    self.postMessage({
      type: 'ready',
      backend: info?.backend || onnxEngine.backend,
      source: info?.source,
      loadMs: info?.loadMs,
      phien_ban_mo_hinh: info?.phien_ban_mo_hinh,
      phien_ban_nguong: info?.phien_ban_nguong,
    });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message || err) });
  }
}

self.onmessage = ({ data }) => {
  if (!data || typeof data !== 'object') return;
  if (data.type === 'load') {
    handleLoad(data.dungNguongDaDuyet === true);
  } else if (data.type === 'classify') {
    // Ưu tiên yêu cầu tương tác (gửi tin, bấm vào tin) lên trước tác vụ quét nền danh sách hội thoại
    if (data.priority) {
      const idx = queue.findIndex((item) => !item.priority);
      if (idx === -1) queue.push(data);
      else queue.splice(idx, 0, data);
    } else {
      queue.push(data);
    }
    drainQueue();
  }
};
