// Mô hình chạy trong Web Worker để UI không bị đơ. Mọi xử lý ở trên máy người dùng:
// worker không gửi tin nhắn đi đâu, chỉ tải file mô hình một lần.
//
// Hợp đồng tin nhắn (UI <-> worker), giữ nguyên khi thay mô hình giả bằng ONNX:
//   UI -> {type: 'load'}
//   worker -> {type: 'progress', loaded, total} ... rồi {type: 'ready', backend}
//   UI -> {type: 'classify', id, lines: string[]}
//   worker -> {type: 'result', id, nhan, xac_suat: {NHAN: p}, bang_chung: [{dong, start, end, text}],
//             chuan_hoa: string[] (câu sau chuẩn hóa, mỗi dòng một phần tử), ms}
//   worker -> {type: 'error', id?, message}
import { mockEngine } from './engine/mock.js';

let engine = null;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      // TODO: thay bằng onnxEngine khi đã xuất mmBERT-DualORC-CRF sang ONNX.
      engine = mockEngine;
      await engine.load((loaded, total) => self.postMessage({ type: 'progress', loaded, total }));
      self.postMessage({ type: 'ready', backend: engine.backend });
    } else if (data.type === 'classify') {
      if (!engine) throw new Error('Mô hình chưa tải xong.');
      const t0 = performance.now();
      const out = await engine.classify(data.lines);
      self.postMessage({ type: 'result', id: data.id, ...out, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: data.id, message: String(err?.message || err) });
  }
};
