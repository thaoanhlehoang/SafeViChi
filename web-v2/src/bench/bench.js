// Đo SafeViChi v2 trong trình duyệt bằng đúng Web Worker của ứng dụng (src/model.worker.js):
// cùng tokenizer, cùng ONNX Runtime Web (WASM SIMD, đa luồng khi trang được cách ly COOP/COEP), cùng bộ giải mã CRF.
// Trình điều khiển Python gọi window.__bench.load() rồi window.__bench.run(rows, opts) qua DevTools Protocol.

const trangThai = document.getElementById('trang-thai');
const worker = new Worker(new URL('../model.worker.js', import.meta.url), { type: 'module' });

let nextId = 1;
const pending = new Map();
let readyWaiter = null;

worker.onmessage = ({ data }) => {
  if (!data || typeof data !== 'object') return;
  if (data.type === 'ready') {
    readyWaiter?.resolve(data);
    readyWaiter = null;
  } else if (data.type === 'error' && data.id == null) {
    readyWaiter?.reject(new Error(data.message));
    readyWaiter = null;
  } else if (data.type === 'result' || data.type === 'error') {
    const job = pending.get(data.id);
    if (!job) return;
    pending.delete(data.id);
    if (data.type === 'error') job.reject(new Error(data.message));
    else job.resolve(data);
  }
};

function classify(lines) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: 'classify', id, lines, priority: true });
  });
}

window.__bench = {
  async load() {
    trangThai.textContent = 'Đang nạp mô hình…';
    const t0 = performance.now();
    const info = await new Promise((resolve, reject) => {
      readyWaiter = { resolve, reject };
      worker.postMessage({ type: 'load' });
    });
    trangThai.textContent = `Đã nạp: ${info.backend}`;
    return {
      backend: info.backend,
      source: info.source,
      session_create_ms: info.loadMs,
      wall_load_ms: performance.now() - t0,
      cross_origin_isolated: globalThis.crossOriginIsolated === true,
      hardware_concurrency: navigator.hardwareConcurrency,
      user_agent: navigator.userAgent,
    };
  },

  // Chạy tuần tự từng đoạn (batch 1, như khi người dùng nhận tin). Lượt đầu ghi dự đoán và độ trễ;
  // nếu minSeconds > 0 thì lặp lại tập cho đủ thời gian để đo năng lượng.
  async run(rows, { minSeconds = 0, warmup = 10 } = {}) {
    for (const r of rows.slice(0, warmup)) await classify(r.text.split('\n'));
    const preds = [];
    const tStart = Date.now();
    let nTotal = 0;
    let passes = 0;
    do {
      for (const r of rows) {
        const t0 = performance.now();
        const out = await classify(r.text.split('\n'));
        const rt = performance.now() - t0;
        nTotal++;
        if (passes === 0) {
          preds.push({
            id: r.id,
            nhan: out.nhan,
            bang_chung: (out.bang_chung || []).map((s) => s.text),
            xac_suat: out.xac_suat,
            ms_khu_hoi: rt,
            ms_worker: out.ms,
            ms_mo_hinh: out.ms_model,
          });
        }
      }
      passes++;
      trangThai.textContent = `Đã chạy ${passes} lượt (${nTotal} đoạn)`;
    } while ((Date.now() - tStart) / 1000 < minSeconds);
    return { preds, n_total: nTotal, passes, t_start_ms: tStart, t_end_ms: Date.now() };
  },
};

window.__benchReady = true;
trangThai.textContent = 'Sẵn sàng';
