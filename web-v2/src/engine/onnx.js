// Engine suy luận SafeViChi v2 (SafeVi-DualORC-CRF, mmBERT-base Hybrid INT8)
// chạy hoàn toàn trên trình duyệt qua ONNX Runtime Web (WASM SIMD, đa luồng khi có COOP/COEP).
//
// Nguồn mô hình (ưu tiên theo thứ tự):
//   1. Biến môi trường VITE_MODEL_BASE_URL (Cloudflare R2 / CDN / Hugging Face tùy chỉnh).
//   2. Đường dẫn nội bộ `/model` (Vite dev/preview và serve.py phục vụ thẳng từ models/dualorc_mmbert_base/onnx/).
//   3. Dự phòng tự động: Hugging Face Hub (`dminhnhat06/safevichi_v2/resolve/main/onnx`).
//
// Sau lần tải đầu tiên từ nguồn từ xa, tệp tokenizer & ONNX được lưu trong Cache API của trình duyệt
// để các lần mở sau hoặc khi tắt mạng vẫn chạy cục bộ mà không phát sinh request mạng.

import * as ort from 'onnxruntime-web/wasm';
import ortWasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { MMBertTokenizer } from './tokenizer.js';
import { prepareDualORCInput, decodeDualORCOutput } from './dualorc.js';

const LOCAL_MODEL_BASE = '/model';
const DEFAULT_REMOTE_BASE = 'https://huggingface.co/dminhnhat06/safevichi_v2/resolve/main/onnx';
const CACHE_NAME = 'safevichi-v2-onnx-v1';
const MODEL_FILE = 'safevichi_v2_int8.onnx';
const TOKENIZER_FILE = 'tokenizer/tokenizer.json';
const CRF_FILE = 'crf_params.json';
const MANIFEST_FILE = 'export_manifest.json';

// Dung lượng ước lượng (bytes) phòng trường hợp server không gửi Content-Length
const FALLBACK_SIZES = {
  [TOKENIZER_FILE]: 34_363_188,
  [MODEL_FILE]: 315_294_659,
};

function normalizeBaseUrl(raw) {
  const trimmed = String(raw || '').trim().replace(/\/+$/, '');
  if (!trimmed) return '';
  if (trimmed.startsWith('/')) return trimmed;
  const parsed = new URL(trimmed);
  const isLocal = parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost';
  if (parsed.protocol !== 'https:' && !isLocal) {
    throw new Error('URL nguồn mô hình ngoài localhost bắt buộc phải dùng HTTPS.');
  }
  return trimmed;
}

function joinUrl(base, rel) {
  return `${base.replace(/\/+$/, '')}/${rel.replace(/^\/+/, '')}`;
}

async function tryFetchJson(url) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) return null;
    const text = await res.text();
    if (!text.trim().startsWith('{')) return null;
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function resolveModelSource() {
  const configured = normalizeBaseUrl(import.meta.env.VITE_MODEL_BASE_URL || '');
  if (configured) {
    const crf = await tryFetchJson(joinUrl(configured, CRF_FILE));
    if (!crf) throw new Error(`Không tải được ${CRF_FILE} từ ${configured}.`);
    const manifest = await tryFetchJson(joinUrl(configured, MANIFEST_FILE));
    return { baseUrl: configured, isLocal: configured.startsWith('/'), crf, manifest };
  }

  // Ưu tiên tệp cục bộ tại /model (không phải tải 334 MiB khi chạy trên máy đã có models/)
  const localCrf = await tryFetchJson(joinUrl(LOCAL_MODEL_BASE, CRF_FILE));
  if (localCrf && localCrf.arch === 'SafeVi-DualORC-CRF') {
    const manifest = await tryFetchJson(joinUrl(LOCAL_MODEL_BASE, MANIFEST_FILE));
    return { baseUrl: LOCAL_MODEL_BASE, isLocal: true, crf: localCrf, manifest };
  }

  // Dự phòng sang Hugging Face Hub (khi deploy Cloudflare Pages không kèm tệp nặng)
  const remoteCrf = await tryFetchJson(joinUrl(DEFAULT_REMOTE_BASE, CRF_FILE));
  if (!remoteCrf) {
    throw new Error(
      'Không tìm thấy mô hình cục bộ (/model) và không kết nối được tới nguồn mô hình Hugging Face.',
    );
  }
  const manifest = await tryFetchJson(joinUrl(DEFAULT_REMOTE_BASE, MANIFEST_FILE));
  return { baseUrl: DEFAULT_REMOTE_BASE, isLocal: false, crf: remoteCrf, manifest };
}

async function openModelCache() {
  if (typeof caches === 'undefined') return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

/** Tải tệp nhị phân có báo tiến độ theo byte, đọc/lưu Cache API khi dùng nguồn từ xa. */
async function fetchBinaryWithProgress(url, { useCache, cacheKey, expectedBytes, onBytes }) {
  const cache = useCache ? await openModelCache() : null;
  const cacheReq = cacheKey || url;

  if (cache) {
    try {
      const hit = await cache.match(cacheReq);
      if (hit && hit.ok) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        onBytes(buf.byteLength, buf.byteLength);
        return { bytes: buf, fromCache: true };
      }
    } catch {
      // Bỏ qua lỗi đọc cache, tải lại từ mạng
    }
  }

  const res = await fetch(url, { cache: useCache ? 'no-cache' : 'no-store' });
  if (!res.ok) {
    throw new Error(`Tải thất bại (${res.status}) tại ${url}`);
  }

  const headerLen = Number(res.headers.get('content-length') || 0);
  const total = headerLen > 0 ? headerLen : expectedBytes;

  let bytes;
  if (!res.body) {
    bytes = new Uint8Array(await res.arrayBuffer());
    onBytes(bytes.byteLength, bytes.byteLength);
  } else {
    const reader = res.body.getReader();
    if (headerLen > 0) {
      // Cấp phát đúng 1 buffer liên tục theo Content-Length để tránh nhân đôi đỉnh RAM (~301 MiB)
      bytes = new Uint8Array(headerLen);
      let offset = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.byteLength <= bytes.byteLength) {
          bytes.set(value, offset);
          offset += value.byteLength;
        } else {
          const grown = new Uint8Array(offset + value.byteLength);
          grown.set(bytes.subarray(0, offset), 0);
          grown.set(value, offset);
          bytes = grown;
          offset += value.byteLength;
        }
        onBytes(offset, Math.max(total, offset));
      }
      if (offset < bytes.byteLength) bytes = bytes.subarray(0, offset);
    } else {
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onBytes(loaded, Math.max(total, loaded));
      }
      bytes = new Uint8Array(loaded);
      let offset = 0;
      for (const c of chunks) {
        bytes.set(c, offset);
        offset += c.byteLength;
      }
    }
  }

  if (cache) {
    try {
      await cache.put(
        cacheReq,
        new Response(bytes, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(bytes.byteLength),
          },
        }),
      );
    } catch {
      // Thiếu dung lượng lưu trữ trình duyệt (QuotaExceededError) — vẫn tiếp tục chạy bình thường
    }
  }

  return { bytes, fromCache: false };
}

let session = null;
let tokenizer = null;
let crfMeta = null;
let runtimeInfo = null;

export const onnxEngine = {
  get backend() {
    return runtimeInfo?.backend || 'wasm';
  },

  async load(onProgress) {
    if (session && tokenizer && crfMeta) return runtimeInfo;

    let lastEmitAt = 0;
    let lastEmitStage = '';
    const emit = (loaded, total, stage) => {
      if (typeof onProgress !== 'function') return;
      const now = performance.now();
      const finished = total > 0 && loaded >= total;
      if (!finished && stage === lastEmitStage && now - lastEmitAt < 80) return;
      lastEmitAt = now;
      lastEmitStage = stage;
      onProgress(loaded, total, stage);
    };

    emit(0, 1, 'Đang tìm mô hình…');
    const { baseUrl, isLocal, crf, manifest } = await resolveModelSource();
    crfMeta = crf;

    const useCache = !isLocal;
    const shaVersion = manifest?.sha256?.onnx_int8?.slice(0, 12) || 'v2';

    const tokEst = FALLBACK_SIZES[TOKENIZER_FILE];
    const onnxEst = FALLBACK_SIZES[MODEL_FILE];
    let tokTotal = tokEst;
    let onnxTotal = onnxEst;
    let tokLoaded = 0;
    let onnxLoaded = 0;

    // 1. Tải và khởi tạo tokenizer mmBERT-base (~32,8 MiB)
    const tokUrl = joinUrl(baseUrl, TOKENIZER_FILE);
    let { bytes: tokBytes } = await fetchBinaryWithProgress(tokUrl, {
      useCache,
      cacheKey: `${tokUrl}?v=${shaVersion}`,
      expectedBytes: tokEst,
      onBytes: (l, t) => {
        tokLoaded = l;
        tokTotal = t;
        emit(tokLoaded, tokTotal + onnxTotal, 'Đang tải tokenizer');
      },
    });
    emit(tokTotal, tokTotal + onnxTotal, 'Đang nạp tokenizer…');
    {
      const tokJson = JSON.parse(new TextDecoder('utf-8').decode(tokBytes));
      tokBytes = null;
      tokenizer = new MMBertTokenizer(tokJson);
    }

    // 2. Cấu hình ONNX Runtime Web (WASM SIMD, tự bật đa luồng nếu trang được cách ly COOP/COEP)
    const isolated = globalThis.crossOriginIsolated === true;
    const threads = isolated
      ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
      : 1;

    const baseHref = globalThis.location?.href;
    ort.env.wasm.wasmPaths = {
      mjs: baseHref ? new URL(ortWasmModuleUrl, baseHref).href : ortWasmModuleUrl,
      wasm: baseHref ? new URL(ortWasmUrl, baseHref).href : ortWasmUrl,
    };
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.simd = true;

    // 3. Tải mô hình SafeVi-DualORC-CRF INT8 (~300,7 MiB)
    const onnxUrl = joinUrl(baseUrl, MODEL_FILE);
    let { bytes: modelBytes, fromCache } = await fetchBinaryWithProgress(onnxUrl, {
      useCache,
      cacheKey: `${onnxUrl}?v=${shaVersion}`,
      expectedBytes: onnxEst,
      onBytes: (l, t) => {
        onnxLoaded = l;
        onnxTotal = t;
        emit(tokTotal + onnxLoaded, tokTotal + onnxTotal, isLocal ? 'Đang nạp mô hình cục bộ' : 'Đang tải mô hình ONNX');
      },
    });

    emit(tokTotal + onnxTotal, tokTotal + onnxTotal, 'Đang khởi tạo ONNX…');
    const t0 = performance.now();
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    // Giải phóng tham chiếu bộ đệm 301 MiB để GC thu hồi sau khi ORT đã nạp vào heap WASM
    modelBytes = null;

    runtimeInfo = {
      backend: `WASM · ${threads} luồng`,
      threads,
      source: isLocal ? 'local' : (fromCache ? 'cache' : 'remote'),
      loadMs: performance.now() - t0,
    };
    return runtimeInfo;
  },

  async classify(lines) {
    if (!session || !tokenizer || !crfMeta) {
      throw new Error('Mô hình ONNX chưa tải xong.');
    }
    const cleanLines = (lines || []).map((s) => String(s ?? ''));
    const prep = prepareDualORCInput(cleanLines, tokenizer, crfMeta);
    if (!prep) {
      const labels = crfMeta.labels || ['AN_TOAN', 'XUC_PHAM', 'QUAY_ROI', 'DE_DOA', 'LUA_DAO'];
      const xac_suat = Object.fromEntries(labels.map((l, i) => [l, i === 0 ? 1 : 0]));
      return { nhan: labels[0], xac_suat, rui_ro: 0, bang_chung: [], ms_model: 0 };
    }

    const { maxL, maxW, maxS, maxWordBytes } = prep.shapes;
    const a = prep.arrays;
    const feeds = {
      ids: new ort.Tensor('int64', a.ids, [1, maxL]),
      attn_mask: new ort.Tensor('bool', a.attn_mask, [1, maxL]),
      word_sub_idx: new ort.Tensor('int64', a.word_sub_idx, [1, maxW, maxS]),
      word_sub_mask: new ort.Tensor('bool', a.word_sub_mask, [1, maxW, maxS]),
      word_mask: new ort.Tensor('bool', a.word_mask, [1, maxW]),
      word_turn: new ort.Tensor('int64', a.word_turn, [1, maxW]),
      word_recency: new ort.Tensor('int64', a.word_recency, [1, maxW]),
      word_bytes: new ort.Tensor('int64', a.word_bytes, [1, maxW, maxWordBytes]),
    };

    const t0 = performance.now();
    let outputs;
    try {
      outputs = await session.run(feeds);
    } finally {
      for (const tensor of Object.values(feeds)) tensor.dispose?.();
    }

    let decoded;
    try {
      const logits = Float32Array.from(outputs.logits.data);
      const emissions = Float32Array.from(outputs.emissions.data);
      const severity = Number(outputs.severity.data[0]);
      decoded = decodeDualORCOutput({ prep, logits, emissions, severity, meta: crfMeta });
    } finally {
      for (const tensor of Object.values(outputs)) tensor.dispose?.();
    }

    return {
      ...decoded,
      ms_model: performance.now() - t0,
    };
  },
};
