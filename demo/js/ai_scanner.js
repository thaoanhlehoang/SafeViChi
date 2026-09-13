/**
 * SafeViChi AI Scanner — Chạy hoàn toàn trên trình duyệt (On-device).
 *
 * Kiến trúc:
 *   - Transformers.js: CHỈ dùng để load tokenizer (đọc tokenizer.json)
 *   - ONNX Runtime Web: load encoder/decoder trực tiếp từ R2
 *   - Normalizer: Port từ Python, xử lý teencode/lookalike/separator
 *   - Occlusion: Che từng từ để giải thích lý do cảnh báo
 *
 * Cách dùng (cho Người B):
 *   import { initModel, scanMessage } from './js/ai_scanner.js';
 *
 *   await initModel({ onStage: (stage) => console.log(stage) });
 *   const result = await scanMessage("m co bị n.gu l k z ma");
 *
 * @module ai_scanner
 */

import { AutoTokenizer, env } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/wasm';
import ortWasmModuleUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import { normalize, init as initNormalizer } from './normalizer.js';
import {
  buildModelAssetUrl,
  configuredModelOrigin,
  validateModelManifest,
} from './model_config.js';

// ============================================================
// Biến toàn cục module
// ============================================================

let tokenizer = null;
let encoderSession = null;   // ONNX InferenceSession cho encoder
let decoderSession = null;   // ONNX InferenceSession cho decoder
let hateTokenId = null;
let cleanTokenId = null;
let isModelReady = false;
let readyDetails = null;

const PROMPT_PREFIX = 'vihsd: ';
const DECODER_START_TOKEN_ID = 0n; // BigInt vì ONNX Runtime dùng int64
const OCCLUSION_WINDOW = 2;        // khớp cấu hình chính thức ở results/step4/

// ============================================================
// Khởi tạo Model
// ============================================================

/**
 * Tải model ViHateT5 ONNX và tokenizer vào bộ nhớ trình duyệt. Tokenizer và
 * runtime đến từ Pages; hai graph ONNX đến trực tiếp từ R2.
 *
 * @param {Object} [options]
 * @param {string} [options.modelOrigin] - Origin R2/custom domain; mặc định VITE_MODEL_ORIGIN
 * @param {string} [options.manifestUrl='/model-manifest.json']
 * @param {string} [options.tokenizerDir='onnx_model']
 * @param {string} [options.dictionaryUrl] - URL từ điển teencode trên Pages
 * @param {Function} [options.onStage] - idle/loading_encoder/loading_decoder/ready/error callback
 * @returns {Promise<{manifest: Object, threads: number}>}
 */
export async function initModel(options = {}) {
  const reportStage = (stage, detail = {}) => {
    if (typeof options.onStage === 'function') options.onStage(stage, detail);
  };

  if (isModelReady) {
    console.log('[SafeViChi] Model đã sẵn sàng.');
    reportStage('ready', readyDetails);
    return readyDetails;
  }

  const {
    modelOrigin: explicitModelOrigin,
    manifestUrl = '/model-manifest.json',
    tokenizerDir = 'onnx_model',
    dictionaryUrl = new URL('../teencode_dict.json', import.meta.url).href,
  } = options;
  const report = (msg) => console.log('[SafeViChi] ' + msg);
  reportStage('idle');

  try {

  const manifestResponse = await fetch(manifestUrl, { cache: 'no-cache' });
  if (!manifestResponse.ok) {
    throw new Error(`Không tải được model manifest (${manifestResponse.status}).`);
  }
  const manifest = validateModelManifest(await manifestResponse.json());
  const modelOrigin = configuredModelOrigin(explicitModelOrigin);

  // --- Bước 0: Load từ điển teencode ---
  report('Đang tải từ điển teencode...');
  await initNormalizer(dictionaryUrl);

  // --- Bước 1: Load tokenizer bằng Transformers.js ---
  report('Đang tải tokenizer...');
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.useBrowserCache = true;
  env.localModelPath = new URL('./', document.baseURI).href;
  tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir);

  // Tìm token ID cho "hate" và "clean"
  const hateEncoded = tokenizer.encode('hate', { add_special_tokens: false });
  const cleanEncoded = tokenizer.encode('clean', { add_special_tokens: false });
  hateTokenId = Number(hateEncoded[0]);
  cleanTokenId = Number(cleanEncoded[0]);
  report(`Token IDs — hate: ${hateTokenId}, clean: ${cleanTokenId}`);

  // Chỉ ship binary WASM chuẩn (~11 MiB); các binary JSEP/JSPI không cần cho
  // executionProviders=['wasm'] và có thể làm build Pages vượt giới hạn file.
  const threads = globalThis.crossOriginIsolated === true
    ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
    : 1;
  ort.env.wasm.wasmPaths = {
    mjs: new URL(ortWasmModuleUrl, document.baseURI).href,
    wasm: new URL(ortWasmUrl, document.baseURI).href,
  };
  ort.env.wasm.numThreads = threads;

  const encoderUrl = buildModelAssetUrl(modelOrigin, manifest, 'encoder');
  const decoderUrl = buildModelAssetUrl(modelOrigin, manifest, 'decoder');

  try {
    report(`Đang tải Encoder ONNX (${manifest.assets.encoder.bytes} bytes)...`);
    reportStage('loading_encoder', { bytes: manifest.assets.encoder.bytes });
    encoderSession = await ort.InferenceSession.create(encoderUrl, { executionProviders: ['wasm'] });
    report('✅ Encoder đã tải xong!');

    report(`Đang tải Decoder ONNX (${manifest.assets.decoder.bytes} bytes)...`);
    reportStage('loading_decoder', { bytes: manifest.assets.decoder.bytes });
    decoderSession = await ort.InferenceSession.create(decoderUrl, { executionProviders: ['wasm'] });
    report('✅ Decoder đã tải xong!');
  } catch (error) {
    await encoderSession?.release?.();
    await decoderSession?.release?.();
    encoderSession = null;
    decoderSession = null;
    throw error;
  }

  // Debug: in ra tên input/output của model
  report('Encoder inputs: ' + JSON.stringify(encoderSession.inputNames));
  report('Encoder outputs: ' + JSON.stringify(encoderSession.outputNames));
  report('Decoder inputs: ' + JSON.stringify(decoderSession.inputNames));
  report('Decoder outputs: ' + JSON.stringify(decoderSession.outputNames));

  isModelReady = true;
  readyDetails = { manifest, threads };
  report('🎉 Model đã sẵn sàng! Có thể quét tin nhắn.');
  reportStage('ready', readyDetails);
  return readyDetails;
  } catch (error) {
    reportStage('error', { error });
    throw error;
  }
}

// ============================================================
// Tính Logit Margin — Cốt lõi bộ phân loại
// ============================================================

/**
 * Chuyển mảng số thường sang BigInt64Array (ONNX Runtime yêu cầu int64).
 */
function toBigInt64(arr) {
  return new BigInt64Array(Array.from(arr).map(v => BigInt(v)));
}

/**
 * Tính logit margin (hate_logit - clean_logit) cho 1 câu.
 *
 * Luồng: tokenize → encoder → decoder (1 bước) → đọc logits
 *
 * @param {string} text - Văn bản đã chuẩn hóa
 * @returns {Promise<number>}
 */
async function getLogitMargin(text) {
  // Tokenize (hạ chữ thường vì tokenizer ViHateT5 không có ký tự hoa)
  const input = PROMPT_PREFIX + text.toLowerCase();
  const encoded = tokenizer(input, {
    padding: true,
    truncation: true,
    max_length: 256,
  });

  // Chuyển sang tensor ONNX Runtime (int64 = BigInt64Array)
  const seqLen = encoded.input_ids.dims
    ? encoded.input_ids.dims[1]
    : encoded.input_ids.size;

  const inputIdsData = encoded.input_ids.data || encoded.input_ids;
  const attMaskData = encoded.attention_mask.data || encoded.attention_mask;

  const inputIds = new ort.Tensor('int64', toBigInt64(inputIdsData), [1, seqLen]);
  const attentionMask = new ort.Tensor('int64', toBigInt64(attMaskData), [1, seqLen]);

  // ---- Chạy Encoder ----
  const encoderOutput = await encoderSession.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  });

  // Lấy hidden states từ encoder (tên output có thể khác tùy model)
  const encoderHidden = encoderOutput.last_hidden_state
    || encoderOutput[encoderSession.outputNames[0]];

  // ---- Chạy Decoder (1 bước duy nhất) ----
  const decoderInputIds = new ort.Tensor(
    'int64',
    new BigInt64Array([DECODER_START_TOKEN_ID]),
    [1, 1]
  );

  // Tự động tìm đúng tên input cho encoder hidden states
  const decoderInputNames = decoderSession.inputNames;
  const encoderHiddenName = decoderInputNames.find(n =>
    n.includes('encoder_hidden_states') || n.includes('encoder_output')
  ) || 'encoder_hidden_states';
  const encoderMaskName = decoderInputNames.find(n =>
    n.includes('encoder_attention_mask')
  ) || 'encoder_attention_mask';

  const decoderFeeds = {
    input_ids: decoderInputIds,
    [encoderHiddenName]: encoderHidden,
    [encoderMaskName]: attentionMask,
  };

  const decoderOutput = await decoderSession.run(decoderFeeds);

  // Lấy logits (tên output có thể là 'logits' hoặc khác)
  const logitsTensor = decoderOutput.logits
    || decoderOutput[decoderSession.outputNames[0]];
  const logits = logitsTensor.data; // Float32Array [1 x 1 x vocab_size]

  const hateLogit = logits[hateTokenId];
  const cleanLogit = logits[cleanTokenId];
  return hateLogit - cleanLogit;
}

/**
 * Tính margins cho nhiều câu (tuần tự vì browser không có batch GPU).
 */
async function getLogitMargins(texts) {
  const margins = [];
  for (const text of texts) {
    margins.push(await getLogitMargin(text));
  }
  return margins;
}

// ============================================================
// Thuật toán Occlusion — Port từ occlusion.py
// ============================================================

function splitWords(text) {
  return text.match(/\S+/g) || [];
}

async function occludeWords(words, window = 1) {
  if (words.length === 0) return [];

  const texts = [words.join(' ')];
  const windows = [];

  for (let start = 0; start < words.length; start++) {
    const end = Math.min(start + window, words.length);
    windows.push([start, end]);
    const occluded = [...words.slice(0, start), ...words.slice(end)];
    texts.push(occluded.join(' '));
  }

  const scores = await getLogitMargins(texts);
  const originalScore = scores[0];
  const occludedScores = scores.slice(1);

  const importance = new Array(words.length).fill(0.0);
  for (let i = 0; i < windows.length; i++) {
    const [start, end] = windows[i];
    const drop = originalScore - occludedScores[i];
    for (let j = start; j < end; j++) {
      importance[j] += drop;
    }
  }
  return importance;
}

function normalizeImportance(importance) {
  if (importance.length === 0) return [];
  const peak = Math.max(...importance);
  if (peak <= 0) return importance.map(() => 0.0);
  return importance.map(v => v / peak);
}

function topKSpans(importance, k = null, threshold = 0.0) {
  if (k !== null) {
    const ranked = importance
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(x => x.idx)
      .sort((a, b) => a - b);
    return ranked;
  }
  return importance.map((v, i) => (v > threshold ? i : -1)).filter(i => i >= 0);
}

// ============================================================
// API chính — scanMessage()
// ============================================================

/**
 * Quét 1 tin nhắn: chuẩn hóa → phân loại → giải thích.
 *
 * @param {string} rawText - Tin nhắn gốc
 * @param {Object} [options]
 * @param {number} [options.topK=3]
 * @param {number} [options.threshold=0.1]
 * @param {Function} [options.onStage] - Callback cho UI: normalizing, classifying, explaining, complete
 * @returns {Promise<{original, normalized, label, confidence, flaggedWords}>}
 */
export async function scanMessage(rawText, options = {}) {
  if (!isModelReady) {
    throw new Error('[SafeViChi] Model chưa được tải. Gọi initModel() trước.');
  }

  const { topK = 3, threshold = 0.1, onStage = null } = options;
  const reportStage = (stage, payload = {}) => {
    if (typeof onStage === 'function') onStage(stage, payload);
  };

  reportStage('normalizing', { originalText: rawText });
  const normalizedText = normalize(rawText);
  reportStage('classifying', { normalizedText });
  const margin = await getLogitMargin(normalizedText);
  const label = margin > 0 ? 'hate' : 'clean';
  const confidence = 1.0 / (1.0 + Math.exp(-margin));

  let flaggedWords = [];
  reportStage('explaining', { normalizedText, label });
  if (label === 'hate') {
    const words = splitWords(normalizedText);
    if (words.length > 0) {
      // window = 2: bám đúng cấu hình chính thức đã chốt ở bước 4.3
      // (src/explainer/demo.py), để demo web ra cùng cụm từ với số liệu Python.
      const importance = await occludeWords(words, OCCLUSION_WINDOW);
      const normImp = normalizeImportance(importance);
      const spans = topKSpans(normImp, topK, threshold);
      flaggedWords = spans.map(i => ({
        index: i,
        word: words[i],
        score: Math.round(normImp[i] * 100) / 100,
      }));
    }
  }

  const result = {
    original: rawText,
    normalized: normalizedText,
    // Trả kèm mảng từ đã tách để phía giao diện tô sáng THEO VỊ TRÍ.
    // Tô bằng tìm-thay chuỗi là sai: từ 1 ký tự như "M" khớp cả vào phần HTML
    // vừa chèn vào trước đó (title="Điểm: 1") và làm vỡ thẻ span.
    words: splitWords(normalizedText),
    label,
    confidence: Math.round(confidence * 10000) / 10000,
    flaggedWords,
  };
  reportStage('complete', result);
  return result;
}

export function isReady() {
  return isModelReady;
}
