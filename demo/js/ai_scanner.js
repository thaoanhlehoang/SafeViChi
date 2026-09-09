/**
 * SafeViChi AI Scanner — Chạy hoàn toàn trên trình duyệt (On-device).
 *
 * Kiến trúc:
 *   - Transformers.js: CHỈ dùng để load tokenizer (đọc tokenizer.json)
 *   - ONNX Runtime Web: Load trực tiếp encoder_model.onnx + decoder_model.onnx
 *   - Normalizer: Port từ Python, xử lý teencode/lookalike/separator
 *   - Occlusion: Che từng từ để giải thích lý do cảnh báo
 *
 * Cách dùng (cho Người B):
 *   import { initModel, scanMessage } from './js/ai_scanner.js';
 *
 *   await initModel('./onnx_model', (pct) => console.log(pct + '%'));
 *   const result = await scanMessage("m co bị n.gu l k z ma");
 *
 * @module ai_scanner
 */

import { normalize, init as initNormalizer } from './normalizer.js';

// ============================================================
// Biến toàn cục module
// ============================================================

let tokenizer = null;
let encoderSession = null;   // ONNX InferenceSession cho encoder
let decoderSession = null;   // ONNX InferenceSession cho decoder
let hateTokenId = null;
let cleanTokenId = null;
let isModelReady = false;

const PROMPT_PREFIX = 'vihsd: ';
const DECODER_START_TOKEN_ID = 0n; // BigInt vì ONNX Runtime dùng int64

// ============================================================
// Khởi tạo Model
// ============================================================

/**
 * Tải model ViHateT5 ONNX và tokenizer vào bộ nhớ trình duyệt.
 *
 * @param {string} modelDir - Đường dẫn thư mục chứa ONNX + tokenizer (VD: './onnx_model')
 * @param {Function} [onProgress] - Callback nhận % tiến trình (0-100)
 */
export async function initModel(modelDir = './onnx_model', onProgress = null) {
  if (isModelReady) {
    console.log('[SafeViChi] Model đã sẵn sàng.');
    return;
  }

  const report = (msg) => console.log('[SafeViChi] ' + msg);

  // --- Bước 0: Load từ điển teencode ---
  report('Đang tải từ điển teencode...');
  const dictUrl = modelDir.replace(/\/onnx_model\/?$/, '') + '/teencode_dict.json';
  await initNormalizer(dictUrl);

  // --- Bước 1: Load tokenizer bằng Transformers.js ---
  report('Đang tải tokenizer...');
  const { AutoTokenizer, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3'
  );
  env.allowLocalModels = true;
  env.useBrowserCache = true;

  tokenizer = await AutoTokenizer.from_pretrained(modelDir);

  // Tìm token ID cho "hate" và "clean"
  const hateEncoded = tokenizer.encode('hate', { add_special_tokens: false });
  const cleanEncoded = tokenizer.encode('clean', { add_special_tokens: false });
  hateTokenId = Number(hateEncoded[0]);
  cleanTokenId = Number(cleanEncoded[0]);
  report(`Token IDs — hate: ${hateTokenId}, clean: ${cleanTokenId}`);

  // --- Bước 2: Load model ONNX bằng ONNX Runtime Web trực tiếp ---
  // (Bỏ qua hoàn toàn phần model loading của Transformers.js)

  // Cấu hình ONNX Runtime Web
  if (typeof ort !== 'undefined') {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  }

  // Load Encoder (~418 MB)
  report('Đang tải Encoder ONNX (~418 MB)...');
  if (onProgress) onProgress(5);
  encoderSession = await ort.InferenceSession.create(
    `${modelDir}/onnx/encoder_model.onnx`,
    { executionProviders: ['wasm'] }
  );
  report('✅ Encoder đã tải xong!');
  if (onProgress) onProgress(50);

  // Load Decoder (~620 MB)
  report('Đang tải Decoder ONNX (~620 MB)...');
  decoderSession = await ort.InferenceSession.create(
    `${modelDir}/onnx/decoder_model.onnx`,
    { executionProviders: ['wasm'] }
  );
  report('✅ Decoder đã tải xong!');
  if (onProgress) onProgress(100);

  // Debug: in ra tên input/output của model
  report('Encoder inputs: ' + JSON.stringify(encoderSession.inputNames));
  report('Encoder outputs: ' + JSON.stringify(encoderSession.outputNames));
  report('Decoder inputs: ' + JSON.stringify(decoderSession.inputNames));
  report('Decoder outputs: ' + JSON.stringify(decoderSession.outputNames));

  isModelReady = true;
  report('🎉 Model đã sẵn sàng! Có thể quét tin nhắn.');
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
 * @returns {Promise<{original, normalized, label, confidence, flaggedWords}>}
 */
export async function scanMessage(rawText, options = {}) {
  if (!isModelReady) {
    throw new Error('[SafeViChi] Model chưa được tải. Gọi initModel() trước.');
  }

  const { topK = 3, threshold = 0.1 } = options;

  const normalizedText = normalize(rawText);
  const margin = await getLogitMargin(normalizedText);
  const label = margin > 0 ? 'hate' : 'clean';
  const confidence = 1.0 / (1.0 + Math.exp(-margin));

  let flaggedWords = [];
  if (label === 'hate') {
    const words = splitWords(normalizedText);
    if (words.length > 0) {
      const importance = await occludeWords(words);
      const normImp = normalizeImportance(importance);
      const spans = topKSpans(normImp, topK, threshold);
      flaggedWords = spans.map(i => ({
        word: words[i],
        score: Math.round(normImp[i] * 100) / 100,
      }));
    }
  }

  return {
    original: rawText,
    normalized: normalizedText,
    label,
    confidence: Math.round(confidence * 10000) / 10000,
    flaggedWords,
  };
}

export function isReady() {
  return isModelReady;
}
