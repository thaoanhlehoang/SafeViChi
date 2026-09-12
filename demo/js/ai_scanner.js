/**
 * SafeViChi AI Scanner - Chạy hoàn toàn trên trình duyệt (on-device).
 *
 * - Transformers.js: Dùng để load tokenizer từ tokenizer.json
 * - ONNX Runtime Web: Load encoder_model.onnx và decoder_model.onnx
 * - Normalizer: Chuẩn hóa teencode, ký tự đồng dạng và ký tự phân tách
 * - Occlusion: Che từng từ để khoanh vùng cụm từ gây cảnh báo
 *
 * @module ai_scanner
 */

import { normalize, init as initNormalizer } from './normalizer.js';

let tokenizer = null;
let encoderSession = null;
let decoderSession = null;
let hateTokenId = null;
let cleanTokenId = null;
let isModelReady = false;

const PROMPT_PREFIX = 'vihsd: ';
const DECODER_START_TOKEN_ID = 0n;
const OCCLUSION_WINDOW = 2; // Cửa sổ n-gram khớp với cấu hình benchmark ở step 4

/**
 * Tải model ViHateT5 ONNX và tokenizer vào bộ nhớ trình duyệt.
 *
 * @param {string} modelDir - Thư mục chứa file ONNX và tokenizer (mặc định: './onnx_model')
 * @param {Function} [onProgress] - Callback nhận % tiến trình tải
 */
export async function initModel(modelDir = './onnx_model', onProgress = null) {
  if (isModelReady) {
    console.log('[SafeViChi] Model đã sẵn sàng.');
    return;
  }

  const report = (msg) => console.log('[SafeViChi] ' + msg);

  // 1. Tải từ điển teencode
  report('Đang tải từ điển teencode...');
  const dictUrl = modelDir.replace(/\/onnx_model\/?$/, '') + '/teencode_dict.json';
  await initNormalizer(dictUrl);

  // 2. Tải tokenizer qua Transformers.js
  report('Đang tải tokenizer...');
  const { AutoTokenizer, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3'
  );
  // Trỏ localModelPath về đúng thư mục demo để đảm bảo 100% on-device
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.useBrowserCache = true;
  const dirParts = modelDir.replace(/\/+$/, '').split('/');
  env.localModelPath = dirParts.slice(0, -1).join('/') || '.';

  tokenizer = await AutoTokenizer.from_pretrained(dirParts[dirParts.length - 1]);

  const hateEncoded = tokenizer.encode('hate', { add_special_tokens: false });
  const cleanEncoded = tokenizer.encode('clean', { add_special_tokens: false });
  hateTokenId = Number(hateEncoded[0]);
  cleanTokenId = Number(cleanEncoded[0]);
  report(`Token IDs - hate: ${hateTokenId}, clean: ${cleanTokenId}`);

  // 3. Khởi tạo phiên suy luận ONNX Runtime Web
  if (typeof ort !== 'undefined') {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  }

  report('Đang tải encoder ONNX (~418 MB)...');
  if (onProgress) onProgress(5);
  encoderSession = await ort.InferenceSession.create(
    `${modelDir}/onnx/encoder_model.onnx`,
    { executionProviders: ['wasm'] }
  );
  report('Encoder đã tải xong');
  if (onProgress) onProgress(50);

  report('Đang tải decoder ONNX (~620 MB)...');
  decoderSession = await ort.InferenceSession.create(
    `${modelDir}/onnx/decoder_model.onnx`,
    { executionProviders: ['wasm'] }
  );
  report('Decoder đã tải xong');
  if (onProgress) onProgress(100);

  isModelReady = true;
  report('Model đã sẵn sàng');
}

function toBigInt64(arr) {
  return new BigInt64Array(Array.from(arr).map(v => BigInt(v)));
}

/**
 * Tính logit margin (hate_logit - clean_logit) cho một câu.
 * Chuỗi xử lý: tokenize -> encoder -> 1 bước decoder -> logits.
 */
async function getLogitMargin(text) {
  // Hạ chữ thường vì tokenizer ViHateT5 không chứa ký tự hoa
  const input = PROMPT_PREFIX + text.toLowerCase();
  const encoded = tokenizer(input, {
    padding: true,
    truncation: true,
    max_length: 256,
  });

  const seqLen = encoded.input_ids.dims
    ? encoded.input_ids.dims[1]
    : encoded.input_ids.size;

  const inputIdsData = encoded.input_ids.data || encoded.input_ids;
  const attMaskData = encoded.attention_mask.data || encoded.attention_mask;

  const inputIds = new ort.Tensor('int64', toBigInt64(inputIdsData), [1, seqLen]);
  const attentionMask = new ort.Tensor('int64', toBigInt64(attMaskData), [1, seqLen]);

  // Chạy encoder
  const encoderOutput = await encoderSession.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  });

  const encoderHidden = encoderOutput.last_hidden_state
    || encoderOutput[encoderSession.outputNames[0]];

  // Chạy decoder một bước khởi đầu
  const decoderInputIds = new ort.Tensor(
    'int64',
    new BigInt64Array([DECODER_START_TOKEN_ID]),
    [1, 1]
  );

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
  const logitsTensor = decoderOutput.logits
    || decoderOutput[decoderSession.outputNames[0]];
  const logits = logitsTensor.data;

  const hateLogit = logits[hateTokenId];
  const cleanLogit = logits[cleanTokenId];
  return hateLogit - cleanLogit;
}

async function getLogitMargins(texts) {
  const margins = [];
  for (const text of texts) {
    margins.push(await getLogitMargin(text));
  }
  return margins;
}

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
    return importance
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(x => x.idx)
      .sort((a, b) => a - b);
  }
  return importance.map((v, i) => (v > threshold ? i : -1)).filter(i => i >= 0);
}

/**
 * Quét một tin nhắn: Chuẩn hóa văn bản -> Dự đoán nhãn -> Khoanh vùng từ ngữ vi phạm.
 */
export async function scanMessage(rawText, options = {}) {
  if (!isModelReady) {
    throw new Error('[SafeViChi] Model chưa được tải. Hãy gọi initModel() trước.');
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
      // Dùng window = 2 đồng bộ với thiết lập chính thức của bước 4
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

  return {
    original: rawText,
    normalized: normalizedText,
    words: splitWords(normalizedText),
    label,
    confidence: Math.round(confidence * 10000) / 10000,
    flaggedWords,
  };
}

export function isReady() {
  return isModelReady;
}
