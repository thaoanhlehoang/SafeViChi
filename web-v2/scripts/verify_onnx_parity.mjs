// Kiểm thử đối chiếu end-to-end mô hình SafeVi-DualORC-CRF trên ONNX Runtime Web (WASM)
// so với dự đoán khóa `real_test_int8_all_predictions.jsonl` (từ Python ONNX Runtime).
//
// Chạy: npm run verify:parity  (từ thư mục web-v2)

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as ort from 'onnxruntime-web/wasm';
import { MMBertTokenizer } from '../src/engine/tokenizer.js';
import { prepareDualORCInput, decodeDualORCOutput } from '../src/engine/dualorc.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_DIR = resolve(HERE, '..');
const REPO_DIR = resolve(WEB_DIR, '..');
const ONNX_DIR = resolve(process.env.SAFEVICHI_ONNX_DIR || resolve(REPO_DIR, 'models', 'dualorc_mmbert_base', 'onnx'));

const modelPath = resolve(ONNX_DIR, 'safevichi_v2_int8.onnx');
const crfPath = resolve(ONNX_DIR, 'crf_params.json');
const tokPath = resolve(ONNX_DIR, 'tokenizer', 'tokenizer.json');

for (const p of [modelPath, crfPath, tokPath]) {
  if (!existsSync(p)) {
    console.error(`Thiếu tệp mô hình ONNX: ${p}`);
    console.error('Hãy chạy `python -m src.v2.download_model` ở thư mục gốc repo trước.');
    process.exit(1);
  }
}

const distDir = resolve(WEB_DIR, 'node_modules', 'onnxruntime-web', 'dist');
ort.env.wasm.wasmPaths = pathToFileURL(distDir + '/').href;
ort.env.wasm.numThreads = 1;

const tLoad = performance.now();
const crfMeta = JSON.parse(readFileSync(crfPath, 'utf8'));
const tokenizer = new MMBertTokenizer(JSON.parse(readFileSync(tokPath, 'utf8')));
const session = await ort.InferenceSession.create(readFileSync(modelPath), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
console.log(`Đã nạp tokenizer + ONNX INT8 (WASM) trong ${(performance.now() - tLoad).toFixed(0)} ms`);

const slices = [
  ['C0', resolve(REPO_DIR, 'data', 'v2', 'test_day_du.jsonl'), 6],
  ['C1', resolve(REPO_DIR, 'data', 'v2', 'test_perturb_C1.jsonl'), 6],
  ['C2', resolve(REPO_DIR, 'data', 'v2', 'test_perturb_C2.jsonl'), 6],
];

const refPath = resolve(REPO_DIR, 'results', 'v2', 'official_eval', 'dualorc_mmbert_base', 'test', 'real_test_int8_all_predictions.jsonl');
const refById = new Map();
if (existsSync(refPath)) {
  for (const line of readFileSync(refPath, 'utf8').replace(/^\uFEFF/, '').split(/\r?\n/)) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    refById.set(rec.id, rec);
  }
}

let total = 0;
let labelMatch = 0;
let verbatimOk = 0;
let latencySum = 0;

for (const [sliceName, slicePath, limit] of slices) {
  if (!existsSync(slicePath)) continue;
  const rows = readFileSync(slicePath, 'utf8')
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .slice(0, limit)
    .map((l) => JSON.parse(l));

  for (const row of rows) {
    const lines = row.text.split('\n');
    const prep = prepareDualORCInput(lines, tokenizer, crfMeta);
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
    const out = await session.run(feeds);
    latencySum += performance.now() - t0;

    const dec = decodeDualORCOutput({
      prep,
      logits: out.logits.data,
      emissions: out.emissions.data,
      severity: out.severity.data[0],
      meta: crfMeta,
    });

    for (const t of Object.values(feeds)) t.dispose?.();
    for (const t of Object.values(out)) t.dispose?.();

    total++;
    const allVerbatim = dec.bang_chung.every((sp) => lines[sp.dong]?.slice(sp.start, sp.end) === sp.text);
    if (allVerbatim) verbatimOk++;
    else console.error(`[VERBATIM FAIL] ${sliceName} ${row.id}`);

    const ref = refById.get(row.id);
    if (ref) {
      if (dec.nhan === ref.nhan) labelMatch++;
      else console.error(`[LABEL DIFF] ${sliceName} ${row.id}: ref=${ref.nhan} wasm=${dec.nhan}`);
    }
  }
}

await session.release();

console.log(`Đã kiểm tra ${total} mẫu (C0/C1/C2):`);
console.log(`  - Khớp nhãn với bản khóa INT8: ${labelMatch}/${refById.size ? total : 0}`);
console.log(`  - Bằng chứng verbatim 100%:    ${verbatimOk}/${total}`);
console.log(`  - Độ trễ trung bình (1 luồng): ${(latencySum / Math.max(1, total)).toFixed(1)} ms/mẫu`);

if (verbatimOk !== total || (refById.size > 0 && labelMatch !== total)) {
  process.exit(1);
}
