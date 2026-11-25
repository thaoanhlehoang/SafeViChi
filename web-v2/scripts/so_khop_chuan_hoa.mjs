// So khớp normalizer.js với Python. Chạy: node scripts/so_khop_chuan_hoa.mjs <file_cau.json> <file_python_ra.json>
import { readFileSync } from 'node:fs';
import { normalize, setTeencodeDict } from '../src/normalizer.js';

setTeencodeDict(JSON.parse(readFileSync(new URL('../public/teencode_dict.json', import.meta.url), 'utf8')));
const inputs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const expected = JSON.parse(readFileSync(process.argv[3], 'utf8'));
let bad = 0;
let badMap = 0;
inputs.forEach((s, i) => {
  const { text, map } = normalize(s);
  if (text !== expected[i]) {
    if (bad++ < 10) console.log('LỆCH', JSON.stringify(s), '\n  py:', JSON.stringify(expected[i]), '\n  js:', JSON.stringify(text));
  }
  if (map.length !== [...text].length || map.some(([a, b], k) => a < 0 || b > s.length || a > b || (k && a < map[k - 1][0])) ) badMap++;
});
console.log(`${inputs.length} dòng, lệch ${bad}, map sai ${badMap}`);
process.exit(bad || badMap ? 1 : 0);
