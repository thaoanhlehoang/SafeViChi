// Xuất mẫu phản hồi đã duyệt (có kèm đoạn hội thoại, trạng thái da_duyet) từ D1 cục bộ sang JSONL
// theo định dạng dữ liệu v2 để đưa vào TRAIN. Chạy tay, không có API xuất hàng loạt.
//
//   node scripts/xuat_mau_da_duyet.mjs --out <tệp.jsonl> [--db DEMO_FEEDBACK_DB]
//
// - Mọi mẫu có nguon="phan_hoi", cach_gan="nguoi", split="train". Không bao giờ ghi vào test.
// - Bỏ mẫu có text trùng nguyên văn với một câu trong data/v2/test*.jsonl hoặc kho test.
// - Không ghi đè tệp đã có. Sau khi xuất vẫn phải khử trùng gần đúng cùng các nguồn khác trước khi gộp vào train.
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DATA_V2 = resolve(ROOT, '..', 'data', 'v2');

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
}

const out = arg('--out');
const db = arg('--db', 'DEMO_FEEDBACK_DB');
if (!out) {
  console.error('Cần --out <tệp.jsonl>. Ví dụ: --out ../data/v2/interim/phan_hoi_da_duyet_2026-11-28.jsonl');
  process.exit(2);
}
const outPath = resolve(out);
if (existsSync(outPath)) {
  console.error('Tệp đã tồn tại, không ghi đè: ' + outPath);
  process.exit(2);
}
if (!/^[A-Z0-9_]+$/.test(db)) {
  console.error('Tên binding D1 không hợp lệ.');
  process.exit(2);
}

const SQL = `SELECT p.id, p.ngay, p.du_lieu, d.nhan, d.bang_chung
FROM phan_hoi p JOIN luot_duyet d ON d.id = p.quyet_dinh_id
WHERE p.loai = 'muc_2' AND p.trang_thai = 'da_duyet'
ORDER BY p.ngay, p.id;`;

const tmp = mkdtempSync(join(tmpdir(), 'safevichi-xuat-'));
let rows;
try {
  const sqlFile = join(tmp, 'xuat.sql');
  writeFileSync(sqlFile, SQL, 'utf8');
  const args = ['--yes', 'wrangler@4.169.0', 'd1', 'execute', db, '--local', '--json', '--file', sqlFile];
  // Trên Windows npx là tệp .cmd nên phải qua cmd.exe; `db` đã kiểm tra bằng regex, đường dẫn tạm được đặt trong ngoặc kép.
  const [command, commandArgs] = process.platform === 'win32'
    ? ['cmd.exe', ['/d', '/s', '/c', 'npx ' + args.map((a) => (a === sqlFile ? '"' + a + '"' : a)).join(' ')]]
    : ['npx', args];
  const stdout = execFileSync(command, commandArgs, {
    cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], windowsVerbatimArguments: process.platform === 'win32',
  });
  rows = JSON.parse(stdout.slice(stdout.indexOf('[')))[0].results;
} finally {
  rmSync(tmp, { recursive: true, force: true });
}

// Câu trong tập test: không được để mẫu phản hồi trùng nguyên văn lọt vào train.
const testTexts = new Set();
const testFiles = existsSync(DATA_V2)
  ? readdirSync(DATA_V2).filter((name) => /^test.*\.jsonl$/.test(name)).map((name) => join(DATA_V2, name))
  : [];
const khoTest = join(DATA_V2, 'interim', 'kho_test.jsonl');
if (existsSync(khoTest)) testFiles.push(khoTest);
for (const file of testFiles) {
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { testTexts.add(JSON.parse(line).text); } catch { /* bỏ dòng hỏng */ }
  }
}

const dem = { da_duyet: rows.length, xuat: 0, trung_test: 0, bang_chung_loi: 0 };
const records = [];
for (const row of rows) {
  const feedback = JSON.parse(row.du_lieu);
  const text = feedback.doan_da_che;
  const lines = text.split('\n');
  const evidence = JSON.parse(row.bang_chung);
  if (testTexts.has(text)) { dem.trung_test += 1; continue; }
  // Mỗi cụm bằng chứng phải nằm gọn trong một dòng, đúng vị trí đã lưu.
  if (!evidence.every(({ dong, start, end, text: span }) => lines[dong]?.slice(start, end) === span)) {
    dem.bang_chung_loi += 1;
    continue;
  }
  records.push({
    id: 'v2_ph_' + row.id.replaceAll('-', '').slice(0, 12),
    text,
    nhan: row.nhan,
    bang_chung: evidence.map(({ text: span }) => span),
    nguon: 'phan_hoi',
    nguon_id: row.id,
    cach_gan: 'nguoi',
    the: [],
    so_dong: lines.length,
    bien_dang: 'C0',
    nhom_trung: 'phan_hoi_' + row.id,
    split: 'train',
  });
}
dem.xuat = records.length;
writeFileSync(outPath, records.map((r) => JSON.stringify(r)).join('\n') + (records.length ? '\n' : ''), { encoding: 'utf8', flag: 'wx' });
console.log(JSON.stringify(dem));
console.log('Đã ghi ' + records.length + ' mẫu vào ' + outPath + ' (split=train). Nhớ khử trùng gần đúng trước khi gộp vào train.');
