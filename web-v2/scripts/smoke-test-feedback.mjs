import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { access, appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { randomBytes, randomUUID } from 'node:crypto';
import { parseEnv } from 'node:util';
import { sha256 } from '../src/feedback/contract.js';
import { redactPreviewSelection, createPreview, createLevel1Payload, createLevel2Payload } from '../src/feedback/client.js';
import { applyThreshold, fetchActiveThresholds } from '../src/feedback/thresholds.js';

const envFile = new URL('../.env', import.meta.url);
const defaultTokenFile = new URL('../.wrangler/feedback-demo-tokens.json', import.meta.url);
// Dùng lại token đã tạo nếu chạy trên bản thử trước khi đổi tên tệp.
const legacyTokenFile = new URL('../.wrangler/phan-hoi-demo-tokens.json', import.meta.url);
const tokenFile = !existsSync(defaultTokenFile) && existsSync(legacyTokenFile) ? legacyTokenFile : defaultTokenFile;
if (process.argv[2] === '--init-demo') {
  const contents = await readFile(envFile, 'utf8').catch((err) => {
    if (err.code === 'ENOENT') return '';
    throw err;
  });
  const secret = parseEnv(contents).FEEDBACK_RATE_SECRET;
  if (secret === undefined) {
    const separator = contents && !contents.endsWith('\n') ? '\n' : '';
    await appendFile(envFile, separator + 'FEEDBACK_RATE_SECRET=' + randomBytes(32).toString('base64url') + '\n');
    console.log('Đã tạo cấu hình giới hạn tần suất trong .env cho demo.');
  } else {
    assert.ok(secret.length >= 32, 'FEEDBACK_RATE_SECRET trong .env cần ít nhất 32 ký tự.');
    console.log('Dùng lại cấu hình giới hạn tần suất cục bộ.');
  }
  process.exit(0);
}
if (process.argv[2] === '--init') {
  await mkdir(new URL('../.wrangler/', import.meta.url), { recursive: true });
  const [hasEnv, hasTokens] = await Promise.all([envFile, tokenFile].map((file) =>
    access(file).then(() => true, (error) => {
      if (error.code === 'ENOENT') return false;
      throw error;
    })));
  if (hasEnv && hasTokens) {
    console.log('Dùng lại .env và tệp token kiểm thử hiện có.');
    process.exit(0);
  }
  assert.ok(!hasEnv && !hasTokens, 'Cấu hình cục bộ đang có sẵn một phần. Giữ nguyên tệp hiện có.');
  const tokens = Object.fromEntries(['duyet_a', 'duyet_b', 'quan_tri']
    .map((id) => [id, randomBytes(32).toString('base64url')]));
  const roleHashes = Object.fromEntries(await Promise.all(Object.entries(tokens).map(async ([id, token]) =>
    [await sha256(token), { id, vai_tro: id === 'quan_tri' ? 'quan_tri' : 'nguoi_duyet' }])));
  await writeFile(envFile, 'FEEDBACK_RATE_SECRET=' + randomBytes(32).toString('base64url') +
    '\nINTERNAL_TOKEN_HASHES=\'' + JSON.stringify(roleHashes) + '\'\n', { flag: 'wx' });
  await writeFile(tokenFile, JSON.stringify(tokens), { flag: 'wx' });
  console.log('Đã tạo .env và tệp token kiểm thử trong .wrangler/.');
  process.exit(0);
}

const base = new URL(process.argv[2] ?? 'http://127.0.0.1:8788');
assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(base.hostname), 'Chỉ chạy với máy chủ cục bộ.');
const { demo } = await call('noi-bo/che-do');
const keys = demo ? null : JSON.parse(await readFile(tokenFile, 'utf8'));

async function call(path, { body, role, status = 200 } = {}) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (role && demo) headers['X-SafeViChi-Demo-Role'] = role;
  if (role && !demo) headers.Authorization = 'Bearer ' + keys[role];
  const res = await fetch(new URL('/api/' + path, base), {
    method: body === undefined ? 'GET' : 'POST', headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(15_000),
  });
  assert.equal(res.headers.get('Cache-Control'), 'no-store');
  const data = await res.json();
  assert.equal(res.status, status, path + ' · ' + JSON.stringify(data));
  return data;
}

const map = new Map();
const storage = { getItem: (key) => map.get(key), setItem: (key, value) => map.set(key, value) };
const formData = {
  ma_bao_cao: 'tin_nhan_an_toan', nhan_mo_hinh: 'XUC_PHAM', nhan_de_xuat: 'AN_TOAN',
  ma_ly_do: 'dua_noi_than_mat', ma_chi_tiet: 'luot_truoc',
  co_to_sang: true, khoang_diem: 8, phien_ban_mo_hinh: 'demo@' + randomUUID(), phien_ban_nguong: 'baseline',
};
const rest = 'phan-hoi';
const level1 = await createLevel1Payload(formData, { storage });
await call(rest, { body: level1, status: 201 });
assert.equal((await call(rest, { body: level1 })).gui_lai, true);
await call(rest, { body: { ...level1, ma_chi_tiet: null }, status: 409 });
await call(rest, { body: { ...level1, text: 'Tin giả lập' }, status: 400 });
await call(rest, { body: { ...level1, doan_da_che: null }, status: 400 });
await call(rest, { body: { ...level1, ma_bao_cao: ['tin_nhan_an_toan'] }, status: 400 });
const unsureReport = await createLevel1Payload({
  ...formData, ma_bao_cao: 'khac_chua_chac', nhan_de_xuat: null, ma_ly_do: 'chua_chac', ma_chi_tiet: null,
}, { storage });
await call(rest, { body: unsureReport, status: 201 });
const incorrectCategory = await createLevel1Payload({
  ...formData, ma_bao_cao: 'sai_loai', nhan_de_xuat: 'LUA_DAO', ma_ly_do: null, ma_chi_tiet: 'mao_danh',
}, { storage });
await call(rest, { body: incorrectCategory, status: 201 });
const missedHarm = await createLevel1Payload({
  ...formData, ma_bao_cao: 'bo_sot', nhan_mo_hinh: 'AN_TOAN', nhan_de_xuat: 'LUA_DAO', co_to_sang: false,
  ma_ly_do: 'dau_hieu_truc_tiep', ma_chi_tiet: 'mao_danh',
}, { storage });
await call(rest, { body: missedHarm, status: 201 });
const highlightIssue = await createLevel1Payload({
  ...formData, ma_bao_cao: 'loi_to_sang', nhan_de_xuat: 'XUC_PHAM', ma_ly_do: 'to_thieu', ma_chi_tiet: null,
}, { storage });
await call(rest, { body: highlightIssue, status: 201 });

// Còn 15 lượt sau 5 lượt hợp lệ. Các gói cạnh tranh không được vượt trần.
const calls = await Promise.all(Array.from({ length: 20 }, async () => {
  const payload = { ...level1, ma_lan_gui: randomUUID() };
  const response = await fetch(new URL('/api/phan-hoi', base), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    signal: AbortSignal.timeout(15_000),
  });
  await response.json();
  return response.status;
}));
assert.equal(calls.filter((status) => status === 201).length, 15);
assert.equal(calls.filter((status) => status === 429).length, 5);
assert.equal((await call(rest, { body: level1 })).gui_lai, true);
await call('thong-ke', { status: 401 });
const stats = await call('thong-ke?theo=phien_ban', { role: 'duyet_a' });
assert.equal(stats.nhom.find((row) => row.phien_ban_mo_hinh === formData.phien_ban_mo_hinh).so_luot, 20);
console.log('Mức 1 đạt: 5 nhánh, loại trường thừa, gửi lại, giới hạn và thống kê.');

const preview = createPreview(['Chị Lan gọi 0912345678', 'Có dấu hiệu đe dọa trong ca giả lập']);
const finalPreview = redactPreviewSelection(preview, 0, preview.text.indexOf('gọi'), preview.text.indexOf('gọi') + 3);
const level2 = await createLevel2Payload(formData, finalPreview, { consent: true, storage });
await call(rest, { body: { ...level2, dong_y: null }, status: 400 });
await call(rest, { body: { ...level2, doan_da_che: preview.text }, status: 400 });
const raw = 'Số điện thoại giả lập 0912345678';
await call(rest, { body: { ...level2, doan_da_che: raw,
  dong_y: { ...level2.dong_y, ban_xem_truoc_sha256: await sha256(raw) } }, status: 400 });
const { id } = await call(rest, { body: level2, status: 201 });
await call('noi-bo/phien', { status: 401 });
if (demo) await call('noi-bo/phien', { role: 'vai_tro_khac', status: 401 });
assert.deepEqual(await call('noi-bo/phien', { role: 'duyet_a' }), { id: 'duyet_a', vai_tro: 'nguoi_duyet' });
assert.deepEqual(await call('noi-bo/phien', { role: 'quan_tri' }), { id: 'quan_tri', vai_tro: 'quan_tri' });
await call('duyet', { role: 'quan_tri', status: 403 });
const queue = await call('duyet?sau_ngay=1900-01-01&sau_id=00000000-0000-4000-8000-000000000001', { role: 'duyet_a' });
assert.ok(queue.hang_doi.length > 0 && queue.hang_doi.length <= 25);
assert.ok(queue.hang_doi.every((row) => !Object.hasOwn(row, 'doan_da_che') && !Object.hasOwn(row, 'ly_do_khac')));
assert.deepEqual(Object.keys(queue.dem).sort(), ['cho_duyet', 'da_duyet']);
assert.ok(queue.hang_doi.every((row, i, rows) => i === 0 || rows[i - 1].ngay <= row.ngay));
await call('duyet?trang_thai=sai', { role: 'duyet_a', status: 400 });
await call('duyet?sau_id=sai', { role: 'duyet_a', status: 400 });
await call('nhat-ky', { role: 'duyet_a', status: 403 });
await call('duyet/' + id, { status: 401 });
await call('duyet/' + id, { role: 'quan_tri', status: 403 });
const { mau: sample } = await call('duyet/' + id, { role: 'duyet_a' });
assert.equal(sample.doan_da_che, finalPreview.text);
assert.ok(Object.hasOwn(await call('duyet/tiep-theo', { role: 'duyet_a' }), 'mau'));
const review = { nhan: 'DE_DOA', bang_chung: [{ dong: 1, start: 3, end: 18 }] };
await call('duyet/' + id, { body: { ...review, ghi_chu: 'x'.repeat(201) }, role: 'duyet_a', status: 400 });
const first = await call('duyet/' + id, { body: { ...review, ghi_chu: 'Lời dọa rõ ràng.' }, role: 'duyet_a', status: 201 });
// Một người duyệt: một lượt duyệt là chốt (quyết định 28/11/2026).
assert.equal(first.trang_thai, 'da_duyet');
await call('duyet/' + id, { body: review, role: 'duyet_a', status: 409 });
await call('duyet/' + id, { body: review, role: 'duyet_b', status: 409 });
const final = (await call('duyet/' + id, { role: 'duyet_b' })).mau.quyet_dinh;
assert.equal(final.bang_chung[0].text, finalPreview.text.split('\n')[1].slice(3, 18));
assert.equal(final.nguon, 'phan_hoi');
const auditLog = (await call('nhat-ky/' + id, { role: 'quan_tri' })).nhat_ky;
assert.equal(auditLog.filter((row) => row.hanh_dong === 'luu_luot_duyet').length, 1);
assert.ok(auditLog.some((row) => row.hanh_dong === 'mo_mau'));
assert.ok((await call('nhat-ky', { role: 'quan_tri' })).nhat_ky.length > 0);
const submitNewLevel2 = () => call(rest, { body: { ...level2, ma_lan_gui: randomUUID() }, status: 201 });
const groupTargeted = await submitNewLevel2();
assert.equal((await call('duyet/' + groupTargeted.id, {
  body: { nhan: 'XUC_PHAM', nham_nhom: true }, role: 'duyet_a', status: 201,
})).trang_thai, 'da_duyet');
assert.equal((await call('duyet/' + groupTargeted.id, { role: 'duyet_b' })).mau.quyet_dinh.nhan, 'XUC_PHAM');
// Tài khoản kiểm duyệt duy nhất của bản demo làm được cả việc của người duyệt lẫn quản trị.
if (demo) {
  assert.deepEqual((await call('noi-bo/che-do')).vai_tro.map(({ id: actor }) => actor), ['kiem_duyet']);
  assert.ok(Object.hasOwn(await call('duyet', { role: 'kiem_duyet' }), 'hang_doi'));
  assert.ok((await call('nhat-ky', { role: 'kiem_duyet' })).nhat_ky.length > 0);
}
for (let i = 0; i < 3; i++) await submitNewLevel2();
await call(rest, { body: { ...level2, ma_lan_gui: randomUUID() }, status: 429 });
assert.equal((await call(rest, { body: level2 })).gui_lai, true);
console.log('Mức 2 đạt: đồng ý, bản xem trước, giới hạn, phân quyền và nhật ký.');
console.log('Kiểm duyệt đạt: một lượt là chốt, tab hàng đợi theo ngày, ghi chú, tài khoản kiểm duyệt duy nhất.');

const model = formData.phien_ban_mo_hinh;
const thresholds = { XUC_PHAM: 0.8, QUAY_ROI: 0.2, DE_DOA: 0.2, LUA_DAO: 0.2 };
const proposalInput = {
  phien_ban: 'nguong-demo-1', phien_ban_mo_hinh: model, phien_ban_goc: 'baseline',
  nguong: thresholds, ly_do: 'Số đo giả lập cho luồng phê duyệt.',
  validation: { tap: 'gia-lap', so_mau: 100,
    truoc: { macro_f1: 0.8, fpr_an_toan: 0.1 }, sau: { macro_f1: 0.82, fpr_an_toan: 0.08 } },
};
const proposal = await call('de-xuat', { body: proposalInput, role: 'duyet_a', status: 201 });
const stale = await call('de-xuat', {
  body: { ...proposalInput, phien_ban: 'nguong-demo-2' }, role: 'duyet_a', status: 201,
});
const active = () => fetchActiveThresholds(model, (url, opts) => fetch(new URL(url, base), opts));
assert.equal(await active(), null);
const approval = { ly_do: 'Đã xem số đo giả lập.' };
await call('de-xuat/' + proposal.id + '/phe-duyet', { body: approval, role: 'duyet_a', status: 403 });
await call('de-xuat/' + proposal.id + '/phe-duyet', { body: approval, role: 'quan_tri' });
await call('de-xuat/' + stale.id + '/phe-duyet', { body: approval, role: 'quan_tri', status: 409 });
const config = await active();
assert.equal(config.phien_ban, proposalInput.phien_ban);
assert.equal(applyThreshold('XUC_PHAM', 0.7, config.nguong), 'AN_TOAN');
assert.equal(applyThreshold('XUC_PHAM', 0.9, config.nguong), 'XUC_PHAM');
assert.ok((await call('nhat-ky/' + proposal.id, { role: 'quan_tri' })).nhat_ky
  .some((row) => row.hanh_dong === 'phe_duyet_nguong'));
console.log('Ngưỡng đạt: chỉ quản trị phê duyệt, kiểm tra phiên bản gốc và đọc cấu hình đã kích hoạt.');
