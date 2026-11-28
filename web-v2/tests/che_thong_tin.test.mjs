// Kiểm thử che / ẩn danh hóa PII trên thiết bị (SafeViChi v3).
// Chạy: node --test web-v2/tests/   (dữ liệu giả lập, xem tests/ca_che_thong_tin.mjs)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cheThongTin } from '../src/phan_hoi/che_thong_tin.js';
import { CA_CHE_THONG_TIN } from './ca_che_thong_tin.mjs';

test('bộ ca có đúng 30 ca, id không trùng', () => {
  assert.equal(CA_CHE_THONG_TIN.length, 30);
  assert.equal(new Set(CA_CHE_THONG_TIN.map((c) => c.id)).size, 30);
});

for (const ca of CA_CHE_THONG_TIN) {
  test(`ca ${ca.id} [${ca.nhom}] ${ca.mo_ta}`, () => {
    const kq = cheThongTin(ca.text, { cheThem: ca.cheThem ?? [] });
    for (const p of ca.pii) assert.ok(!kq.text.includes(p), `sót PII: "${p}" còn trong "${kq.text}"`);
    for (const g of ca.giu) assert.ok(kq.text.includes(g), `che nhầm: mất "${g}" trong "${kq.text}"`);
    assert.equal(kq.text, ca.ky_vong);
    // Số dòng và "\n" phải giữ nguyên.
    assert.equal(kq.text.split('\n').length, ca.text.split('\n').length);
    // cac_doan: vị trí theo văn bản gốc, không chồng lấn, chỉ có loai/start/end (không chứa nội dung gốc).
    let truoc = 0;
    for (const d of kq.cac_doan) {
      assert.deepEqual(Object.keys(d).sort(), ['end', 'loai', 'start']);
      assert.ok(d.start >= truoc && d.end > d.start && d.end <= ca.text.length);
      assert.ok(!ca.text.slice(d.start, d.end).includes('\n'));
      truoc = d.end;
    }
  });
}

test('che thêm chỉ thêm, không bỏ được đoạn đã che tự động', () => {
  const text = 'stk 0123456789 nhé';
  const goc = cheThongTin(text);
  const them = cheThongTin(text, { cheThem: [{ start: 15, end: 18 }] });
  for (const d of goc.cac_doan) {
    assert.ok(them.cac_doan.some((x) => x.start <= d.start && x.end >= d.end));
  }
});
