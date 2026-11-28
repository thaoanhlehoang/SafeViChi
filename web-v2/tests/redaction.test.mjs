import assert from 'node:assert/strict';
import test from 'node:test';
import { redactPersonalInfo } from '../src/feedback/redaction.js';
import { redactPreviewSelection, createPreview, createLevel1Payload, createLevel2Payload } from '../src/feedback/client.js';
import { recordFalsePositive, getWarningPreference, resetWarnings } from '../src/feedback/warning-preferences.js';

const cases = [
  ['Gọi 0912345678 nhé', 'Gọi [SĐT] nhé'],
  ['Gọi 0912 345 678 nhé', 'Gọi [SĐT] nhé'],
  ['Gọi +84 912 345 678 nhé', 'Gọi [SĐT] nhé'],
  ['Gọi 0912.345.678 nhé', 'Gọi [SĐT] nhé'],
  ['Gọi ０９１２３４５６７８ nhé', 'Gọi [SĐT] nhé'],
  ['Gọi 0912\u200b345678 nhé', 'Gọi [SĐT] nhé'],
  ['Email demo@example.test.', 'Email [EMAIL].'],
  ['X gửi a.b+test@example.test xong', 'X gửi [EMAIL] xong'],
  ['Giấy tờ 123456789012 nhé', 'Giấy tờ [GIẤY_TỜ] nhé'],
  ['Giấy tờ 1234-5678-9012 nhé', 'Giấy tờ [GIẤY_TỜ] nhé'],
  ['CMND 123456789 nhé', 'CMND [GIẤY_TỜ] nhé'],
  ['Các số 1234567890123 nhé', 'Các số [STK] nhé'],
  ['STK: 12345678 nhé', 'STK: [STK] nhé'],
  ['số tài khoản 1234 5678 9012 nhé', 'số tài khoản [STK] nhé'],
  ['Thẻ 0000 1111 2222 3333 nhé', 'Thẻ [STK] nhé'],
  ['OTP: 123456 nhé', 'OTP: [MÃ] nhé'],
  ['Mã xác minh 123 456 nhé', 'Mã xác minh [MÃ] nhé'],
  ['Code 4567 nhé', 'Code [MÃ] nhé'],
  ['Mã 123 nhé', 'Mã 123 nhé'],
  ['Đến 123 đường Thử Nghiệm, xong', 'Đến [ĐỊA_CHỈ], xong'],
  ['Đến 12A ngõ Demo nhé!', 'Đến [ĐỊA_CHỈ] nhé!'],
  ['Chị Lan ơi', 'Chị [TÊN_1] ơi'],
  ['Anh Nguyễn Văn An nhé', 'Anh [TÊN_1] nhé'],
  ['Nhờ @tai_khoan nhé', 'Nhờ @[TÊN_1] nhé'],
  ['Bạn OTP nói gì?', 'Bạn OTP nói gì?'],
  ['Mở https://example.test/p?x=abc nhé', 'Mở [LINK] nhé'],
  ['Xem www.example.test/demo nhé', 'Xem [LINK] nhé'],
  ['Mở vidu.net?test=abc nhé', 'Mở [LINK] nhé'],
  ['Phí 200.000 đồng', 'Phí 200.000 đồng'],
  ['Gọi 0912345678\nEmail demo@example.test', 'Gọi [SĐT]\nEmail [EMAIL]'],
  ['Link https://demo:abc@example.test/duong-dan?ma=482913.', 'Link [LINK].'],
  ['Mở [LINK:example.test] nhé', 'Mở [LINK] nhé'],
  ['Mã OTP [TÊN_1234] và [ĐÃ_CHE]', 'Mã OTP [TÊN_1234] và [ĐÃ_CHE]'],
];

for (const [i, [text, expected]] of cases.entries()) {
  test('Che thông tin, ca ' + (i + 1), () => {
    assert.equal(redactPersonalInfo(text), expected);
    assert.equal(redactPersonalInfo(expected), expected);
  });
}

test('Mức 2 chỉ gửi khi đồng ý và gửi đúng bản đã che', async () => {
  const map = new Map();
  const storage = { getItem: (key) => map.get(key), setItem: (key, value) => map.set(key, value) };
  const data = {
    ma_bao_cao: 'tin_nhan_an_toan', nhan_mo_hinh: 'XUC_PHAM', nhan_de_xuat: 'AN_TOAN',
    ma_ly_do: 'khac', ly_do_khac: 'Ngữ cảnh khác. Gọi 0912345678',
    co_to_sang: true, khoang_diem: 8, phien_ban_mo_hinh: 'mo-hinh-test', phien_ban_nguong: 'baseline',
    text: 'Tin gốc', bang_chung: ['Tin gốc'], uid: 'demo',
  };
  const level1 = await createLevel1Payload(data, { storage });
  for (const key of ['text', 'bang_chung', 'uid', 'doan_da_che']) assert.ok(!Object.hasOwn(level1, key));
  assert.equal(level1.ly_do_khac, 'Ngữ cảnh khác. Gọi [SĐT]');

  const preview = createPreview(['Lan Xuân gọi 0912345678'], { contactNames: ['Lan Xuân'] });
  assert.equal(preview.text, '[TÊN_1] gọi [SĐT]');
  const start = preview.text.indexOf('gọi');
  const manuallyRedacted = redactPreviewSelection(preview, 0, start, start + 'gọi'.length);
  assert.equal(manuallyRedacted.text, '[TÊN_1] [ĐÃ_CHE] [SĐT]');
  await assert.rejects(createLevel2Payload(data, manuallyRedacted, { storage }), /dong_y/);
  const level2 = await createLevel2Payload(data, manuallyRedacted, { storage, consent: true });
  assert.equal(level2.doan_da_che, manuallyRedacted.text);
  assert.equal(level2.dong_y.ban_xem_truoc_sha256.length, 64);
});

test('Bản xem trước tự che theo ngữ cảnh qua nhiều lượt và giữ số tiền', () => {
  const preview = createPreview([
    'Chị Lan ơi, mã OTP của bạn là 482913',
    'Lan gửi 500k tại vidu.net/path',
  ]);
  assert.equal(preview.text, 'Chị [TÊN_1] ơi, mã OTP của bạn là [MÃ]\n[TÊN_1] gửi 500k tại [LINK]');
  const extra = redactPreviewSelection(preview, 1, 4, '[TÊN_1] gửi'.length);
  assert.equal(extra.text, 'Chị [TÊN_1] ơi, mã OTP của bạn là [MÃ]\n[ĐÃ_CHE] 500k tại [LINK]');
  assert.equal(redactPersonalInfo(extra.text), extra.text);
});

test('Làm dịu chỉ áp dụng cục bộ trong 7 ngày và giữ cảnh báo mạnh', () => {
  const map = new Map();
  const storage = { getItem: (key) => map.get(key), setItem: (key, value) => map.set(key, value),
    removeItem: (key) => map.delete(key) };
  const now = Date.now();
  for (const label of ['XUC_PHAM', 'DE_DOA', 'LUA_DAO']) {
    for (let i = 2; i >= 0; i--) recordFalsePositive(label, storage, now - i);
  }
  assert.deepEqual(getWarningPreference('XUC_PHAM', storage, now), { blur: false });
  assert.equal(getWarningPreference('DE_DOA', storage, now), null);
  assert.equal(getWarningPreference('LUA_DAO', storage, now), null);
  assert.equal(getWarningPreference('XUC_PHAM', storage, now + 7 * 86400_000), null);
  resetWarnings(storage);
  assert.equal(getWarningPreference('XUC_PHAM', storage, now), null);
});
