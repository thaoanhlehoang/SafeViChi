// Đo che / ẩn danh hóa PII trên 30 ca tự viết (dữ liệu giả lập), ghi results/v3/che_thong_tin/metrics.json.
// Chạy từ gốc repo: node web-v2/scripts/do_che_thong_tin.mjs
//
// Phân loại mỗi ca:
//   dung            : kết quả khớp tuyệt đối văn bản kỳ vọng
//   sot             : còn ít nhất một chuỗi PII kỳ vọng phải che
//   che_nham        : mất ít nhất một chuỗi không phải PII phải giữ
//   sot_va_che_nham : cả hai
//   sai_nhan        : không sót, không che nhầm nhưng nhãn giữ chỗ / ranh giới khác kỳ vọng
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cheThongTin } from '../src/phan_hoi/che_thong_tin.js';
import { CA_CHE_THONG_TIN } from '../tests/ca_che_thong_tin.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(here, '../../results/v3/che_thong_tin/metrics.json');

const chiTiet = CA_CHE_THONG_TIN.map((ca) => {
  const kq = cheThongTin(ca.text, { cheThem: ca.cheThem ?? [] });
  const conLo = ca.pii.filter((p) => kq.text.includes(p));
  const biMat = ca.giu.filter((g) => !kq.text.includes(g));
  let ket_qua;
  if (kq.text === ca.ky_vong) ket_qua = 'dung';
  else if (conLo.length && biMat.length) ket_qua = 'sot_va_che_nham';
  else if (conLo.length) ket_qua = 'sot';
  else if (biMat.length) ket_qua = 'che_nham';
  else ket_qua = 'sai_nhan';
  return {
    id: ca.id, nhom: ca.nhom, mo_ta: ca.mo_ta, ket_qua,
    // Dữ liệu giả lập nên được phép ghi văn bản ra file kết quả.
    ky_vong: ca.ky_vong, thuc_te: kq.text,
    pii_con_lo: conLo, bi_che_nham: biMat,
    cac_doan: kq.cac_doan,
  };
});

const dem = (k) => chiTiet.filter((c) => c.ket_qua === k).length;
const metrics = {
  ghi_chu: 'Dữ liệu giả lập do nhóm tự viết (web-v2/tests/ca_che_thong_tin.mjs); không phải thông tin thật.',
  tong: chiTiet.length,
  dung: dem('dung'),
  sot: dem('sot') + dem('sot_va_che_nham'),
  che_nham: dem('che_nham') + dem('sot_va_che_nham'),
  sai_nhan: dem('sai_nhan'),
  ca_sai: chiTiet.filter((c) => c.ket_qua !== 'dung').map((c) => c.id),
  chi_tiet: chiTiet,
  lenh: 'node web-v2/scripts/do_che_thong_tin.mjs',
  node_version: process.version,
  ngay: new Date().toISOString().slice(0, 10),
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(metrics, null, 2) + '\n', 'utf8');
console.log(`Che đúng ${metrics.dung}/${metrics.tong}; sót ${metrics.sot}; che nhầm ${metrics.che_nham}; sai nhãn ${metrics.sai_nhan}`);
for (const c of chiTiet.filter((x) => x.ket_qua !== 'dung')) {
  console.log(`  ca ${c.id} (${c.ket_qua}): kỳ vọng "${c.ky_vong}" | thực tế "${c.thuc_te}"`);
}
console.log(`Đã ghi ${outPath}`);
