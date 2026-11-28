import assert from 'node:assert/strict';
import test from 'node:test';
import { LABEL_INDICATORS, REPORT_OPTIONS } from '../src/feedback/contract.js';
import {
  DAU_HIEU, LOAI_BAO_CAO, LY_DO_AN_TOAN, LY_DO_BO_SOT, LY_DO_KHAC, LY_DO_SAI_LOAI, LY_DO_TO_SANG,
} from '../src/phan_hoi/ma.js';

test('Mã lựa chọn trên biểu mẫu mức 1 khớp với API', () => {
  assert.deepEqual(LOAI_BAO_CAO.map(({ ma }) => ma), Object.keys(REPORT_OPTIONS));
  assert.deepEqual(
    Object.fromEntries(Object.entries(DAU_HIEU).map(([nhan, items]) => [nhan, items.map(({ ma }) => ma)])),
    LABEL_INDICATORS,
  );
  const nhom = {
    tin_nhan_an_toan: LY_DO_AN_TOAN,
    sai_loai: LY_DO_SAI_LOAI,
    bo_sot: LY_DO_BO_SOT,
    loi_to_sang: LY_DO_TO_SANG,
    khac_chua_chac: LY_DO_KHAC,
  };
  for (const [ma, options] of Object.entries(nhom)) {
    assert.deepEqual(
      Object.fromEntries(options.map(({ ma, phu }) => [ma, phu?.options.map((option) => option.ma) ?? []])),
      REPORT_OPTIONS[ma],
    );
  }
});
