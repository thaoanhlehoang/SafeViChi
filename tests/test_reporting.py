"""Kiểm tra module vẽ biểu đồ (src/reporting/).

Trọng tâm không phải "hình có đẹp không" — cái đó phải nhìn bằng mắt. Ở đây
kiểm ba thứ máy kiểm được:

  1. Ràng buộc CHỈ XUẤT PDF của đề tài không bị lách.
  2. Bảng số liệu ghép đúng nguồn: B2/B3 lấy số ở ngưỡng đã dò, B1 lấy số ở
     ngưỡng mặc định — ghép nhầm là báo cáo sai.
  3. Nhãn rút gọn vẫn phân biệt được bản có/không bật chuẩn hóa.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporting import style
from src.reporting.load import DEFAULT_RESULTS_DIR, load_report_table

pytest.importorskip("matplotlib")

RESULTS_AVAILABLE = (DEFAULT_RESULTS_DIR / "test" / "matrix.json").exists()
needs_results = pytest.mark.skipif(
    not RESULTS_AVAILABLE, reason="chưa có results/step3/ trên máy này")


def test_save_pdf_tu_choi_dinh_dang_khac(tmp_path: Path):
    """Đề tài bắt buộc PDF; mọi đuôi khác phải bị chặn ngay, không im lặng ghi ra."""
    from matplotlib import pyplot as plt

    style.apply_rcparams()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(AssertionError):
        style.save_pdf(fig, tmp_path / "hinh.png")
    plt.close(fig)


def test_save_pdf_ghi_duoc_file_pdf(tmp_path: Path):
    from matplotlib import pyplot as plt

    style.apply_rcparams()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "hinh.pdf"
    style.save_pdf(fig, out)
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-", "file ghi ra không phải PDF thật"


def test_nhan_rut_gon_phan_biet_duoc_ban_co_chuan_hoa():
    """Rút gọn thành "B2"/"B3" sẽ làm bảng McNemar ghi sai cặp đang so."""
    shorts = [style.SYSTEM_SHORT[s] for s in style.SYSTEM_ORDER]
    assert len(set(shorts)) == len(shorts), f"nhãn rút gọn bị trùng: {shorts}"
    for system in style.SYSTEM_ORDER:
        if style.uses_normalizer(system):
            assert style.SYSTEM_SHORT[system].endswith("+ch")


def test_moi_he_thong_co_mau_va_hoa_tiet():
    for system in style.SYSTEM_ORDER:
        spec = style.bar_style(system)
        assert spec["color"].startswith("#")
        assert bool(spec["hatch"]) == style.uses_normalizer(system)


@needs_results
def test_bang_bao_cao_ghep_dung_nguon_so_lieu():
    table = load_report_table(DEFAULT_RESULTS_DIR)
    assert len(table) == len(style.SYSTEM_ORDER) * len(style.CONDITION_ORDER)

    for (system, _condition), cell in table.items():
        if system.startswith("b1_"):
            # Blacklist không sinh xác suất -> không dò ngưỡng được.
            assert cell.threshold is None
            assert cell.tuned is False
        else:
            assert cell.threshold is not None
            assert cell.tuned is True
            assert 0.0 < cell.threshold < 1.0


@needs_results
def test_cung_mot_he_thong_dung_chung_mot_nguong_o_moi_dieu_kien():
    """Ngưỡng dò trên validation rồi ĐÓNG BĂNG — không được đổi theo điều kiện.

    Đổi ngưỡng theo từng kiểu tấn công là gian lận: lúc chạy thật không ai biết
    trước câu đang tới thuộc kiểu nào.
    """
    table = load_report_table(DEFAULT_RESULTS_DIR)
    for system in style.SYSTEM_ORDER:
        thresholds = {table[(system, c)].threshold for c in style.CONDITION_ORDER}
        assert len(thresholds) == 1, f"{system} dùng nhiều ngưỡng khác nhau: {thresholds}"


@needs_results
def test_khoang_tin_cay_bao_quanh_gia_tri_do_duoc():
    table = load_report_table(DEFAULT_RESULTS_DIR)
    for key, cell in table.items():
        assert cell.f1_ci_low <= cell.hate_f1 <= cell.f1_ci_high, key
