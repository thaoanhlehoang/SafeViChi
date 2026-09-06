"""Kiểm tra module seed tập trung (src/utils/seed.py)."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np

from src.utils.seed import (
    SEED_DEFAULT,
    SEED_STEP1_DATASET,
    derive_np_rng,
    derive_rng,
    derive_seed,
    set_global_seed,
)


def test_hang_seed_khong_doi():
    """Đổi hai hằng này là mất khả năng tái tạo dữ liệu đã sinh."""
    assert SEED_STEP1_DATASET == 20260902
    assert SEED_DEFAULT == 42


def test_derive_seed_on_dinh():
    assert derive_seed(42, "train", "row_007") == derive_seed(42, "train", "row_007")


def test_derive_seed_khac_nhau_theo_dinh_danh():
    assert derive_seed(42, "train") != derive_seed(42, "validation")
    assert derive_seed(42, "train") != derive_seed(43, "train")


def test_derive_seed_khong_phu_thuoc_pythonhashseed():
    """`hash()` của str bị ngẫu nhiên hóa theo tiến trình; BLAKE2b thì không.

    Chạy hai tiến trình con với PYTHONHASHSEED khác nhau và đòi cùng kết quả.
    """
    code = ("import sys; sys.path.insert(0, '.');"
            "from src.utils.seed import derive_seed; print(derive_seed(42, 'abc'))")
    outs = []
    for hashseed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": hashseed, "PATH": ""})
        assert out.returncode == 0, out.stderr
        outs.append(out.stdout.strip())
    assert len(set(outs)) == 1, f"derive_seed đổi theo PYTHONHASHSEED: {outs}"


def test_derive_rng_tai_lap_duoc():
    a = [derive_rng(42, "x").random() for _ in range(3)]
    assert len(set(a)) == 1, "cùng định danh phải cho cùng luồng số"
    assert derive_rng(42, "x").random() != derive_rng(42, "y").random()


def test_derive_np_rng_tai_lap_duoc():
    a = derive_np_rng(42, "x").integers(0, 10_000, size=5)
    b = derive_np_rng(42, "x").integers(0, 10_000, size=5)
    assert np.array_equal(a, b)


def test_set_global_seed_cho_ket_qua_lap_lai():
    set_global_seed(123, verbose=False)
    first = (random.random(), float(np.random.rand()))
    set_global_seed(123, verbose=False)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_set_global_seed_bao_cao_dung():
    report = set_global_seed(7, verbose=False)
    assert report.seed == 7
    assert report.deterministic is False
    assert "random" in report.seeded and "numpy" in report.seeded
    assert "PYTHONHASHSEED" in report.seeded
    assert report.as_dict()["seed"] == 7
