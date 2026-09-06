"""Thống kê cho ma trận baseline: khoảng tin cậy bootstrap + kiểm định McNemar.

Tập test giữ phân bố tự nhiên (~18% HATE / 82% CLEAN) nên chỉ có 432 dòng
HATE. Hai công cụ ở đây trả lời trực tiếp câu hỏi "số này có đáng tin không":

  - bootstrap_ci: lấy mẫu lại có hoàn lại trên chính các dòng test, cho error
    bar quanh F1. Với n_HATE=432, sai số 95% của recall vào khoảng ±3-5 điểm,
    tức đủ nhỏ để phân biệt các baseline (khoảng cách kỳ vọng >10 điểm).
  - mcnemar: so CẶP hai hệ thống trên cùng từng dòng — mạnh hơn nhiều so với
    đặt cạnh nhau hai con số F1 rời rạc, vì loại bỏ được phương sai do độ khó
    của từng câu.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

from src.utils.seed import SEED_DEFAULT


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED_DEFAULT,
    pos_label: int = 1,
) -> dict[str, float]:
    """Khoảng tin cậy percentile cho F1 lớp HATE.

    Seed cố định (mặc định 42) nên chạy lại ra ĐÚNG cùng khoảng tin cậy. Đây là
    chủ ý: khoảng tin cậy in trong báo cáo phải kiểm chứng lại được, không được
    nhảy số mỗi lần chạy.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], pos_label=pos_label,
                             average="binary", zero_division=0)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"f1_ci_low": float(lo), "f1_ci_high": float(hi),
            "f1_ci_halfwidth": float((hi - lo) / 2)}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar ghép cặp giữa 2 hệ thống chạy trên CÙNG các dòng.

    b = số dòng A đúng / B sai, c = số dòng A sai / B đúng. Chỉ các dòng hai
    bên khác nhau mới mang thông tin. Dùng binomial test chính xác (đúng cả
    khi b + c nhỏ, không cần xấp xỉ chi-square).
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0, "significant": False}
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"b": b, "c": c, "p_value": p, "significant": p < 0.05}
