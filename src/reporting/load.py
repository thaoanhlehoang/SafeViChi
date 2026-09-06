"""Đọc và hợp nhất kết quả Bước 3 từ results/step3/.

Có một chỗ dễ sai phải nói rõ: hai nguồn số liệu khác nhau.

  - `test/matrix.json`  — mọi hệ thống, đo ở NGƯỠNG MẶC ĐỊNH 0.5.
  - `threshold/threshold.json` — chỉ 4 hệ thống T5, đo ở NGƯỠNG ĐÃ DÒ trên
    validation rồi đóng băng. Blacklist không có mặt ở đây vì nó không sinh ra
    điểm xác suất nào để mà dò ngưỡng — nó chỉ khớp từ khóa.

Bảng báo cáo cuối dùng số ĐÃ DÒ NGƯỠNG cho B2/B3 (công bằng: mỗi model được
chọn điểm vận hành tốt nhất của riêng nó) và số ở 0.5 cho B1 (không áp dụng
được). `load_report_table()` ghép đúng như vậy và đánh dấu nguồn từng dòng để
không ai vô tình so hai loại số với nhau.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_DIR = Path("results/step3")


@dataclass(frozen=True)
class Cell:
    """Một ô của ma trận: một hệ thống × một điều kiện."""

    system: str
    condition: str
    hate_f1: float
    hate_precision: float
    hate_recall: float
    macro_f1: float
    f1_ci_low: float
    f1_ci_high: float
    threshold: float | None      # None = không áp dụng (blacklist)
    tuned: bool                  # True = số ở ngưỡng đã dò, False = ở 0.5
    n: int
    n_hate: int
    accuracy: float | None
    pr_auc: float | None


def load_matrix(results_dir: Path = DEFAULT_RESULTS_DIR, split: str = "test") -> dict:
    return json.load(open(results_dir / split / "matrix.json", encoding="utf-8"))


def load_threshold(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict:
    return json.load(open(results_dir / "threshold" / "threshold.json", encoding="utf-8"))


def load_report_table(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[tuple[str, str], Cell]:
    """Bảng số liệu chính thức: B2/B3 ở ngưỡng đã dò, B1 ở ngưỡng mặc định."""
    matrix = load_matrix(results_dir)
    thresholds = load_threshold(results_dir)
    tuned_results = thresholds["results"]
    chosen = thresholds["chosen_thresholds"]

    table: dict[tuple[str, str], Cell] = {}
    for system, per_condition in matrix["results"].items():
        for condition, raw in per_condition.items():
            tuned_entry = tuned_results.get(system, {}).get(condition, {}).get("tuned")
            if tuned_entry is not None:
                table[(system, condition)] = Cell(
                    system=system, condition=condition,
                    hate_f1=tuned_entry["hate_f1"],
                    hate_precision=tuned_entry["hate_precision"],
                    hate_recall=tuned_entry["hate_recall"],
                    macro_f1=tuned_entry["macro_f1"],
                    f1_ci_low=tuned_entry["f1_ci_low"],
                    f1_ci_high=tuned_entry["f1_ci_high"],
                    threshold=chosen.get(system),
                    tuned=True,
                    n=raw["n"], n_hate=raw["n_hate"],
                    accuracy=raw.get("accuracy"), pr_auc=raw.get("pr_auc"),
                )
            else:
                table[(system, condition)] = Cell(
                    system=system, condition=condition,
                    hate_f1=raw["hate_f1"],
                    hate_precision=raw["hate_precision"],
                    hate_recall=raw["hate_recall"],
                    macro_f1=raw["macro_f1"],
                    f1_ci_low=raw["f1_ci_low"],
                    f1_ci_high=raw["f1_ci_high"],
                    threshold=None, tuned=False,
                    n=raw["n"], n_hate=raw["n_hate"],
                    accuracy=raw.get("accuracy"), pr_auc=raw.get("pr_auc"),
                )
    return table


def load_comparisons(results_dir: Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Danh sách kiểm định McNemar ghép cặp."""
    return load_matrix(results_dir)["comparisons"]


def load_scores(results_dir: Path = DEFAULT_RESULTS_DIR,
                split: str = "test") -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """P(hate) từng dòng -> {(hệ thống, điều kiện): {"y":…, "p":…}}.

    Chỉ 4 hệ thống T5 có mặt: blacklist không sinh ra xác suất.
    """
    buckets: dict[tuple[str, str], dict[str, list]] = {}
    path = results_dir / split / "scores.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["system"], row["condition"])
            bucket = buckets.setdefault(key, {"y": [], "p": []})
            bucket["y"].append(row["label"])
            bucket["p"].append(row["p_hate"])
    return {k: {"y": np.asarray(v["y"]), "p": np.asarray(v["p"])}
            for k, v in buckets.items()}
