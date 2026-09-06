"""Vẽ toàn bộ biểu đồ + bảng so sánh 3 baseline của Bước 3, xuất ra PDF.

    python -m src.reporting.make_figures

Chạy trên CPU trong vài giây — chỉ đọc results/step3/, KHÔNG cần GPU và không
nạp model. Đường cong ROC/PR dựng lại từ `scores.jsonl` (P(hate) từng dòng đã
được `run_matrix.py --dump_scores` ghi sẵn).

Đầu ra (chỉ .pdf, vector, chữ nhúng dạng font — nhúng thẳng vào Word/LaTeX được):

    results/step3/figures/
      hinh1_ma_tran_f1.pdf          F1 lớp HATE, 6 hệ thống × 3 điều kiện + KTC 95%
      hinh2_do_tut_hieu_nang.pdf    hiệu năng tụt dần khi mức tấn công tăng
      hinh3_duong_cong_roc.pdf      ROC, 3 điều kiện
      hinh4_duong_cong_pr.pdf       Precision-Recall, 3 điều kiện
      hinh5_dong_gop_tung_buoc.pdf  tách bạch Bước 2 và Bước 3 đóng góp bao nhiêu
      hinh6_precision_recall.pdf    đánh đổi precision/recall từng hệ thống
      bang1_chi_so_day_du.pdf       bảng chỉ số đầy đủ
      bang2_mcnemar.pdf             bảng kiểm định ý nghĩa thống kê
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.reporting import style
from src.reporting.load import (
    DEFAULT_RESULTS_DIR,
    load_comparisons,
    load_report_table,
    load_scores,
)
from src.reporting.style import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    SYSTEM_SHORT,
)

# Câu hỏi trong matrix.json viết dài, tràn cột. Viết lại gọn, giữ nguyên ý.
QUESTION_SHORT = {
    "normalizer đóng góp bao nhiêu (Step 2)": "Bước 2 (chuẩn hóa) góp bao nhiêu?",
    "fine-tune đối kháng đóng góp bao nhiêu (Step 3)": "Bước 3 (fine-tune) góp bao nhiêu?",
    "pipeline đầy đủ hơn T5 gốc bao nhiêu": "Pipeline hơn T5 gốc bao nhiêu?",
    "model đã fine-tune còn cần normalizer không": "Fine-tune rồi còn cần Bước 2?",
    "pipeline hơn cách ngây thơ nhất bao nhiêu": "Pipeline hơn blacklist bao nhiêu?",
}
from src.utils.seed import SEED_DEFAULT, set_global_seed

# 4 hệ thống có điểm xác suất (blacklist chỉ khớp từ khóa nên không có)
SCORED_SYSTEMS = ["b2_t5base", "b2_t5base_norm", "b3_t5ft", "b3_t5ft_norm"]


# --------------------------------------------------------------------------
# Hình 1 — ma trận F1
# --------------------------------------------------------------------------

def figure_f1_matrix(table, out_path: Path) -> None:
    """Cột nhóm: 6 hệ thống × 3 điều kiện, kèm thanh sai số KTC 95%."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    style.style_axes(ax)

    n_sys = len(SYSTEM_ORDER)
    group_width = 0.82
    bar_width = group_width / n_sys

    for i, system in enumerate(SYSTEM_ORDER):
        xs, heights, lo_err, hi_err = [], [], [], []
        for j, condition in enumerate(CONDITION_ORDER):
            cell = table[(system, condition)]
            xs.append(j - group_width / 2 + bar_width * (i + 0.5))
            heights.append(cell.hate_f1)
            lo_err.append(cell.hate_f1 - cell.f1_ci_low)
            hi_err.append(cell.f1_ci_high - cell.hate_f1)

        ax.bar(xs, heights, width=bar_width * 0.9,
               label=SYSTEM_LABELS[system], zorder=3, **style.bar_style(system))
        ax.errorbar(xs, heights, yerr=[lo_err, hi_err], fmt="none",
                    ecolor=style.INK_SECONDARY, elinewidth=1.1,
                    capsize=2.5, capthick=1.1, zorder=4)
        for x, h, hi in zip(xs, heights, hi_err):
            ax.text(x, h + hi + 0.018, f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.4, color=style.INK_SECONDARY, rotation=90, zorder=5)

    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_xlabel("Điều kiện đầu vào", labelpad=10)
    ax.set_ylabel("F1 (HATE)", labelpad=10)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(ncol=2, loc="upper right", columnspacing=1.4, handlelength=1.6)
    style.add_caption(ax, "Hình 1 — So sánh 3 baseline trên ba điều kiện đầu vào", -0.22)

    style.save_pdf(fig, out_path)


# --------------------------------------------------------------------------
# Hình 2 — độ tụt hiệu năng
# --------------------------------------------------------------------------

def figure_degradation(table, out_path: Path) -> None:
    """Đường gấp khúc: F1 tụt thế nào khi mức độ né lọc tăng."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    style.style_axes(ax)

    xs = np.arange(len(CONDITION_ORDER))
    for system in SYSTEM_ORDER:
        ys = [table[(system, c)].hate_f1 for c in CONDITION_ORDER]
        ax.plot(xs, ys, label=SYSTEM_LABELS[system], zorder=3,
                **style.line_style(system))

    # Không ghi nhãn trực tiếp lên từng đường: ở C2 sáu đường chụm lại, nhãn
    # lơ lửng sẽ bị gán nhầm đường. Thay bằng việc đo thẳng KHOẢNG CÁCH giữa
    # "không phòng vệ" và "pipeline đầy đủ" — đó mới là điều hình này cần nói.
    for x_idx, condition in ((1, "C1_perturbed"), (2, "C2_blindspot")):
        low = table[("b2_t5base", condition)].hate_f1
        high = table[("b3_t5ft_norm", condition)].hate_f1
        x_gap = x_idx - 0.16
        ax.annotate("", xy=(x_gap, high), xytext=(x_gap, low),
                    arrowprops=dict(arrowstyle="<->", color=style.INK_SECONDARY,
                                    linewidth=1.2, shrinkA=0, shrinkB=0), zorder=6)
        ax.text(x_gap - 0.04, (low + high) / 2, f"+{high - low:.3f}",
                ha="right", va="center", fontsize=10, fontweight="bold",
                color=style.INK_PRIMARY, zorder=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_xlim(-0.30, len(CONDITION_ORDER) - 0.78)
    ax.set_xlabel("Điều kiện đầu vào", labelpad=10)
    ax.set_ylabel("F1 (HATE)", labelpad=10)
    ax.set_ylim(0.40, 1.02)
    ax.legend(ncol=2, loc="upper right", columnspacing=1.4, handlelength=2.2)
    style.add_caption(ax, "Hình 2 — Hiệu năng tụt khi văn bản bị né lọc", -0.20)

    style.save_pdf(fig, out_path)


# --------------------------------------------------------------------------
# Hình 3 & 4 — ROC và Precision-Recall
# --------------------------------------------------------------------------

def _curve_panels(scores, out_path: Path, kind: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2), sharey=True)

    area_label = "AUC" if kind == "roc" else "AP"

    for ax, condition in zip(axes, CONDITION_ORDER):
        style.style_axes(ax)
        ax.grid(axis="x")
        areas: list[tuple[str, float]] = []
        for system in SCORED_SYSTEMS:
            bucket = scores.get((system, condition))
            if bucket is None:
                continue
            y, p = bucket["y"], bucket["p"]
            if kind == "roc":
                x_vals, y_vals, _ = roc_curve(y, p)
            else:
                y_vals, x_vals, _ = precision_recall_curve(y, p)
            areas.append((system, float(auc(x_vals, y_vals))))

            line_kwargs = style.line_style(system)
            for key in ("marker", "markersize", "markeredgecolor", "markeredgewidth"):
                line_kwargs.pop(key)
            # Đường "có chuẩn hóa" vẽ mảnh hơn để khi nó TRÙNG KHÍT đường thô
            # (đúng như ở C2) người đọc vẫn thấy được cả hai.
            if style.uses_normalizer(system):
                line_kwargs["linewidth"] = 1.4
            ax.plot(x_vals, y_vals, zorder=3, label=SYSTEM_LABELS[system], **line_kwargs)

        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=style.INK_MUTED, linewidth=1.0, zorder=2)
            ax.text(0.60, 0.53, "đoán ngẫu nhiên", fontsize=8,
                    color=style.INK_MUTED, rotation=39, ha="center", va="center")
        else:
            base = float(np.mean(scores[(SCORED_SYSTEMS[0], condition)]["y"]))
            ax.axhline(base, color=style.INK_MUTED, linewidth=1.0, zorder=2)
            # Đặt giữa: góc trái dưới đã dành cho chú giải, góc phải thì đường
            # cong lao xuống chạm mức nền.
            ax.text(0.55, base + 0.025, f"tỉ lệ HATE thực tế = {base:.3f}",
                    fontsize=8, color=style.INK_MUTED, ha="center", va="bottom")

        # Diện tích dưới đường phải ghi RIÊNG TỪNG Ô: mỗi điều kiện có giá trị
        # riêng, dán một bộ số vào chú giải dùng chung là ghi sai cho 2 ô kia.
        # Đặt ở dòng phụ của tiêu đề ô, không đặt trong lòng đồ thị — mọi góc
        # trống của đồ thị đều bị đường cong quét qua ở ít nhất một điều kiện.
        short = {"b2_t5base": "B2", "b2_t5base_norm": "B2+ch",
                 "b3_t5ft": "B3", "b3_t5ft_norm": "B3+ch"}
        # Bỏ số 0 đứng đầu (".931" thay vì "0.931") để dòng này vừa bề ngang một
        # ô — viết đủ bốn giá trị ở cỡ chữ đọc được thì không còn chỗ.
        pairs = [f"{short[s]} {a:.3f}".replace("0.", ".") for s, a in areas]
        ax.set_title(CONDITION_LABELS[condition].replace("\n", " "),
                     fontsize=11, pad=24, loc="left")
        ax.text(0.0, 1.015, f"{area_label}  " + " · ".join(pairs),
                transform=ax.transAxes, fontsize=8.4, color=style.INK_SECONDARY,
                ha="left", va="bottom")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("FPR" if kind == "roc" else "Recall", labelpad=8)

    axes[0].set_ylabel("TPR" if kind == "roc" else "Precision", labelpad=10)

    # Chú giải đặt trong khung, ở góc mà đường cong không quét qua: ROC trống
    # góc dưới-phải (dưới đường chéo), PR trống góc dưới-trái (dưới mức nền).
    # Có nền trắng đục: đường mức nền / đường chéo chạy xuyên qua chỗ đặt chú
    # giải, không che thì chữ bị gạch ngang.
    axes[0].legend(loc="lower right" if kind == "roc" else "lower left",
                   handlelength=2.2, labelspacing=0.35, borderpad=0.5,
                   frameon=True, facecolor=style.SURFACE, edgecolor="none",
                   framealpha=1.0).set_zorder(10)
    fig.subplots_adjust(wspace=0.12)

    caption = ("Hình 3 — Đường cong ROC theo từng điều kiện đầu vào" if kind == "roc"
               else "Hình 4 — Đường cong Precision-Recall theo từng điều kiện đầu vào")
    style.add_caption(axes[1], caption, -0.18)

    style.save_pdf(fig, out_path)


def figure_roc(scores, out_path: Path) -> None:
    _curve_panels(scores, out_path, "roc")


def figure_pr(scores, out_path: Path) -> None:
    _curve_panels(scores, out_path, "pr")


# --------------------------------------------------------------------------
# Hình 5 — đóng góp từng bước
# --------------------------------------------------------------------------

def figure_contribution(table, out_path: Path) -> None:
    """Tách bạch: Bước 2 (chuẩn hóa) và Bước 3 (fine-tune) mỗi bên góp bao nhiêu."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    panels = [
        ("C1_perturbed", "Trên C1 — câu bị né lọc"),
        ("C2_blindspot", "Trên C2 — điểm mù của bộ chuẩn hóa"),
    ]

    for ax, (condition, panel_title) in zip(axes, panels):
        style.style_axes(ax)
        chain = ["b2_t5base", "b2_t5base_norm", "b3_t5ft_norm"]
        values = [table[(s, condition)].hate_f1 for s in chain]
        labels = ["ViHateT5 gốc\n(không phòng vệ)",
                  "+ Bước 2\nchuẩn hóa",
                  "+ Bước 3\nfine-tune đối kháng"]
        colors = [style.FAMILY_COLORS["b2"], style.FAMILY_COLORS["b2"],
                  style.FAMILY_COLORS["b3"]]
        hatches = ["", style.HATCH_WITH_NORMALIZER, style.HATCH_WITH_NORMALIZER]

        xs = np.arange(len(chain))
        ax.bar(xs, values, width=0.6, color=colors, hatch=hatches,
               edgecolor=style.SURFACE, linewidth=0, zorder=3)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.012, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=style.INK_PRIMARY, zorder=5)

        # Mũi tên ghi rõ mức tăng của từng bước
        for i in range(len(chain) - 1):
            delta = values[i + 1] - values[i]
            y = max(values[i], values[i + 1]) + 0.075
            ax.annotate("", xy=(i + 1, y), xytext=(i, y),
                        arrowprops=dict(arrowstyle="->", color=style.INK_SECONDARY,
                                        linewidth=1.3))
            ax.text(i + 0.5, y + 0.012, f"{delta:+.4f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold",
                    color=style.FAMILY_COLORS["b3"] if delta > 0.02 else style.INK_MUTED)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_ylabel("F1 (HATE)", labelpad=10)
        ax.set_title(panel_title, fontsize=11.5, pad=12, loc="left")

    fig.subplots_adjust(wspace=0.22)
    style.add_caption(axes[0], "Hình 5 — Bước 2 và Bước 3 mỗi bên đóng góp bao nhiêu",
                      -0.22, x=1.11)

    style.save_pdf(fig, out_path)


# --------------------------------------------------------------------------
# Hình 6 — precision vs recall
# --------------------------------------------------------------------------

def figure_precision_recall_bars(table, out_path: Path) -> None:
    """Vì sao không dùng accuracy: xem precision của blacklist."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.4), sharey=True)

    for ax, condition in zip(axes, CONDITION_ORDER):
        style.style_axes(ax)
        xs = np.arange(len(SYSTEM_ORDER))
        precisions = [table[(s, condition)].hate_precision for s in SYSTEM_ORDER]
        recalls = [table[(s, condition)].hate_recall for s in SYSTEM_ORDER]

        for i, system in enumerate(SYSTEM_ORDER):
            base = style.bar_style(system)
            ax.bar(i - 0.19, precisions[i], width=0.34, zorder=3, **base)
            faded = dict(base)
            faded["alpha"] = 0.42
            ax.bar(i + 0.19, recalls[i], width=0.34, zorder=3, **faded)

        ax.set_xticks(xs)
        ax.set_xticklabels([SYSTEM_LABELS[s].split(" · ")[0] + "\n"
                            + ("+chuẩn hóa" if style.uses_normalizer(s) else "thô")
                            for s in SYSTEM_ORDER], fontsize=8.5)
        ax.set_xlabel(CONDITION_LABELS[condition], labelpad=10)
        ax.set_ylim(0, 1.06)
        ax.set_yticks(np.arange(0, 1.01, 0.2))

    axes[0].set_ylabel("Giá trị chỉ số", labelpad=10)
    legend_handles = [
        Patch(facecolor=style.INK_SECONDARY, label="Precision"),
        Patch(facecolor=style.INK_SECONDARY, alpha=0.42, label="Recall"),
    ]
    # Đặt ở ô C2 — hai ô kia có cột cao chạm tới góc trên bên phải.
    axes[2].legend(handles=legend_handles, loc="upper right", borderpad=0.4)
    fig.subplots_adjust(wspace=0.1)
    style.add_caption(axes[1], "Hình 6 — Đánh đổi giữa precision và recall", -0.26)

    style.save_pdf(fig, out_path)


# --------------------------------------------------------------------------
# Bảng 1 & 2 — xuất bảng ra PDF
# --------------------------------------------------------------------------

def _render_table(column_labels, rows, col_widths, out_path: Path,
                  title: str, figsize, highlight_rows=()) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=column_labels,
                     colWidths=col_widths, cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.55)

    n_cols = len(column_labels)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor(style.GRID)
        if row_idx == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_PRIMARY)
            cell.set_facecolor("#f2f1ee")
            cell.visible_edges = "B"
            cell.set_linewidth(1.0)
        else:
            cell.set_facecolor("#f7f7f5" if row_idx % 2 == 0 else style.SURFACE)
            cell.set_text_props(color=style.INK_PRIMARY)
            if row_idx in highlight_rows:
                cell.set_facecolor("#e6f6ef")
                cell.set_text_props(fontweight="bold", color=style.INK_PRIMARY)
        if col_idx == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
        _ = n_cols

    ax.set_title(title, fontsize=13, fontweight="bold", loc="center", pad=22)

    style.save_pdf(fig, out_path)


def table_full_metrics(table, out_path: Path) -> None:
    columns = ["Hệ thống", "Điều kiện", "Ngưỡng", "F1 (HATE)", "KTC 95%",
               "Precision", "Recall", "macro F1", "Accuracy"]
    rows, highlight = [], []
    for system in SYSTEM_ORDER:
        for condition in CONDITION_ORDER:
            cell = table[(system, condition)]
            rows.append([
                SYSTEM_LABELS[system],
                condition.replace("_", " · "),
                f"{cell.threshold:.3f}" if cell.threshold is not None else "—",
                f"{cell.hate_f1:.4f}",
                f"{cell.f1_ci_low:.3f}–{cell.f1_ci_high:.3f}",
                f"{cell.hate_precision:.3f}",
                f"{cell.hate_recall:.3f}",
                f"{cell.macro_f1:.4f}",
                f"{cell.accuracy:.3f}" if cell.accuracy is not None else "—",
            ])
            if system == "b3_t5ft_norm":
                highlight.append(len(rows))  # +1 vì hàng 0 là tiêu đề

    _render_table(columns, rows, [0.25, 0.12, 0.07, 0.09, 0.12, 0.09, 0.08, 0.09, 0.09],
                  out_path, "Bảng 1 — Chỉ số đầy đủ: 6 hệ thống × 3 điều kiện (tập test)",
                  figsize=(13.5, 5.6), highlight_rows=tuple(highlight))


def table_mcnemar(comparisons, out_path: Path) -> None:
    columns = ["Câu hỏi so sánh", "Điều kiện", "Hệ thống A", "Hệ thống B",
               "F1 A", "F1 B", "ΔF1", "p-value", "Kết luận"]
    rows, highlight = [], []
    for comparison in comparisons:
        delta = comparison["hate_f1_b"] - comparison["hate_f1_a"]
        significant = comparison["significant"]
        rows.append([
            QUESTION_SHORT.get(comparison["question"], comparison["question"]),
            comparison["condition"].replace("_", " · "),
            SYSTEM_SHORT[comparison["system_a"]],
            SYSTEM_SHORT[comparison["system_b"]],
            f"{comparison['hate_f1_a']:.4f}",
            f"{comparison['hate_f1_b']:.4f}",
            f"{delta:+.4f}",
            f"{comparison['p_value']:.2e}",
            "có ý nghĩa" if significant else "không",
        ])
        if significant:
            highlight.append(len(rows))

    _render_table(columns, rows, [0.27, 0.11, 0.08, 0.08, 0.07, 0.07, 0.08, 0.09, 0.09],
                  out_path, "Bảng 2 — Kiểm định ý nghĩa thống kê (McNemar ghép cặp)",
                  figsize=(14.5, 5.0), highlight_rows=tuple(highlight))


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ + bảng Bước 3, xuất PDF")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--out_dir", type=str, default=None,
                        help="mặc định: {results_dir}/figures")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    set_global_seed(args.seed, verbose=False)
    style.apply_rcparams()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_report_table(results_dir)
    comparisons = load_comparisons(results_dir)
    scores = load_scores(results_dir, "test")

    jobs = [
        ("hinh1_ma_tran_f1.pdf", lambda p: figure_f1_matrix(table, p)),
        ("hinh2_do_tut_hieu_nang.pdf", lambda p: figure_degradation(table, p)),
        ("hinh3_duong_cong_roc.pdf", lambda p: figure_roc(scores, p)),
        ("hinh4_duong_cong_pr.pdf", lambda p: figure_pr(scores, p)),
        ("hinh5_dong_gop_tung_buoc.pdf", lambda p: figure_contribution(table, p)),
        ("hinh6_precision_recall.pdf", lambda p: figure_precision_recall_bars(table, p)),
        ("bang1_chi_so_day_du.pdf", lambda p: table_full_metrics(table, p)),
        ("bang2_mcnemar.pdf", lambda p: table_mcnemar(comparisons, p)),
    ]

    for filename, draw in jobs:
        path = out_dir / filename
        draw(path)
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")

    print(f"\nXong {len(jobs)} file PDF trong {out_dir}/")


if __name__ == "__main__":
    main()
