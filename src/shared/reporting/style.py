"""Bảng màu + thiết lập matplotlib dùng chung cho mọi hình của Bước 3.

Nguyên tắc đã áp dụng (theo phương pháp trực quan hóa dữ liệu chuẩn):

  - MÀU MANG DANH TÍNH, KHÔNG MANG THỨ HẠNG. Mỗi họ baseline (B1/B2/B3) giữ
    nguyên một màu ở MỌI hình. Người đọc học "B3 màu xanh ngọc" một lần là dùng
    được cho cả tài liệu.
  - CHỈ 3 MÀU, KHÔNG PHẢI 6. Biến thể "có normalizer" được mã hóa bằng GẠCH CHÉO
    (hatch) chứ không phải màu thứ 4-5-6. Đây là mã hóa kép: màu = hệ thống nào,
    hoạ tiết = có bật Bước 2 hay không. Sáu màu rời rạc sẽ vượt ngưỡng an toàn
    cho người mù màu và khiến người đọc phải tra chú giải liên tục.
  - Ba màu blue/orange/aqua lấy từ bảng màu đã kiểm định mù màu (khoảng cách
    Delta E cặp kề >= 8 trên thang OKLab), an toàn cả khi in đen trắng nhờ hoạ tiết.
  - Lưới kẻ MỜ, NÉT LIỀN, chỉ theo trục ngang. Lưới nét đứt đọc nhầm thành
    "ngưỡng" hoặc "dự báo" trong khi nó chỉ là lưới.
  - Bỏ khung viền trên/phải để mực giấy dồn cho dữ liệu.

Font: DejaVu Sans (mặc định của matplotlib) có đủ dấu tiếng Việt, không cần cài
thêm. Không dùng font serif/trang trí cho số liệu.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # không cần màn hình; chạy được trên máy chủ/Kaggle
import matplotlib.pyplot as plt

# --- Bảng màu định danh (đã kiểm định mù màu) -------------------------------

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#ffffff"
GRID = "#e3e2de"

FAMILY_COLORS = {
    "b1": "#2a78d6",  # xanh dương — blacklist
    "b2": "#eb6834",  # cam        — ViHateT5 gốc
    "b3": "#1baf7a",  # xanh ngọc  — ViHateT5 fine-tune (pipeline SafeViChi)
}

HATCH_WITH_NORMALIZER = "///"

# --- Tên hiển thị -----------------------------------------------------------

SYSTEM_LABELS = {
    "b1_blacklist": "B1 · Blacklist",
    "b1_blacklist_norm": "B1 · Blacklist + chuẩn hóa",
    "b2_t5base": "B2 · ViHateT5 gốc",
    "b2_t5base_norm": "B2 · ViHateT5 gốc + chuẩn hóa",
    "b3_t5ft": "B3 · ViHateT5 fine-tune",
    "b3_t5ft_norm": "B3 · Pipeline SafeViChi đầy đủ",
}

# Nhãn RÚT GỌN nhưng vẫn phân biệt được có/không bật chuẩn hóa. Cắt thành
# "B2"/"B3" là làm mất phân biệt đó, khiến bảng McNemar ghi sai cặp đang so.
SYSTEM_SHORT = {
    "b1_blacklist": "B1",
    "b1_blacklist_norm": "B1+ch",
    "b2_t5base": "B2",
    "b2_t5base_norm": "B2+ch",
    "b3_t5ft": "B3",
    "b3_t5ft_norm": "B3+ch",
}

SYSTEM_ORDER = [
    "b1_blacklist", "b1_blacklist_norm",
    "b2_t5base", "b2_t5base_norm",
    "b3_t5ft", "b3_t5ft_norm",
]

CONDITION_LABELS = {
    "C0_clean": "C0 · Câu sạch",
    "C1_perturbed": "C1 · Đã né lọc",
    "C2_blindspot": "C2 · Điểm mù",
}

CONDITION_SHORT = {
    "C0_clean": "C0\ncâu sạch",
    "C1_perturbed": "C1\nné lọc",
    "C2_blindspot": "C2\nđiểm mù",
}

CONDITION_ORDER = ["C0_clean", "C1_perturbed", "C2_blindspot"]


def family_of(system: str) -> str:
    return system.split("_", 1)[0]


def uses_normalizer(system: str) -> bool:
    return system.endswith("_norm")


def bar_style(system: str) -> dict:
    """Màu = họ baseline; gạch chéo = có bật normalizer."""
    return {
        "color": FAMILY_COLORS[family_of(system)],
        "hatch": HATCH_WITH_NORMALIZER if uses_normalizer(system) else "",
        "edgecolor": SURFACE,
        "linewidth": 0.0,
    }


def line_style(system: str) -> dict:
    """Nét liền = input thô; nét đứt = có bật normalizer."""
    return {
        "color": FAMILY_COLORS[family_of(system)],
        "linestyle": "--" if uses_normalizer(system) else "-",
        "linewidth": 2.0,
        "marker": "s" if uses_normalizer(system) else "o",
        "markersize": 6,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 1.5,
    }


def apply_rcparams() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
        "grid.color": GRID,
        "grid.linestyle": "-",      # KHÔNG dùng nét đứt cho lưới
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "hatch.linewidth": 0.8,
        "hatch.color": SURFACE,
        # PDF vector thật, chữ nhúng dạng font chứ không phải đường viền
        "pdf.fonttype": 42,
        "pdf.compression": 6,
    })


def style_axes(ax, ygrid: bool = True) -> None:
    """Lưới ngang mờ, nằm DƯỚI dữ liệu."""
    if ygrid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major")
        ax.grid(axis="x", visible=False)


def add_caption(ax, title: str, y: float, x: float = 0.5) -> None:
    """Chú thích "Hình N — …" đặt BÊN DƯỚI hình, CĂN GIỮA, không kèm diễn giải.

    Neo theo TỌA ĐỘ CỦA TRỤC chứ không theo tọa độ hình: `bbox_inches="tight"`
    co khung hình lại sau khi vẽ nên vị trí tuyệt đối trong tọa độ hình không
    đoán trước được, còn tọa độ trục thì luôn giữ đúng khoảng cách dưới trục.
    """
    ax.text(x, y, title, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", color=INK_PRIMARY, ha="center", va="top")


def save_pdf(fig, path) -> None:
    """Ghi hình ra PDF vector. KHÔNG hỗ trợ định dạng khác — theo yêu cầu đề tài."""
    path = str(path)
    assert path.endswith(".pdf"), f"Chỉ xuất PDF, nhận được: {path}"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
