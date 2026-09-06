"""Quản lý ngẫu nhiên tập trung cho toàn bộ SafeViChi.

Vì sao cần file này: trước đây mỗi script tự seed một kiểu — `build.py` tự dẫn
seed bằng hash, `build_mixed.py` dùng `random.Random(f"{seed}:{split}")`,
`stats.py` dùng `np.random.default_rng(42)` cứng, còn `train.py` thì KHÔNG hề
seed `torch`/`numpy` tường minh mà phó mặc cho `Seq2SeqTrainingArguments(seed=)`.
Kết quả là không ai trả lời được câu "chạy lại có ra đúng số cũ không".

Ở đây gom lại hai nhóm hàm:

  1. `set_global_seed()` — seed MỌI nguồn ngẫu nhiên toàn cục (python, numpy,
     torch CPU/CUDA, PYTHONHASHSEED). Gọi ở dòng đầu của mọi entrypoint.

  2. `derive_seed()` / `derive_rng()` — sinh seed CON dẫn xuất từ seed gốc theo
     một chuỗi định danh (vd. tên split, row_id). Cùng seed gốc + cùng định
     danh thì luôn ra cùng seed con, KHÔNG phụ thuộc thứ tự duyệt hay số dòng
     đã xử lý trước đó. Đây là điểm khác biệt quan trọng so với việc dùng chung
     một `random.Random` cho cả tập: nếu chèn/xóa một dòng ở giữa, mọi dòng
     phía sau sẽ đổi kết quả.

QUY ƯỚC SEED CỦA DỰ ÁN
----------------------
  20260902  — Bước 1, sinh 75K biến thể (`build.py --seed`). Giữ nguyên giá trị
              lịch sử này, đổi là mất khả năng tái tạo bộ v1_2.
  42        — mọi khâu còn lại: build_mixed, build_eval_matrix, train, bootstrap.

CẢNH BÁO — KHÔNG ĐỔI NGỮ NGHĨA RNG CŨ
-------------------------------------
`build_mixed.py` và `build_eval_matrix.py` vẫn dùng đúng cách seed cũ
(`random.Random(seed)` dùng chung cho cả tập). Cách đó kém bền hơn `derive_rng`,
nhưng bộ dữ liệu đã dùng để tạo ra kết quả Bước 3 được sinh ra bằng chính ngữ
nghĩa đó. Sửa sang `derive_rng` sẽ ra dữ liệu KHÁC, khiến mọi con số trong báo
cáo không tái lập được. Nên: giữ nguyên, và dùng `scripts/verify_data.py` để
kiểm tra checksum sau khi dựng lại.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- Hằng seed của dự án ---------------------------------------------------

SEED_STEP1_DATASET = 20260902  # Bước 1: sinh 75K biến thể
SEED_DEFAULT = 42              # Bước 2/3: trộn dữ liệu, train, bootstrap

_UINT64 = (1 << 64) - 1
_UINT32 = (1 << 32) - 1


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Seed con xác định từ `base_seed` + các thành phần định danh.

    Dùng BLAKE2b chứ không dùng `hash()` của Python: `hash()` của str bị ngẫu
    nhiên hóa theo từng tiến trình (PYTHONHASHSEED) nên không tái lập được
    giữa các lần chạy.

    >>> derive_seed(42, "train", "row_007") == derive_seed(42, "train", "row_007")
    True
    """
    material = "\0".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _UINT64


def derive_rng(base_seed: int, *parts: Any) -> random.Random:
    """`random.Random` độc lập, tái lập được, cho một đơn vị công việc."""
    return random.Random(derive_seed(base_seed, *parts))


def derive_np_rng(base_seed: int, *parts: Any) -> np.random.Generator:
    """`numpy.random.Generator` độc lập, tái lập được."""
    return np.random.default_rng(derive_seed(base_seed, *parts) & _UINT32)


@dataclass
class SeedReport:
    """Ghi lại chính xác đã seed những gì — đem dump vào run_config.json."""

    seed: int
    deterministic: bool
    seeded: list[str] = field(default_factory=list)
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "seeded": self.seeded,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
        }


def set_global_seed(seed: int = SEED_DEFAULT, deterministic: bool = False,
                    verbose: bool = True) -> SeedReport:
    """Seed mọi nguồn ngẫu nhiên toàn cục. Gọi ở dòng đầu của mọi entrypoint.

    `deterministic=True` bật thêm chế độ tất định của cuDNN/cuBLAS. Nó làm
    train CHẬM ĐI RÕ RỆT (tắt tự dò thuật toán tích chập nhanh nhất), nên mặc
    định để False; chỉ bật khi cần chứng minh tái lập bit-cho-bit.

    Lưu ý trung thực: kể cả `deterministic=True`, train trên GPU với fp16 và
    `nn.DataParallel` vẫn có thể lệch nhỏ giữa các lần chạy do thứ tự cộng dồn
    số thực khi gộp gradient từ nhiều thiết bị. Seed đảm bảo cùng dữ liệu, cùng
    thứ tự batch, cùng khởi tạo — không đảm bảo cùng bit cuối cùng.
    """
    report = SeedReport(seed=seed, deterministic=deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)
    report.seeded.append("PYTHONHASHSEED")

    random.seed(seed)
    report.seeded.append("random")

    np.random.seed(seed & _UINT32)
    report.seeded.append("numpy")

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        report.seeded.append("torch")
        report.torch_version = torch.__version__
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            report.seeded.append("torch.cuda")
            report.cuda_available = True
            report.cuda_device_count = torch.cuda.device_count()
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                report.seeded.append("torch.deterministic")
            except Exception as exc:  # pragma: no cover - phụ thuộc build torch
                print(f"  [!] Không bật được use_deterministic_algorithms: {exc}")

    if verbose:
        flag = " (chế độ tất định)" if deterministic else ""
        print(f"[seed] seed={seed}{flag} -> đã seed: {', '.join(report.seeded)}")

    return report
