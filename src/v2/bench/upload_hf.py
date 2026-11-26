"""Đẩy gói mô hình SafeViChi v2 (`safevichi_v2`) lên Hugging Face Hub.

Yêu cầu đăng nhập trước một lần: `hf auth login` (token có quyền `write`).

Ví dụ:
    python -m src.v2.bench.upload_hf --repo-id dminhnhat06/safevichi_v2
    python -m src.v2.bench.upload_hf --repo-id dminhnhat06/safevichi_v2 --private
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "dminhnhat06/safevichi_v2"
DEFAULT_MODEL_DIR = Path("models/dualorc_mmbert_base")
DEFAULT_EVAL_DIR = Path("results/v2/official_eval/dualorc_mmbert_base")

MODEL_CARD = """---
language:
- vi
license: mit
base_model: jhu-clsp/mmBERT-base
tags:
- vietnamese
- content-moderation
- hate-speech-detection
- toxicity
- onnx
- int8
- safevichi
pipeline_tag: text-classification
---

# SafeViChi v2 (`safevichi_v2`)

Mô hình phát hiện tin nhắn tiếng Việt nguy hại **chạy hoàn toàn cục bộ trên thiết bị (on-device)**, phân loại **5 nhãn theo tình huống** và trích xuất **cụm từ bằng chứng chép nguyên văn (verbatim)** trên hội thoại 1–8 lượt, suy luận trực tiếp trên văn bản gốc mà không cần bộ chuẩn hóa.

- **Kiến trúc:** `SafeVi-DualORC-CRF` (backbone `jhu-clsp/mmBERT-base`, 307M tham số + UTF-8 `Byte-CNN` + `HAT` + `TurnAwareBIOCRF` + `ORC`).
- **Bộ 5 nhãn:** `AN_TOAN`, `XUC_PHAM`, `QUAY_ROI`, `DE_DOA`, `LUA_DAO`.
- **Triển khai mặc định:** `onnx/safevichi_v2_int8.onnx` (`300,69 MiB`, lượng tử hóa Hybrid INT8: `PerRowInt8Embedding` + `W8A8` Attention/`mlp.Wi` + `W8A32` cho 22 tầng `GeGLU mlp.Wo` + đầu `DualORC-CRF` FP32).

## Cấu trúc tệp trong kho mô hình

| Tệp | Dung lượng | Mục đích |
| :--- | :---: | :--- |
| `onnx/safevichi_v2_int8.onnx` | ~300,7 MiB | Mô hình suy luận ONNX Hybrid INT8 mặc định cho CLI `safevichi` (CPU) |
| `onnx/safevichi_v2_fp32.onnx` | ~1,15 GiB | Mô hình suy luận ONNX FP32 (`safevichi --fp32`, đối chiếu chính xác) |
| `onnx/crf_params.json` | ~1 KiB | Tham số nhiệt độ hiệu chuẩn `T`, ngưỡng giải mã bằng chứng & ma trận chuyển BIO-CRF |
| `onnx/export_manifest.json` | ~2 KiB | SHA-256, dung lượng và kiểm chứng sai số số học FP32 / INT8 |
| `onnx/tokenizer/` | ~32,8 MiB | Tokenizer `mmBERT-base` (`tokenizer.json`, `tokenizer_config.json`, `backbone_config.json`) |
| `model_best.safetensors` | ~1,15 GiB | Trọng số PyTorch FP32 gốc để tái lập hoặc xuất lại ONNX |
| `config.json`, `calibration.json` | ~33 KiB | Cấu hình kiến trúc và tham số hiệu chuẩn đóng băng trên tập `cal` |

## Hiệu năng trên bộ test khóa (1.670 mẫu: 334 C0 + 668 C1 + 668 C2)

| Tầng đánh giá | FP32 Macro-F1 | INT8 Macro-F1 | FP32 Binary-F1 | INT8 Binary-F1 | FP32 Evidence F1 theo ý | INT8 Evidence F1 theo ý |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C0` (Sạch, n=334)** | `0,8647` | `0,8598` (`99,43%`) | `0,9287` | `0,9333` | `0,8303` | `0,8401` |
| **`C1` (Né lọc, n=668)** | `0,7859` | `0,7767` (`98,83%`) | `0,9029` | `0,9053` | `0,7649` | `0,7674` |
| **`C2` (Điểm mù 100% chưa thấy, n=668)** | `0,7708` | `0,7567` (`98,17%`) | `0,9087` | `0,9061` | `0,7531` | `0,7630` |
| **Toàn bộ (`ALL`, n=1.670)** | `0,7960` | `0,7857` (`98,71%`) | `0,9104` | `0,9113` | `0,7737` | `0,7808` |

- **Hiệu năng Laptop CPU (`AMD Ryzen AI 7 H 350`, 8 luồng, `batch=1`, ONNX Hybrid INT8):** độ trễ `p50 = 24,68 ms`, `p95 = 31,54 ms` (`41,3 tin/s`), khởi động lạnh `1,53 s`, RAM đỉnh `986 MiB`, `0,00%` cụm bằng chứng lệch nguyên văn.

## Cách sử dụng nhanh với SafeViChi CLI

```bash
git clone https://github.com/thaoanhlehoang/SafeViChi.git
cd SafeViChi
pip install -e .

# Lần chạy đầu sẽ tự động tải gói ONNX INT8 (~334 MiB) từ Hugging Face
safevichi "Tài khoản sẽ bị khóa sau 2 giờ, bấm link xác minh OTP ngay"
```

Hoặc tải trước mô hình:

```bash
python -m src.v2.download_model              # Chỉ gói ONNX INT8 (~334 MiB, mặc định)
python -m src.v2.download_model --fp32       # Gói ONNX FP32 (~1,18 GiB)
python -m src.v2.download_model --all        # Trọn gói ONNX INT8 + FP32 + PyTorch safetensors
```

## Giấy phép

Phát hành theo giấy phép [MIT](https://opensource.org/licenses/MIT).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face repo ID (mặc định: {DEFAULT_REPO_ID})")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--private", action="store_true", help="Tạo repo riêng tư")
    parser.add_argument("--include-fp32", action="store_true", default=True)
    parser.add_argument("--no-fp32", dest="include_fp32", action="store_false")
    parser.add_argument("--include-pytorch", action="store_true", default=True)
    parser.add_argument("--no-pytorch", dest="include_pytorch", action="store_false")
    parser.add_argument("--commit-message", default="Upload SafeViChi v2 (safevichi_v2) ONNX INT8/FP32 and PyTorch weights")
    args = parser.parse_args(argv)

    onnx_dir = args.model_dir / "onnx"
    files_to_upload: list[tuple[Path, str]] = [
        (onnx_dir / "safevichi_v2_int8.onnx", "onnx/safevichi_v2_int8.onnx"),
        (onnx_dir / "crf_params.json", "onnx/crf_params.json"),
        (onnx_dir / "export_manifest.json", "onnx/export_manifest.json"),
        (onnx_dir / "tokenizer" / "tokenizer.json", "onnx/tokenizer/tokenizer.json"),
        (onnx_dir / "tokenizer" / "tokenizer_config.json", "onnx/tokenizer/tokenizer_config.json"),
        (onnx_dir / "tokenizer" / "backbone_config.json", "onnx/tokenizer/backbone_config.json"),
        (args.eval_dir / "config.json", "config.json"),
        (args.eval_dir / "calibration.json", "calibration.json"),
    ]
    if args.include_fp32:
        files_to_upload.append((onnx_dir / "safevichi_v2_fp32.onnx", "onnx/safevichi_v2_fp32.onnx"))
    if args.include_pytorch:
        files_to_upload.append((args.model_dir / "model_best.safetensors", "model_best.safetensors"))

    missing = [str(src) for src, _ in files_to_upload if not src.exists()]
    if missing:
        raise SystemExit("Thiếu tệp cần upload:\n  - " + "\n  - ".join(missing))

    api = HfApi()
    who = api.whoami()
    print(f"Đăng nhập Hugging Face với tài khoản: {who['name']}")

    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo sẵn sàng: https://huggingface.co/{args.repo_id}")

    total_mib = sum(src.stat().st_size for src, _ in files_to_upload) / (1024 ** 2)
    print(f"Chuẩn bị tải lên {len(files_to_upload) + 1} tệp (~{total_mib:.1f} MiB)...")

    stage_parent = args.model_dir.resolve().parent
    with tempfile.TemporaryDirectory(prefix="_hf_stage_", dir=str(stage_parent)) as tmp:
        stage = Path(tmp)
        (stage / "README.md").write_text(MODEL_CARD.strip() + "\n", encoding="utf-8")
        for src, rel in files_to_upload:
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Dùng hardlink khi cùng ổ đĩa để tránh copy vài GiB; fallback sang copy2 nếu khác ổ đĩa
            try:
                dst.hardlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

        api.upload_folder(
            folder_path=str(stage),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    print(f"\nHoàn tất! Xem mô hình tại: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
