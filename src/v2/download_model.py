"""Tự động tải gói mô hình SafeViChi v2 (`safevichi_v2`) từ Hugging Face Hub.

Dùng bởi CLI `safevichi` khi chạy lần đầu trên máy chưa có trọng số cục bộ,
hoặc chạy độc lập để tải trước:

    python -m src.v2.download_model                 # Gói ONNX INT8 mặc định (~334 MiB)
    python -m src.v2.download_model --fp32          # Gói ONNX FP32 (~1,18 GiB)
    python -m src.v2.download_model --with-pytorch  # Thêm trọng số PyTorch model_best.safetensors
    python -m src.v2.download_model --all           # Tải đủ INT8 + FP32 + PyTorch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_REPO_ID = os.environ.get("SAFEVICHI_HF_REPO", "").strip() or "dminhnhat06/safevichi_v2"
DEFAULT_HF_REVISION = os.environ.get("SAFEVICHI_HF_REVISION", "").strip() or "main"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "dualorc_mmbert_base"
_ENV_ONNX_DIR = os.environ.get("SAFEVICHI_ONNX_DIR", "").strip()
DEFAULT_ONNX_DIR = Path(_ENV_ONNX_DIR) if _ENV_ONNX_DIR else (DEFAULT_MODEL_DIR / "onnx")

COMMON_ONNX_FILES: tuple[str, ...] = (
    "crf_params.json",
    "export_manifest.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/backbone_config.json",
)
INT8_ONNX_FILENAME = "safevichi_v2_int8.onnx"
FP32_ONNX_FILENAME = "safevichi_v2_fp32.onnx"
PYTORCH_EXTRA_FILES: tuple[str, ...] = (
    "model_best.safetensors",
    "config.json",
    "calibration.json",
)


def required_onnx_files(use_fp32: bool = False) -> tuple[str, ...]:
    model_file = FP32_ONNX_FILENAME if use_fp32 else INT8_ONNX_FILENAME
    return (model_file, *COMMON_ONNX_FILES)


def missing_onnx_files(onnx_dir: Path, use_fp32: bool = False) -> list[str]:
    base = Path(onnx_dir)
    return [rel for rel in required_onnx_files(use_fp32=use_fp32) if not (base / rel).exists()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one_hf(
    repo_id: str,
    remote_path: str,
    local_dest: Path,
    revision: str = DEFAULT_HF_REVISION,
) -> Path:
    """Tải 1 tệp từ Hugging Face Hub về `local_dest` (ưu tiên huggingface_hub, fallback urllib)."""
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                revision=revision,
                repo_type="model",
            )
        )
        tmp_dest = local_dest.with_suffix(local_dest.suffix + ".part")
        shutil.copy2(cached, tmp_dest)
        tmp_dest.replace(local_dest)
        return local_dest
    except ImportError:
        pass

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{remote_path}"
    tmp_dest = local_dest.with_suffix(local_dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "SafeViChi/2.0"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_dest.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0 and sys.stderr.isatty():
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  Đang tải {remote_path}: {downloaded / (1024 ** 2):.1f}/{total / (1024 ** 2):.1f} MiB ({pct:.0f}%)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
        if sys.stderr.isatty():
            print(file=sys.stderr)
        tmp_dest.replace(local_dest)
        return local_dest
    except Exception:
        tmp_dest.unlink(missing_ok=True)
        raise


def _verify_downloaded_manifest(onnx_dir: Path, downloaded_rels: Iterable[str]) -> None:
    manifest_path = onnx_dir / "export_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    shas = manifest.get("sha256") or {}
    check_map = {
        INT8_ONNX_FILENAME: shas.get("onnx_int8"),
        FP32_ONNX_FILENAME: shas.get("onnx_fp32"),
        "crf_params.json": shas.get("crf_params"),
    }
    for rel in downloaded_rels:
        expected = check_map.get(rel)
        if not expected:
            continue
        target = onnx_dir / rel
        if target.exists():
            actual = _sha256_file(target)
            if actual != expected:
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Sai lệch SHA-256 cho {rel} (kỳ vọng {expected[:12]}..., nhận {actual[:12]}...). "
                    "Hãy chạy lại để tải lại tệp."
                )


def ensure_onnx_model(
    onnx_dir: Path = DEFAULT_ONNX_DIR,
    use_fp32: bool = False,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Đảm bảo đủ bộ tệp ONNX cục bộ trong `onnx_dir`; tự động tải từ Hugging Face nếu còn thiếu."""
    target_dir = Path(onnx_dir)
    needed = list(required_onnx_files(use_fp32=use_fp32)) if force else missing_onnx_files(target_dir, use_fp32=use_fp32)
    if not needed:
        return target_dir

    label = "ONNX FP32 (~1,18 GiB)" if use_fp32 else "ONNX Hybrid INT8 (~334 MiB)"
    if not quiet:
        print(
            f"Chưa có đủ mô hình cục bộ tại {target_dir}. "
            f"Đang tải gói {label} từ Hugging Face ({repo_id})...",
            file=sys.stderr,
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in needed:
        remote_path = f"onnx/{rel}"
        if not quiet:
            print(f"  ↓ {remote_path}", file=sys.stderr)
        _download_one_hf(repo_id=repo_id, remote_path=remote_path, local_dest=target_dir / rel, revision=revision)

    _verify_downloaded_manifest(target_dir, needed)
    if not quiet:
        print(f"Đã tải và xác thực SHA-256 mô hình tại: {target_dir}", file=sys.stderr)
    return target_dir


def download_pytorch_weights(
    model_dir: Path = DEFAULT_MODEL_DIR,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str = DEFAULT_HF_REVISION,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Tải thêm trọng số PyTorch (`model_best.safetensors`, `config.json`, `calibration.json`)."""
    target_dir = Path(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in PYTORCH_EXTRA_FILES:
        dst = target_dir / rel
        if dst.exists() and not force:
            continue
        if not quiet:
            print(f"  ↓ {rel}", file=sys.stderr)
        _download_one_hf(repo_id=repo_id, remote_path=rel, local_dest=dst, revision=revision)
    return target_dir


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID, help=f"Hugging Face repo ID (mặc định: {DEFAULT_HF_REPO_ID})")
    parser.add_argument("--revision", default=DEFAULT_HF_REVISION, help="Nhánh/tag/commit trên Hugging Face (mặc định: main)")
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR, help="Thư mục lưu gói ONNX")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Thư mục lưu trọng số PyTorch (--with-pytorch)")
    parser.add_argument("--fp32", action="store_true", help="Tải gói ONNX FP32 thay vì INT8")
    parser.add_argument("--with-pytorch", action="store_true", help="Tải thêm model_best.safetensors (~1,15 GiB)")
    parser.add_argument("--all", action="store_true", help="Tải trọn gói ONNX INT8 + ONNX FP32 + PyTorch safetensors")
    parser.add_argument("--force", action="store_true", help="Tải đè lại dù tệp đã tồn tại cục bộ")
    args = parser.parse_args(argv)

    if args.all:
        ensure_onnx_model(args.onnx_dir, use_fp32=False, repo_id=args.repo_id, revision=args.revision, force=args.force)
        ensure_onnx_model(args.onnx_dir, use_fp32=True, repo_id=args.repo_id, revision=args.revision, force=args.force)
        download_pytorch_weights(args.model_dir, repo_id=args.repo_id, revision=args.revision, force=args.force)
    else:
        ensure_onnx_model(args.onnx_dir, use_fp32=args.fp32, repo_id=args.repo_id, revision=args.revision, force=args.force)
        if args.with_pytorch:
            download_pytorch_weights(args.model_dir, repo_id=args.repo_id, revision=args.revision, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
