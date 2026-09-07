"""Đẩy checkpoint ViHateT5-v2 lên Hugging Face Hub.

Cần đăng nhập trước một lần:  hf auth login   (token phải có quyền `write`)

    python -m scripts.upload_model_to_hf --repo_id <user>/safevichi-vihatet5-v2
    python -m scripts.upload_model_to_hf --repo_id <user>/... --private   # repo riêng tư

Script upload cả thư mục checkpoint (config, safetensors, tokenizer, model card)
cộng thêm normalizer.py + teencode_dict.json để người dùng chạy được đúng pipeline
đầu vào mà model đã được huấn luyện trên đó.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

CHECKPOINT_DIR = Path("models/vihatet5-v2-best")
NORMALIZER_DIR = Path("src/normalization")
EXTRA_FILES = ["normalizer.py", "teencode_dict.json"]

# Không đẩy lên Hub: log train nội bộ, cache, artefact của Trainer không cần cho inference.
IGNORE = ["*.pyc", "__pycache__/*", "optimizer.pt", "scheduler.pt", "rng_state*.pth"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True, help="vd. truongdv/safevichi-vihatet5-v2")
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--private", action="store_true", help="tạo repo riêng tư")
    parser.add_argument("--commit_message", default="Upload SafeViChi ViHateT5-v2 checkpoint")
    args = parser.parse_args()

    ckpt = args.checkpoint_dir
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"Không thấy model.safetensors trong {ckpt}")
    if not (ckpt / "README.md").exists():
        raise SystemExit(f"Thiếu model card {ckpt / 'README.md'}")

    api = HfApi()
    who = api.whoami()
    print(f"Đăng nhập với: {who['name']}")

    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"Repo sẵn sàng: https://huggingface.co/{args.repo_id}")

    print(f"Upload {ckpt} ...")
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=IGNORE,
        commit_message=args.commit_message,
    )

    for name in EXTRA_FILES:
        src = NORMALIZER_DIR / name
        if not src.exists():
            print(f"  [!] bỏ qua, không tìm thấy {src}")
            continue
        print(f"Upload {src} ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {name} (lớp chuẩn hóa đầu vào)",
        )

    print(f"\nXong: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
