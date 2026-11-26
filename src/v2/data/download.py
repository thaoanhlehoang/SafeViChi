"""Tải dữ liệu gốc cho v2 từ Hugging Face, ghim theo revision để tái hiện được.

- ViHSD (`uitnlp/vihsd`): đủ train/dev/test. Bộ v1_2 chỉ dùng split train, v2 cần cả ba.
- ViTHSD (`sonlam1102/vithsd`): train/dev/test dạng `*.xlsx.csv` đúng như `merge.py` đọc.

Cả hai bộ đều gated: trước khi chạy cần mở trang dataset trên Hugging Face, đăng nhập và bấm chấp nhận
điều khoản, rồi tạo token (quyền read) và đặt vào biến môi trường `HF_TOKEN` (hoặc dòng `HF_TOKEN=...`
trong `.env`, không commit). File CSV bị `.gitignore` chặn, không phân phối lại qua repo; chỉ commit
`data/raw/NGUON_TAI.json` (revision và SHA-256 từng file).

Chạy:
    python -m src.v2.data.download            # tải cả hai bộ, bỏ qua file đã có đúng checksum
    python -m src.v2.data.download --chi vithsd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.v2.labeling.llm import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "NGUON_TAI.json"

DATASETS = {
    "vihsd": {
        "repo": "uitnlp/vihsd",
        "revision": "88e81b36ca376867640dad9df295e0f7d499ab80",  # trùng revision v1_2 đã dùng
        "files": ["train.csv", "dev.csv", "test.csv"],
    },
    "vithsd": {
        "repo": "sonlam1102/vithsd",
        "revision": "b7634be9c1fecfe8cb94bd71927adf523a875ef4",
        "files": ["train.xlsx.csv", "dev.xlsx.csv", "test.xlsx.csv"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"


def download(url: str, dest: Path, token: str) -> None:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except urllib.error.HTTPError as error:
        tmp.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise RuntimeError(
                f"{error.code} khi tải {url}. Kiểm tra HF_TOKEN và đã bấm chấp nhận điều khoản trên trang dataset."
            ) from error
        raise
    tmp.replace(dest)


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chi", nargs="*", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        print("Thiếu HF_TOKEN (biến môi trường hoặc .env). Xem docstring của script.", file=sys.stderr)
        return 2

    manifest = load_manifest()
    for name in args.chi:
        spec = DATASETS[name]
        entry = manifest.get(name, {})
        files_meta = entry.get("files", {}) if entry.get("revision") == spec["revision"] else {}
        for filename in spec["files"]:
            dest = args.raw_dir / name / filename
            known = files_meta.get(filename, {}).get("sha256")
            if dest.exists() and known and sha256_file(dest) == known:
                print(f"Đã có {dest} (đúng checksum), bỏ qua.")
                continue
            url = file_url(spec["repo"], spec["revision"], filename)
            print(f"Tải {spec['repo']}@{spec['revision'][:8]}/{filename} ...")
            download(url, dest, token)
            files_meta[filename] = {"sha256": sha256_file(dest), "bytes": dest.stat().st_size}
        manifest[name] = {"repo": spec["repo"], "revision": spec["revision"], "files": files_meta}

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
