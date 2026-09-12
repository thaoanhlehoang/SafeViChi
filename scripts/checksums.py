"""Băm SHA-256 mọi file dữ liệu của Bước 3 và đối chiếu.

Vì sao cần: seed chỉ đảm bảo "cùng đầu vào + cùng code -> cùng đầu ra". Nó
KHÔNG phát hiện được chuyện file bị sửa tay, bị copy nhầm, hay bị sinh lại bằng
một phiên bản code khác. Checksum thì phát hiện được.

Ghi bảng băm (chỉ chạy khi cố ý chốt lại một bộ dữ liệu mới):
    python -m scripts.checksums --write

Đối chiếu (mặc định — dùng trước mọi lần train/eval):
    python -m scripts.checksums

Mã thoát: 0 nếu khớp hết, 1 nếu có sai lệch. Dùng được trong CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MANIFEST_PATH = Path("data/CHECKSUMS.json")

# Các file làm nên kết quả Bước 3. Thiếu file nào ở đây là mất khả năng tái lập.
TRACKED = [
    "data/data_final/train.jsonl",
    "data/data_final/validation.jsonl",
    "data/data_final/validation_mixed.jsonl",
    "data/data_final/test.jsonl",
    "data/data_final/manifest_mix.json",
    "data/eval_baseline/test/C0_clean.jsonl",
    "data/eval_baseline/test/C1_perturbed.jsonl",
    "data/eval_baseline/test/C2_blindspot.jsonl",
    "data/eval_baseline/test/manifest.json",
    "data/eval_baseline/val/C0_clean.jsonl",
    "data/eval_baseline/val/C1_perturbed.jsonl",
    "data/eval_baseline/val/C2_blindspot.jsonl",
    "data/eval_baseline/val/manifest.json",
    "models/vihatet5-v2-best/model.safetensors",
    "models/vihatet5-v2-best/config.json",
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def line_count(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def collect() -> dict:
    entries = {}
    for rel in TRACKED:
        path = Path(rel)
        if not path.exists():
            entries[rel] = {"missing": True}
            continue
        entries[rel] = {
            "sha256": sha256_of(path),
            "bytes": path.stat().st_size,
            "lines": line_count(path),
        }
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Băm và đối chiếu dữ liệu Bước 3")
    parser.add_argument("--write", action="store_true",
                        help="ghi đè bảng băm bằng trạng thái đĩa hiện tại")
    args = parser.parse_args()

    current = collect()

    if args.write:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        for rel, info in current.items():
            mark = "THIẾU" if info.get("missing") else info["sha256"][:16]
            print(f"  {mark:16s} {rel}")
        print(f"\nĐã ghi {MANIFEST_PATH} ({len(current)} mục)")
        return 0

    if not MANIFEST_PATH.exists():
        print(f"[lỗi] Chưa có {MANIFEST_PATH}. Chạy --write để chốt lần đầu.")
        return 1

    expected = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    problems = []
    for rel, want in expected.items():
        got = current.get(rel, {"missing": True})
        if got.get("missing"):
            problems.append(f"THIẾU FILE   {rel}")
        elif want.get("missing"):
            problems.append(f"THỪA FILE    {rel} (bảng băm ghi là thiếu)")
        elif got["sha256"] != want["sha256"]:
            problems.append(
                f"KHÁC NỘI DUNG {rel}\n"
                f"    mong đợi {want['sha256'][:16]}… ({want['lines']} dòng, {want['bytes']} byte)\n"
                f"    thực tế  {got['sha256'][:16]}… ({got['lines']} dòng, {got['bytes']} byte)")
        else:
            print(f"  OK  {rel}")

    if problems:
        print("\n=== SAI LỆCH ===")
        for p in problems:
            print(f"  {p}")
        print(f"\n{len(problems)}/{len(expected)} mục không khớp.")
        return 1

    print(f"\nKhớp toàn bộ {len(expected)} mục.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
