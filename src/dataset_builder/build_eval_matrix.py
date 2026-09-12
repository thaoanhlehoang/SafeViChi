"""Dựng 3 điều kiện input cho ma trận đánh giá baseline (Step 3).

Cả 3 file sinh ra từ CÙNG một tập dòng (`test.jsonl` của v1_2, 2376 dòng),
giữ nguyên thứ tự và gắn `row_id` để so cặp (paired) được giữa các điều kiện
và giữa các hệ thống:

  C0 `C0_clean.jsonl`      — cột `original_text`: câu gốc, CHƯA bị cấy biến
                               thể. Không dedup theo duplicate_group_id (khác
                               với hai điều kiện kia) để khớp 1-1
                               từng dòng với C1.
  C1 `test_perturbed.jsonl`  — cột `text`: câu ĐÃ bị cấy biến thể, CHƯA
                               normalize. Đây là input chính, mọi baseline
                               đều nhận đúng chuỗi này; normalize() (nếu có)
                               là bộ phận bên trong từng hệ thống, chạy lúc
                               eval chứ không nướng sẵn vào file.
  C2 `test_blindspot.jsonl`  — biến thể điểm mù sinh từ `original_text`, cố ý
                               nằm ngoài vùng phủ normalizer (bất biến
                               normalize(x) == x được verify từng dòng).
                               Dòng nào sinh không thành công thì bỏ, nên file
                               này có thể ít hơn 2376 dòng.

Nhãn giữ nguyên phân bố tự nhiên (~18% HATE / 82% CLEAN) — KHÔNG cân bằng,
xem README ma trận đánh giá: cân bằng bằng cách downsample CLEAN sẽ phá mất
ước lượng precision, vốn là chỗ baseline blacklist bộc lộ điểm yếu.

Chạy: python -m src.dataset_builder.build_eval_matrix
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.normalization.normalizer import normalize
from src.utils.seed import SEED_DEFAULT, set_global_seed
from src.variant_generator.blindspot import BLINDSPOT_VERSION, generate_blindspot

LABELS = ("CLEAN", "HATE")


def read_source(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["label"] not in LABELS:
                continue
            rows.append(d)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["label"]] = out.get(r["label"], 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng 3 điều kiện input cho ma trận baseline")
    parser.add_argument("--source_path", type=str,
                        default="data/processed/vietnamese_nonstandard_v1_2/test.jsonl")
    parser.add_argument("--out_dir", type=str, default="data/eval_baseline/test")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    set_global_seed(args.seed)

    source_path = Path(args.source_path)
    out_dir = Path(args.out_dir)
    rows = read_source(source_path)
    print(f"Nguồn: {source_path} — {len(rows)} dòng {label_counts(rows)}")

    clean_rows, perturbed_rows, blindspot_rows = [], [], []
    rng = random.Random(args.seed)
    failed = 0

    for i, d in enumerate(rows):
        row_id = d.get("sample_id", f"row_{i}")
        label = d["label"]
        clean_rows.append({"row_id": row_id, "text": d["original_text"],
                           "label": label, "source": "clean"})
        perturbed_rows.append({"row_id": row_id, "text": d["text"],
                               "label": label, "source": "perturbed"})

        made = generate_blindspot(d["original_text"], normalize, rng)
        if made is None:
            failed += 1
            continue
        text, techniques = made
        blindspot_rows.append({"row_id": row_id, "text": text, "label": label,
                               "source": "blindspot", "techniques": techniques})

    # Bất biến: normalizer phải KHÔNG sửa được gì trong C2, nếu không thì đó
    # là biến thể thường chứ không phải điểm mù.
    for r in blindspot_rows:
        assert normalize(r["text"]) == r["text"], f"Không phải điểm mù: {r['row_id']}"

    write_jsonl(out_dir / "C0_clean.jsonl", clean_rows)
    write_jsonl(out_dir / "C1_perturbed.jsonl", perturbed_rows)
    write_jsonl(out_dir / "C2_blindspot.jsonl", blindspot_rows)

    manifest = {
        "source_path": str(source_path),
        "seed": args.seed,
        "blindspot_version": BLINDSPOT_VERSION,
        "blindspot_generation_failed": failed,
        "conditions": {
            "C0_clean": {"file": "C0_clean.jsonl", "n": len(clean_rows),
                         "by_label": label_counts(clean_rows),
                         "desc": "original_text, chưa cấy biến thể"},
            "C1_perturbed": {"file": "C1_perturbed.jsonl", "n": len(perturbed_rows),
                             "by_label": label_counts(perturbed_rows),
                             "desc": "text đã cấy biến thể, chưa normalize"},
            "C2_blindspot": {"file": "C2_blindspot.jsonl", "n": len(blindspot_rows),
                             "by_label": label_counts(blindspot_rows),
                             "desc": "điểm mù ngoài vùng phủ normalizer"},
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for name, info in manifest["conditions"].items():
        print(f"  {name:15s} {info['n']:5d} dòng  {info['by_label']}  -> {info['file']}")
    print(f"  (blindspot sinh hỏng: {failed} dòng, đã bỏ)")
    print(f"Manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
