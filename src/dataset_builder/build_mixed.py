"""Build bộ dữ liệu trộn 60/30/10 cho cả train/validation/test.

  60%  perturbed_normalized — câu đã bị biến thể rồi qua normalize() (residual
                              noise). Lấy từ data/processed/normalized/.
  30%  clean_original       — câu gốc sạch, chưa biến thể chưa normalize. Lấy
                              từ v1_2, dedup theo duplicate_group_id.
  10%  blindspot_raw        — biến thể dùng kỹ thuật NGOÀI vùng phủ normalizer
                              (src/variant_generator/blindspot.py), giữ nguyên
                              KHÔNG normalize vì normalizer vốn không sửa được
                              (mỗi câu đều đã kiểm tra normalize(x) == x).

Câu nguồn cho nhóm clean và nhóm blindspot được lấy rời nhau trong cùng pool
của CHÍNH split đó, nên không có rò rỉ giữa train/validation/test.

Cân bằng nhãn: mọi nhóm đều lấy theo CHỈ TIÊU RIÊNG CHO TỪNG NHÃN. Điều này
bắt buộc với nhóm blindspot — nếu lấy gộp rồi cắt cho đủ số lượng, phần lớn
sẽ rơi vào nhãn đa số và model sẽ học nhầm rằng "câu bị né lọc kiểu lạ =>
CLEAN", ngược hẳn thực tế (người ta né lọc để giấu nội dung độc hại).

--balance_eval (mặc định bật): validation/test được dựng cân 50/50 giữa hai
nhãn trong TỪNG nhóm, kích thước bị chặn bởi nhãn hiếm hơn — để đọc F1 không
bị nhiễu bởi nhãn đa số. Tập train giữ nguyên toàn bộ dữ liệu perturbed.

Chạy: python -m src.dataset_builder.build_mixed
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.normalization.normalizer import normalize
from src.utils.seed import SEED_DEFAULT, set_global_seed
from src.variant_generator.blindspot import BLINDSPOT_VERSION, generate_blindspot

SPLITS = ("train", "validation", "test")
LABELS = ("CLEAN", "HATE")


def read_perturbed(path: Path) -> dict[str, list[str]]:
    """Câu đã normalize, nhóm theo nhãn."""
    rows: dict[str, list[str]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[d["label"]].append(d["text"])
    return rows


def read_original_pool(path: Path) -> dict[str, list[str]]:
    """original_text dedup theo duplicate_group_id, nhóm theo nhãn."""
    seen: dict[str, tuple[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["label"] not in LABELS:
                continue
            gid = d["duplicate_group_id"]
            if gid not in seen:
                seen[gid] = (d["original_text"], d["label"])
    pool: dict[str, list[str]] = defaultdict(list)
    for text, label in seen.values():
        pool[label].append(text)
    return pool


class LabelPool:
    """Phát câu theo từng nhãn, không bao giờ phát lại câu đã dùng."""

    def __init__(self, pool: dict[str, list[str]], rng: random.Random):
        self.pool = {label: list(texts) for label, texts in pool.items()}
        for texts in self.pool.values():
            rng.shuffle(texts)
        self.cursor = {label: 0 for label in self.pool}

    def available(self, label: str) -> int:
        return len(self.pool.get(label, [])) - self.cursor.get(label, 0)

    def take(self, label: str, n: int) -> list[str]:
        start = self.cursor[label]
        chunk = self.pool[label][start:start + n]
        self.cursor[label] = start + len(chunk)
        return chunk


def make_blindspot_rows(
    pool: LabelPool,
    want: dict[str, int],
    rng: random.Random,
    batch: int = 256,
) -> tuple[list[dict], int]:
    """Sinh biến thể điểm mù cho ĐỦ chỉ tiêu của từng nhãn.

    Một số câu không sinh nổi biến thể đạt điều kiện normalize(x) == x, nên cứ
    rút thêm câu nguồn cho tới khi đủ chỉ tiêu hoặc cạn pool.
    """
    rows: list[dict] = []
    failed = 0
    for label, target in want.items():
        made = 0
        while made < target and pool.available(label) > 0:
            for text in pool.take(label, min(batch, target - made + batch)):
                if made >= target:
                    break
                result = generate_blindspot(text, normalize, rng)
                if result is None:
                    failed += 1
                    continue
                variant, techniques = result
                rows.append({
                    "text": variant,
                    "label": label,
                    "source": "blindspot_raw",
                    "blindspot_techniques": techniques,
                })
                made += 1
    return rows, failed


def plan_counts(
    perturbed: dict[str, list[str]],
    pool: LabelPool,
    clean_ratio: float,
    blind_ratio: float,
    balance: bool,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Trả về chỉ tiêu số dòng theo nhãn cho 3 nhóm (perturbed, clean, blind)."""
    pert_ratio = 1.0 - clean_ratio - blind_ratio

    if balance:
        # Cân 50/50: kích thước bị chặn bởi nhãn hiếm hơn, ở cả nguồn perturbed
        # lẫn pool câu gốc (pool phải gánh cả nhóm clean lẫn nguồn cho blindspot).
        per_label_pert = min(len(perturbed[label]) for label in LABELS)
        per_label_pool = min(pool.available(label) for label in LABELS)
        total = min(
            per_label_pert * 2 / pert_ratio,
            per_label_pool / ((clean_ratio + blind_ratio) / 2),
        )
        half = {
            "pert": round(total * pert_ratio / 2),
            "clean": round(total * clean_ratio / 2),
            "blind": round(total * blind_ratio / 2),
        }
        return (
            {label: half["pert"] for label in LABELS},
            {label: half["clean"] for label in LABELS},
            {label: half["blind"] for label in LABELS},
        )

    # Không cân: giữ trọn perturbed, hai nhóm còn lại theo đúng tỉ lệ nhãn của pool.
    n_pert = sum(len(v) for v in perturbed.values())
    total = n_pert / pert_ratio
    pool_total = sum(pool.available(label) for label in LABELS)
    pert_counts = {label: len(perturbed[label]) for label in LABELS}
    clean_counts = {
        label: round(total * clean_ratio * pool.available(label) / pool_total) for label in LABELS
    }
    blind_counts = {
        label: round(total * blind_ratio * pool.available(label) / pool_total) for label in LABELS
    }
    return pert_counts, clean_counts, blind_counts


def build_split(
    split: str,
    normalized_dir: Path,
    source_dir: Path,
    seed: int,
    clean_ratio: float,
    blind_ratio: float,
    balance: bool,
) -> tuple[list[dict], dict]:
    rng = random.Random(f"{seed}:{split}")

    perturbed = read_perturbed(normalized_dir / f"{split}.jsonl")
    for texts in perturbed.values():
        rng.shuffle(texts)
    pool = LabelPool(read_original_pool(source_dir / f"{split}.jsonl"), rng)

    pert_want, clean_want, blind_want = plan_counts(
        perturbed, pool, clean_ratio, blind_ratio, balance
    )

    rows = [
        {"text": text, "label": label, "source": "perturbed_normalized"}
        for label in LABELS
        for text in perturbed[label][: pert_want[label]]
    ]
    rows += [
        {"text": text, "label": label, "source": "clean_original"}
        for label in LABELS
        for text in pool.take(label, clean_want[label])
    ]
    blind_rows, failed = make_blindspot_rows(pool, blind_want, rng)
    rows += blind_rows
    rng.shuffle(rows)

    by_source_label = Counter((r["source"], r["label"]) for r in rows)
    stats = {
        "total": len(rows),
        "balanced": balance,
        "blindspot_generation_failed": failed,
        "by_source": {
            source: sum(v for (s, _), v in by_source_label.items() if s == source)
            for source in ("perturbed_normalized", "clean_original", "blindspot_raw")
        },
        "by_source_label": {f"{s}|{l}": v for (s, l), v in sorted(by_source_label.items())},
        "by_label": {label: sum(1 for r in rows if r["label"] == label) for label in LABELS},
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bộ dữ liệu trộn 60/30/10")
    parser.add_argument("--normalized_dir", type=str, default="data/processed/normalized")
    parser.add_argument("--source_dir", type=str, default="data/processed/vietnamese_nonstandard_v1_2")
    parser.add_argument("--out_dir", type=str, default="data/data_final")
    parser.add_argument("--clean_ratio", type=float, default=0.30)
    parser.add_argument("--blind_ratio", type=float, default=0.10)
    parser.add_argument("--balance_eval", action="store_true",
                         help="ép validation/test về 50/50 HATE. MẶC ĐỊNH TẮT — cân bằng sẽ "
                              "thổi phồng precision và làm mất khả năng đo đúng điểm yếu của "
                              "blacklist. Bộ dữ liệu thật của Bước 3 dựng với cờ này TẮT.")
    parser.add_argument("--no_balance_eval", action="store_true",
                         help="[phế] giữ lại để lệnh cũ không gãy; giờ đã là mặc định")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                         choices=list(SPLITS),
                         help="split nào cần dựng. Mặc định bỏ 'test': tập test của Bước 3 là "
                              "ma trận 3 điều kiện do build_eval_matrix.py dựng, không phải bản trộn.")
    parser.add_argument("--eval_suffix", type=str, default="_mixed",
                         help="hậu tố cho file validation/test, để phân biệt với "
                              "data_final/validation.jsonl (C0 câu sạch)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()

    set_global_seed(args.seed)
    if args.no_balance_eval:
        print("  [!] --no_balance_eval đã là mặc định, cờ này không còn tác dụng")

    normalized_dir = Path(args.normalized_dir)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "ratios": {
            "perturbed_normalized": round(1.0 - args.clean_ratio - args.blind_ratio, 4),
            "clean_original": args.clean_ratio,
            "blindspot_raw": args.blind_ratio,
        },
        "seed": args.seed,
        "blindspot_version": BLINDSPOT_VERSION,
        "balanced_eval": args.balance_eval,
        "splits": {},
    }

    for split in args.splits:
        balance = (split != "train") and args.balance_eval
        rows, stats = build_split(
            split, normalized_dir, source_dir, args.seed,
            args.clean_ratio, args.blind_ratio, balance,
        )
        filename = f"{split}.jsonl" if split == "train" else f"{split}{args.eval_suffix}.jsonl"
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = stats
        pct = {k: f"{v / stats['total']:.0%}" for k, v in stats["by_source"].items()}
        print(f"{split:11s} total={stats['total']:6d} {pct} labels={stats['by_label']}")
        for key in sorted(stats["by_source_label"]):
            print(f"              {key:32s} {stats['by_source_label'][key]}")

    with open(out_dir / "manifest_mix.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {out_dir}/ ({', '.join(args.splits)} + manifest_mix.json)")


if __name__ == "__main__":
    main()
