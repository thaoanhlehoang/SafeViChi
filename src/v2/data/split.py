"""Ghép tập train v2 (bước 6 trong docs/v2/KE_HOACH.md, phần train; validation/test làm sau khi người gán kho test).

Nguồn:
- Câu thật đã LLM gán lại và qua quy tắc đồng thuận: data/v2/interim/that_gan_lai.jsonl, chỉ lấy dong_thuan=true.
- Đoạn LLM sinh đã lọc lại: data/v2/interim/sinh_llm_loc.jsonl.

Số lượng mỗi nhãn (phương án ~15.000 dòng, chốt 20/11): 3 nhãn hiếm lấy hết; AN_TOAN và XUC_PHAM lấy mẫu xuống.
- AN_TOAN: SO_AN_TOAN_THAT câu thật (ưu tiên ca khó: câu CLEAN có từ trong blacklist v1, câu người gán gốc bất đồng),
  cộng SO_AN_TOAN_SINH đoạn sinh đôi (chia đều theo kịch bản).
- XUC_PHAM: mọi đoạn sinh (hội thoại nhiều dòng) + câu thật cho đủ SO_XUC_PHAM.
Mọi lần lấy mẫu dùng seed 42.

Kiểm tra trước khi ghi:
- Khử trùng gần đúng (src/dataset_builder/dedup.py) trên train + kho test: câu train nào cùng nhóm với một câu kho
  test bị bỏ (không để câu gần trùng giữa train và test). Câu train cùng nhóm với nhau chỉ giữ một câu.
- Không id nào thuộc kho test; mọi mẫu đúng định dạng v2 (CLAUDE.md), bằng chứng nằm nguyên văn trong một dòng.

Đầu ra:
- data/v2/so_sanh.jsonl: bản đủ trường (định dạng v2 + `meta`), dùng để phân tích theo nhóm, ghép với kết quả theo `id`.
- data/v2/train.jsonl: cùng các dòng, cùng thứ tự, chỉ giữ TRAIN_FIELDS (id, text, nhan, bang_chung) để đưa vào train.
- data/v2/train_stats.json, data/v2/CHECKSUMS.json (checksum cả hai file).

Chạy:
    python -m src.v2.data.split
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from v1.src.dataset_builder.dedup import group_near_duplicates

ROOT = Path(__file__).resolve().parents[3]
DATA_V2 = ROOT / "data" / "v2"
INTERIM = DATA_V2 / "interim"
THAT_PATH = INTERIM / "that_gan_lai.jsonl"
SINH_PATH = INTERIM / "sinh_llm_loc.jsonl"
KHO_TEST_PATH = INTERIM / "kho_test.jsonl"
OUT_PATH = DATA_V2 / "train.jsonl"
FULL_PATH = DATA_V2 / "so_sanh.jsonl"
SEED = 42

SO_AN_TOAN_THAT = 4000
SO_AN_TOAN_SINH = 2000
SO_XUC_PHAM = 3000

NHAN = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
FIELDS = ("id", "text", "nhan", "bang_chung", "nguon", "nguon_id", "cach_gan", "the", "so_dong", "bien_dang",
          "nhom_trung", "split")
TRAIN_FIELDS = ("id", "text", "nhan", "bang_chung")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_v2(record: dict, meta: dict) -> dict:
    """Bản ghi đúng định dạng v2 (CLAUDE.md) + trường phụ `meta` để phân tích theo nhóm sau này."""
    out = {k: record.get(k) for k in FIELDS}
    out.update({"bien_dang": "C0", "split": "train", "the": record.get("the") or [],
                "bang_chung": [] if record["nhan"] == "AN_TOAN" else (record.get("bang_chung") or []),
                "so_dong": record["text"].count("\n") + 1, "meta": meta})
    return out


def sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    return rows if len(rows) <= n else rng.sample(rows, n)


def sample_by_key(rows: list[dict], n: int, key: str, rng: random.Random) -> list[dict]:
    """Lấy n dòng chia đều theo giá trị của `key` (vd. kịch bản), phần dư lấy ngẫu nhiên."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key) or ""].append(r)
    per = n // max(1, len(groups))
    chosen = [r for k in sorted(groups) for r in sample(groups[k], per, rng)]
    rest = [r for r in rows if r not in chosen] if len(chosen) < n else []
    return chosen + sample(rest, n - len(chosen), rng)


def chon_that(that: list[dict], rng: random.Random) -> list[dict]:
    giu = [r for r in that if r.get("dong_thuan")]
    out: list[dict] = []
    an_toan = [r for r in giu if r["nhan"] == "AN_TOAN"]
    kho = [r for r in an_toan if r.get("ly_do_chon") == "clean_co_tu_blacklist"
           or r.get("ly_do_dong_thuan") == "khop_it_nhat_mot_nguoi_gan"]
    de = [r for r in an_toan if r not in kho]
    out += sample(kho, SO_AN_TOAN_THAT, rng)
    out += sample(de, SO_AN_TOAN_THAT - min(len(kho), SO_AN_TOAN_THAT), rng)
    out += [r for r in giu if r["nhan"] in ("QUAY_ROI", "DE_DOA", "LUA_DAO")]
    return out


def build(that: list[dict], sinh: list[dict], kho_test: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    picked = []
    for r in chon_that(that, rng):
        picked.append(to_v2(r, {"loai": "that", "nhan_goc": r.get("nhan_goc"), "ly_do_chon": r.get("ly_do_chon"),
                                "ly_do_dong_thuan": r.get("ly_do_dong_thuan")}))
    sinh_xp = [r for r in sinh if r["nhan"] == "XUC_PHAM"]
    sinh_at = [r for r in sinh if r["nhan"] == "AN_TOAN"]
    sinh_hiem = [r for r in sinh if r["nhan"] in ("QUAY_ROI", "DE_DOA", "LUA_DAO")]
    for r in sinh_hiem + sinh_xp + sample_by_key(sinh_at, SO_AN_TOAN_SINH, "kich_ban", rng):
        picked.append(to_v2(r, {"loai": "sinh", "kich_ban": r.get("kich_ban"), "kenh": r.get("kenh"),
                                "giong": r.get("giong"), "do_lo": r.get("do_lo"), "kieu_lo": r.get("kieu_lo"),
                                "sinh_doi": r.get("sinh_doi")}))
    # XUC_PHAM thật: bù cho đủ SO_XUC_PHAM sau khi đã lấy hết đoạn sinh
    xp_that = [r for r in that if r.get("dong_thuan") and r["nhan"] == "XUC_PHAM"]
    for r in sample(xp_that, max(0, SO_XUC_PHAM - len(sinh_xp)), rng):
        picked.append(to_v2(r, {"loai": "that", "nhan_goc": r.get("nhan_goc"), "ly_do_chon": r.get("ly_do_chon"),
                                "ly_do_dong_thuan": r.get("ly_do_dong_thuan")}))

    # Khử trùng gần đúng train + kho test
    texts = [r["text"] for r in picked] + [r["text"] for r in kho_test]
    groups = list(group_near_duplicates([SimpleNamespace(text=t) for t in texts]).group_ids)
    test_groups = set(groups[len(picked):])
    test_ids = {r["id"] for r in kho_test}
    dropped = Counter()
    seen_groups: set[str] = set()
    train = []
    for r, g in zip(picked, groups[: len(picked)]):
        if r["id"] in test_ids or g in test_groups:
            dropped["trung_kho_test"] += 1
            continue
        if g in seen_groups:
            dropped["trung_trong_train"] += 1
            continue
        seen_groups.add(g)
        train.append({**r, "nhom_trung": g})
    rng.shuffle(train)

    stats = {
        "tong": len(train),
        "theo_nhan": {k: v for k, v in Counter(r["nhan"] for r in train).most_common()},
        "theo_nhan_va_loai": {f"{n}|{l}": c for (n, l), c in sorted(
            Counter((r["nhan"], r["meta"]["loai"]) for r in train).items())},
        "theo_so_dong": dict(sorted(Counter(min(r["so_dong"], 8) for r in train).items())),
        "nhieu_dong_theo_nhan": {n: sum(1 for r in train if r["nhan"] == n and r["so_dong"] > 1) for n in NHAN},
        "bi_bo_khi_khu_trung": dict(dropped),
        "seed": seed,
        "cau_hinh": {"SO_AN_TOAN_THAT": SO_AN_TOAN_THAT, "SO_AN_TOAN_SINH": SO_AN_TOAN_SINH,
                     "SO_XUC_PHAM": SO_XUC_PHAM},
    }
    return train, stats


def validate(train: list[dict]) -> list[str]:
    errors = []
    ids = Counter(r["id"] for r in train)
    errors += [f"trùng id {i}" for i, c in ids.items() if c > 1]
    for r in train:
        if r["nhan"] not in NHAN:
            errors.append(f"{r['id']}: nhãn lạ {r['nhan']!r}")
        if not r["text"].strip() or r["so_dong"] > 8:
            errors.append(f"{r['id']}: text rỗng hoặc quá 8 dòng")
        lines = r["text"].split("\n")
        for phrase in r["bang_chung"]:
            if not any(phrase in line for line in lines):
                errors.append(f"{r['id']}: bằng chứng không nằm gọn trong một dòng: {phrase!r}")
    return errors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slim(record: dict) -> dict:
    """Chỉ giữ các trường cần để train (TRAIN_FIELDS)."""
    return {k: record[k] for k in TRAIN_FIELDS}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="file train rút gọn (TRAIN_FIELDS)")
    parser.add_argument("--full", type=Path, default=FULL_PATH, help="file đủ trường để phân tích")
    args = parser.parse_args(argv)

    train, stats = build(read_jsonl(THAT_PATH), read_jsonl(SINH_PATH), read_jsonl(KHO_TEST_PATH))
    errors = validate(train)
    if errors:
        for e in errors[:20]:
            print("LỖI:", e)
        raise SystemExit(f"{len(errors)} lỗi, không ghi file.")
    write_jsonl(args.full, train)
    write_jsonl(args.out, [slim(r) for r in train])
    stats["nguon"] = {p.name: sha256(p) for p in (THAT_PATH, SINH_PATH, KHO_TEST_PATH)}
    stats_path = args.out.with_name("train_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksums = DATA_V2 / "CHECKSUMS.json"
    current = json.loads(checksums.read_text(encoding="utf-8")) if checksums.exists() else {}
    for path in (args.full, args.out):
        current[path.name] = sha256(path)
    checksums.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "nguon"}, ensure_ascii=False, indent=2))
    print(f"Đã ghi {args.full} và {args.out} ({len(train)} dòng), {stats_path}, {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
