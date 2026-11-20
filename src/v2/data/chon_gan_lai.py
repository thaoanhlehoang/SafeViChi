"""Tách kho test và chọn câu thật cần LLM gán lại cho train (chạy trên máy, không gọi API).

1. Khử trùng gần đúng toàn bộ data/v2/interim/merged.jsonl bằng src/dataset_builder/dedup.py; mỗi bản ghi có
   `nhom_trung`. Cả nhóm luôn đi cùng một phía.
2. KHO TEST: các nhóm mà KHÔNG câu nào từng nằm trong split train của ViHSD hoặc ViHOS. ViHateT5 (nền của v1)
   đã học split train của hai bộ này, nên test chỉ lấy câu ngoài đó để so v1 công bằng. Kho test KHÔNG được đưa
   cho LLM gán và không bao giờ vào train; câu test/validation chọn từ đây do người gán (xem CLAUDE.md).
   Nhóm có câu đã từng gửi cho LLM (cache gán nhãn, lượt chạy thử) cũng bị đẩy về train.
3. KHO TRAIN (phần còn lại), chọn câu cần LLM gán lại:
   - mọi câu nhãn gốc OFFENSIVE/HATE (nguồn XUC_PHAM, nơi ẩn DE_DOA/QUAY_ROI thật);
   - mọi ứng viên từ khóa trong ung_vien.jsonl (nghi QUAY_ROI/DE_DOA/LUA_DAO);
   - mọi câu CLEAN chứa từ trong blacklist v1 (có từ tục/lóng nhưng vô hại: ca khó cho yêu cầu 1);
   - ngẫu nhiên N câu CLEAN còn lại (seed 42).

Đầu ra:
    data/v2/interim/kho_test.jsonl          kho test (giữ riêng)
    data/v2/interim/can_gan_lai.jsonl       câu cần LLM gán lại, có `ly_do_chon`
    data/v2/interim/chon_gan_lai_stats.json thống kê

Chạy:
    python -m src.v2.data.chon_gan_lai
    python -m src.v2.data.chon_gan_lai --so-clean-ngau-nhien 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from src.baselines.blacklist import predict as blacklist_predict
from src.dataset_builder.dedup import group_near_duplicates

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "v2" / "interim"
MERGED_PATH = INTERIM / "merged.jsonl"
UNG_VIEN_PATH = INTERIM / "ung_vien.jsonl"
BLACKLIST_PATH = ROOT / "src" / "baselines" / "blacklist_words.json"
KHO_TEST_PATH = INTERIM / "kho_test.jsonl"
OUT_PATH = INTERIM / "can_gan_lai.jsonl"
SEED = 42
# Split train của các bộ mà ViHateT5-base-HSD đã học.
NGUON_V1_DA_HOC = {"vihsd", "vihos"}


def v1_da_thay(record: dict) -> bool:
    return any(o["nguon"] in NGUON_V1_DA_HOC and o["split_goc"] == "train" for o in record.get("nguon_goc", []))


def ids_da_gui_llm(paths: list[Path]) -> set[str]:
    """id mọi câu đã từng gửi cho LLM gán nhãn (cache prelabel, kết quả chạy thử)."""
    ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def tach_kho(records: list[dict], group_ids: list[str],
             da_gui_llm: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """(kho_train, kho_test) theo nhóm trùng: nhóm có ít nhất một câu v1 đã thấy, hoặc một câu đã gửi LLM, thì
    cả nhóm về train."""
    da_gui_llm = da_gui_llm or set()
    nhom_train = {g for r, g in zip(records, group_ids) if v1_da_thay(r) or r["id"] in da_gui_llm}
    train, test = [], []
    for record, group in zip(records, group_ids):
        record = {**record, "nhom_trung": group}
        (train if group in nhom_train else test).append(record)
    return train, test


def chon(train: list[dict], ung_vien_ids: set[str], blacklist: set[str], so_clean: int,
         seed: int = SEED) -> list[dict]:
    """Câu cần gán lại trong kho train, mỗi câu kèm ly_do_chon (lý do đầu tiên khớp)."""
    chosen: list[dict] = []
    rest_clean: list[dict] = []
    for record in train:
        if record["nhan_goc"] != "CLEAN":
            reason = "nhan_goc_nguy_hai"
        elif record["id"] in ung_vien_ids:
            reason = "ung_vien_tu_khoa"
        elif blacklist_predict(record["text"], blacklist) != "CLEAN":
            reason = "clean_co_tu_blacklist"
        else:
            rest_clean.append(record)
            continue
        chosen.append({**record, "ly_do_chon": reason})
    rng = random.Random(seed)
    sample = rng.sample(rest_clean, min(so_clean, len(rest_clean)))
    chosen += [{**r, "ly_do_chon": "clean_ngau_nhien"} for r in sample]
    return chosen


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-clean-ngau-nhien", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--kho-test", type=Path, default=KHO_TEST_PATH)
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in MERGED_PATH.open(encoding="utf-8") if line.strip()]
    ung_vien_ids = {json.loads(line)["id"] for line in UNG_VIEN_PATH.open(encoding="utf-8") if line.strip()}
    blacklist = set(json.loads(BLACKLIST_PATH.read_text(encoding="utf-8")))
    print(f"Khử trùng gần đúng {len(records)} câu...", file=sys.stderr)
    group_ids = list(group_near_duplicates([SimpleNamespace(text=r["text"]) for r in records]).group_ids)

    llm_files = sorted(INTERIM.glob("prelabel/*.jsonl")) + sorted((ROOT / "results" / "v2" / "thu_model").glob("*/*.jsonl"))
    da_gui_llm = ids_da_gui_llm(llm_files)
    train, test = tach_kho(records, group_ids, da_gui_llm)
    chosen = chon(train, ung_vien_ids, blacklist, args.so_clean_ngau_nhien)
    write_jsonl(test, args.kho_test)
    write_jsonl(chosen, args.out)

    sizes = Counter(group_ids)
    by_reason: dict[str, Counter] = defaultdict(Counter)
    for r in chosen:
        by_reason[r["ly_do_chon"]][r["nhan_goc"]] += 1
    stats = {
        "tong_cau": len(records),
        "so_nhom_trung": len(sizes),
        "so_nhom_nhieu_cau": sum(1 for n in sizes.values() if n > 1),
        "so_cau_da_gui_llm_truoc_do": len(da_gui_llm & {r["id"] for r in records}),
        "kho_test": {"so_cau": len(test), "nhan_goc": dict(Counter(r["nhan_goc"] for r in test)),
                     "ung_vien_tu_khoa": sum(r["id"] in ung_vien_ids for r in test)},
        "kho_train": {"so_cau": len(train), "nhan_goc": dict(Counter(r["nhan_goc"] for r in train))},
        "can_gan_lai": {"so_cau": len(chosen), "theo_ly_do": {k: dict(v) for k, v in by_reason.items()}},
        "so_clean_ngau_nhien": args.so_clean_ngau_nhien, "seed": SEED,
    }
    # Kiểm tra bất biến: không câu nào vừa ở kho test vừa được chọn, không nhóm nào nằm hai phía.
    test_groups = {r["nhom_trung"] for r in test}
    assert not test_groups & {r["nhom_trung"] for r in train}
    assert not {r["id"] for r in test} & {r["id"] for r in chosen}
    stats_path = args.out.with_name("chon_gan_lai_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Đã ghi {args.out} ({len(chosen)} câu) và {args.kho_test} ({len(test)} câu, KHÔNG gửi LLM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
