"""Lọc lại đoạn hội thoại do LLM sinh: cho LLM gán nhãn lại từng đoạn theo hướng dẫn (như prelabel.py, không biết
nhãn yêu cầu), chỉ giữ đoạn mà nhãn gán lại TRÙNG nhãn yêu cầu khi sinh.

Mục đích: loại đoạn sinh lệch kịch bản, ví dụ ô DE_DOA mà lời dọa quá nhẹ (thật ra là AN_TOAN), hoặc ô sinh đôi
AN_TOAN lỡ có nội dung nguy hại. Đoạn bị loại vẫn được ghi lại kèm lý do để xem và báo cáo tỉ lệ loại.

Đầu vào: data/v2/interim/sinh_llm.jsonl (đầu ra của generate.py).
Đầu ra:
    data/v2/interim/sinh_llm_loc.jsonl        đoạn giữ lại (thêm trường kiem_lai_*)
    data/v2/interim/sinh_llm_bi_loai.jsonl    đoạn bị loại, kèm nhãn gán lại và lý do
    data/v2/interim/sinh_llm_loc_stats.json   tỉ lệ giữ theo nhãn và kịch bản
Cache gán lại nằm ở data/v2/interim/prelabel/cache_loc_*.jsonl (không commit); chạy lại sẽ tiếp tục chỗ dừng.

Chạy:
    python -m src.v2.labeling.loc_sinh --dry-run
    python -m src.v2.labeling.loc_sinh --batch-size 20 --workers 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.v2.labeling import llm
from src.v2.labeling import prelabel as p

ROOT = Path(__file__).resolve().parents[3]
INTERIM = ROOT / "data" / "v2" / "interim"
IN_PATH = INTERIM / "sinh_llm.jsonl"
CACHE_DIR = INTERIM / "prelabel"


def split_by_check(records: list[dict], checked: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Tách (giữ, loại). `checked` là kết quả prelabel.finalize theo id. Giữ khi nhãn gán lại trùng nhãn sinh."""
    kept, dropped = [], []
    for record in records:
        check = checked.get(record["id"])
        out = dict(record)
        if check is None or check.get("llm_loi"):
            out["ly_do_loai"] = "chua_kiem_duoc"
            dropped.append(out)
            continue
        out.update({
            "kiem_lai_nhan": check["nhan"],
            "kiem_lai_do_chac": check.get("llm_do_chac"),
            "kiem_lai_ly_do": check.get("llm_ly_do"),
        })
        if check["nhan"] == record["nhan"]:
            kept.append(out)
        else:
            out["ly_do_loai"] = f"kiem_lai_ra_{check['nhan']}"
            dropped.append(out)
    return kept, dropped


def summarize(kept: list[dict], dropped: list[dict]) -> dict:
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_scenario: dict[str, Counter] = defaultdict(Counter)
    for status, rows in (("giu", kept), ("loai", dropped)):
        for r in rows:
            label = r["nhan"] + ("(sinh_doi)" if r.get("sinh_doi") else "")
            by_label[label][status] += 1
            by_scenario[r.get("kich_ban", "?")][status] += 1

    def rate(c: Counter) -> dict:
        total = c["giu"] + c["loai"]
        return {"giu": c["giu"], "loai": c["loai"], "ti_le_giu": round(c["giu"] / total, 3) if total else None}

    return {
        "tong": len(kept) + len(dropped), "giu": len(kept), "loai": len(dropped),
        "ly_do_loai": dict(Counter(r["ly_do_loai"] for r in dropped).most_common()),
        "theo_nhan": {k: rate(v) for k, v in sorted(by_label.items())},
        "theo_kich_ban": {k: rate(v) for k, v in sorted(by_scenario.items())},
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--out-dir", type=Path, default=INTERIM)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="chỉ đếm và in prompt lô đầu, không gọi API")
    args = parser.parse_args(argv)

    records = list(p.read_jsonl(args.input))
    system = p.build_system_prompt()
    version = p.prompt_version(system)
    # Chỉ gửi id và text: người kiểm không được biết nhãn yêu cầu khi sinh.
    blind = [{"id": r["id"], "text": r["text"]} for r in records]
    print(f"{len(records)} đoạn sinh, {len({r['text'] for r in records})} text khác nhau, "
          f"khoảng {-(-len(blind) // args.batch_size)} lượt gọi. Theo nhãn: {dict(Counter(r['nhan'] for r in records))}")
    if args.dry_run:
        print(p.build_user_prompt([r["text"] for r in blind[: args.batch_size]]))
        print(f"(prompt_version={version}, không gọi API)")
        return 0

    llm.load_dotenv()
    config = llm.config_from_env()
    meta = {"provider": config.provider, "model": config.model, "prompt_version": version, **config.describe()}
    tag = llm.safe_name(f"{config.provider}_{config.model}_{version}")

    def complete(system_prompt: str, user_prompt: str) -> str:
        return llm.with_retries(lambda: llm.call_llm(config, system_prompt, user_prompt))

    results = p.run(blind, system=system, complete=complete, cache_path=CACHE_DIR / f"cache_loc_{tag}.jsonl",
                    meta=meta, batch_size=args.batch_size, workers=args.workers)
    kept, dropped = split_by_check(records, {r["id"]: r for r in results})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(kept, args.out_dir / "sinh_llm_loc.jsonl")
    write_jsonl(dropped, args.out_dir / "sinh_llm_bi_loai.jsonl")
    stats = {**summarize(kept, dropped), **meta, "dau_vao": str(args.input)}
    (args.out_dir / "sinh_llm_loc_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Token đã dùng lượt này: {llm.USAGE.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
