"""Chạy thử nhỏ một model LLM để so sánh giữa các model trên CÙNG một mẫu cố định (seed 42).

- Gán lại nhãn: N câu chọn ngẫu nhiên từ data/v2/interim/merged.jsonl.
- Sinh mới: mỗi nhãn nguy hại trong bảng kịch bản một ô chọn ngẫu nhiên, kèm ô sinh đôi AN_TOAN của nó;
  mỗi ô M đoạn.

Mẫu chọn chỉ phụ thuộc seed, không phụ thuộc model, nên chạy lại với model khác (đổi SAFEVICHI_LLM_MODEL)
sẽ cho cùng câu và cùng ô. Kết quả mỗi model một thư mục:
    results/v2/thu_model/<model>/prelabel.jsonl   nhãn LLM cho các câu (kèm nhãn gốc để đối chiếu)
    results/v2/thu_model/<model>/sinh.jsonl       đoạn hội thoại sinh ra (đã qua kiểm tra của generate.py)
    results/v2/thu_model/<model>/raw_*_<prompt_version>.jsonl   phản hồi thô (đổi prompt thì gọi lại)
    results/v2/thu_model/<model>/tom_tat.json     cấu hình, token đã dùng, thời gian, tỉ lệ lỗi
    results/v2/thu_model/<model>/env.txt          phiên bản Python, thư viện, lệnh chạy

Chạy (gọi API thật):
    python -m src.v2.labeling.thu_model --so-cau 5 --so-doan 5
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.v2.labeling import generate as g
from src.v2.labeling import llm
from src.v2.labeling import prelabel as p

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "results" / "v2" / "thu_model"
SEED = 42


def pick_records(records: list[dict], n: int, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(records, n)


def pick_cells(bang: g.BangKichBan, so_doan: int, seed: int = SEED) -> list[g.O]:
    """Mỗi nhãn nguy hại một ô ngẫu nhiên (thứ tự nhãn cố định), ngay sau là ô sinh đôi AN_TOAN của nó."""
    rng = random.Random(seed)
    cells = g.plan_cells(bang, None, so_doan, co_sinh_doi=True)
    twins = {c.o_id: c for c in cells if c.sinh_doi}
    chosen: list[g.O] = []
    for nhan in sorted({c.nhan for c in cells if not c.sinh_doi}):
        cell = rng.choice([c for c in cells if not c.sinh_doi and c.nhan == nhan])
        chosen.append(replace(cell, so_doan=so_doan))
        twin = twins.get(f"{cell.o_id}-SD")
        if twin is not None:
            chosen.append(replace(twin, so_doan=so_doan))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-cau", type=int, default=5)
    parser.add_argument("--so-doan", type=int, default=5)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args(argv)

    llm.load_dotenv()
    config = llm.config_from_env()
    out_dir = args.out_root / llm.safe_name(config.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chỉ chọn trong danh sách cần gán (kho train), không bao giờ chạm kho test.
    records = pick_records(list(p.read_jsonl(p.IN_PATH)), args.so_cau)
    bang = g.load_kich_ban()
    cells = pick_cells(bang, args.so_doan)

    def complete(system_prompt: str, user_prompt: str) -> str:
        return llm.with_retries(lambda: llm.call_llm(config, system_prompt, user_prompt))

    # Gán lại nhãn
    system_p = p.build_system_prompt()
    meta_p = {"provider": config.provider, "model": config.model,
              "prompt_version": p.prompt_version(system_p), **config.describe()}
    t0 = time.perf_counter()
    labeled = p.run(records, system=system_p, complete=complete, cache_path=out_dir / f"raw_prelabel_{meta_p['prompt_version']}.jsonl",
                    meta=meta_p, batch_size=args.so_cau)
    t_label = time.perf_counter() - t0
    usage_label = llm.USAGE.summary()
    with (out_dir / "prelabel.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for r in labeled:
            keep = {k: r.get(k) for k in ("id", "text", "nhan", "bang_chung", "the", "llm_do_chac", "llm_ly_do",
                                         "can_duyet", "ly_do_duyet", "llm_loi", "llm_bang_chung_bi_bo", "nhan_goc", "nhan_goc_bat_dong")}
            keep["nhan_goc_theo_nguon"] = {o["nguon"]: o["nhan_goc"] for o in r.get("nguon_goc", [])}
            handle.write(json.dumps(keep, ensure_ascii=False) + "\n")

    # Sinh mới
    system_g = g.build_system_prompt()
    meta_g = {"provider": config.provider, "model": config.model,
              "prompt_version": p.prompt_version(system_g), **config.describe()}
    gen_config = replace(config, temperature=1.0)

    def complete_gen(system_prompt: str, user_prompt: str) -> str:
        return llm.with_retries(lambda: llm.call_llm(gen_config, system_prompt, user_prompt))

    t0 = time.perf_counter()
    done = g.run_cells(cells, bang, system=system_g, complete=complete_gen,
                       raw_path=out_dir / f"raw_sinh_{meta_g['prompt_version']}.jsonl", meta=meta_g, workers=len(cells))
    t_gen = time.perf_counter() - t0
    sinh, stats = g.collect(cells, done, meta_g)
    with (out_dir / "sinh.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for r in sinh:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": config.model, "seed": SEED, **config.describe(),
        "gan_nhan": {"so_cau": len(records), "ids": [r["id"] for r in records],
                     "so_loi": sum(1 for r in labeled if r.get("llm_loi")), "giay": round(t_label, 1),
                     "token": usage_label},
        "sinh": {"o": [c.o_id for c in cells], "so_doan_moi_o": args.so_doan,
                 "so_doan_hop_le": len(sinh), "giay": round(t_gen, 1), "thong_ke": stats},
        "token_tong": llm.USAGE.summary(),
        "token_chi_tiet": {"calls": llm.USAGE.calls, "prompt": llm.USAGE.prompt,
                           "completion": llm.USAGE.completion, "reasoning": llm.USAGE.reasoning},
    }
    (out_dir / "tom_tat.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        import requests
        req_version = requests.__version__
    except ImportError:
        req_version = "?"
    (out_dir / "env.txt").write_text(
        f"lenh: python -m src.v2.labeling.thu_model {' '.join(argv if argv is not None else sys.argv[1:])}\n"
        f"python: {sys.version.split()[0]}\nplatform: {platform.platform()}\nrequests: {req_version}\n"
        f"thoi_gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã ghi kết quả vào {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
