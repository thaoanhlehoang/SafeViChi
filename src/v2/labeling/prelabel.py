"""LLM gán sơ bộ 5 nhãn v2 theo docs/v2/HUONG_DAN_GAN_NHAN.md (mục 2–5 + prompt ở phụ lục).

Cấu hình qua biến môi trường (hoặc file .env ở gốc repo, không commit):
    SAFEVICHI_LLM_PROVIDER   anthropic | openai | gemini   (bắt buộc khi gọi thật, không có mặc định)
    SAFEVICHI_LLM_MODEL      tên model của nhà cung cấp     (bắt buộc khi gọi thật)
    SAFEVICHI_LLM_BASE_URL   tùy chọn, cho API tương thích OpenAI (OpenRouter, DeepSeek, ...)
    ANTHROPIC_API_KEY | OPENAI_API_KEY | GEMINI_API_KEY   khóa tương ứng
    (hoặc SAFEVICHI_LLM_API_KEY dùng chung cho mọi nhà cung cấp)

Chạy:
    python -m src.v2.labeling.prelabel --dry-run --limit 3
    python -m src.v2.labeling.prelabel --limit 20
Kết quả: cache theo id (ghi nối, chạy lại sẽ tiếp tục chỗ dừng) và file gán sơ bộ đầy đủ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.v2.labeling.llm import (  # noqa: F401 - giữ tên cũ cho người gọi
    KEY_ENV, USAGE, LLMConfig, call_llm, config_from_env, load_dotenv, safe_name, with_retries,
)

ROOT = Path(__file__).resolve().parents[3]
GUIDE_PATH = ROOT / "docs" / "v2" / "HUONG_DAN_GAN_NHAN.md"
IN_PATH = ROOT / "data" / "v2" / "interim" / "can_gan_lai.jsonl"  # tạo bằng python -m src.v2.data.chon_gan_lai
KHO_TEST_PATH = ROOT / "data" / "v2" / "interim" / "kho_test.jsonl"


def load_kho_test_ids(path: Path = KHO_TEST_PATH) -> set[str]:
    """id câu thuộc kho test: KHÔNG bao giờ gửi cho LLM (test do người gán)."""
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()}
OUT_DIR = ROOT / "data" / "v2" / "interim" / "prelabel"

LABELS = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
TAGS = ("tieng_long", "vung_mien", "an_y", "trich_dan")
CONFIDENCE_THRESHOLD = 0.7
NHAN_GOC_SACH = {"CLEAN"}
NHAN_GOC_NGUY_HAI = {"OFFENSIVE", "HATE"}

BATCH_INSTRUCTION = (
    "Bạn sẽ nhận nhiều mẫu, mỗi mẫu mở đầu bằng [số thứ tự] và được ghi dạng chuỗi JSON. Một mẫu có thể là một "
    "tin nhắn hoặc một đoạn hội thoại hai người nhiều dòng (ngăn bằng \\n, mỗi dòng là toàn bộ lượt nói của một "
    "người, các dòng luân phiên hai người, không ghi người nói); "
    "khi đó gán nhãn cho cả đoạn theo tình huống nặng nhất, mỗi cụm bằng chứng nằm gọn trong một dòng. "
    "Gán từng mẫu độc lập với nhau. Trả về đúng một dòng JSON cho mỗi mẫu, theo đúng thứ tự, và thêm trường "
    "\"stt\" là số thứ tự của mẫu đó."
)


# ---------------------------------------------------------------- prompt

def load_guide_sections(path: Path = GUIDE_PATH) -> tuple[str, str]:
    """Trả về (mục 2–5 của hướng dẫn, prompt trong phụ lục)."""
    text = path.read_text(encoding="utf-8")
    start = text.index("## 2.")
    end = text.index("## 6.")
    sections = text[start:end].strip()
    appendix = text[text.index("## Phụ lục"):]
    match = re.search(r"```\n(.*?)```", appendix, re.S)
    if not match:
        raise ValueError("Không tìm thấy khối prompt trong phụ lục của hướng dẫn")
    return sections, match.group(1).strip()


def build_system_prompt(path: Path = GUIDE_PATH) -> str:
    sections, appendix_prompt = load_guide_sections(path)
    return f"{sections}\n\n{appendix_prompt}\n\n{BATCH_INSTRUCTION}"


def build_user_prompt(texts: list[str]) -> str:
    # json.dumps để xuống dòng/ngoặc kép trong tin nhắn không phá cấu trúc.
    return "\n".join(f"[{i}] {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(texts, 1))


def prompt_version(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- kiểm tra đầu ra

@dataclass
class Validated:
    nhan: str | None
    bang_chung: list[str]
    bang_chung_bi_bo: list[str]
    the: list[str]
    do_chac: float | None
    ly_do: str
    loi: list[str]


def validate_output(raw: dict, text: str) -> Validated:
    """Kiểm một kết quả LLM: nhãn hợp lệ, bằng chứng có nguyên văn trong text, thẻ và độ chắc hợp lệ."""
    loi: list[str] = []
    nhan = raw.get("nhan")
    if isinstance(nhan, str):
        nhan = nhan.strip().upper()
    if nhan not in LABELS:
        loi.append(f"nhan_khong_hop_le:{raw.get('nhan')!r}")
        nhan = None

    kept: list[str] = []
    dropped: list[str] = []
    evidence = raw.get("bang_chung") or []
    if not isinstance(evidence, list):
        loi.append("bang_chung_khong_phai_list")
        evidence = []
    for phrase in evidence:
        if not isinstance(phrase, str):
            dropped.append(str(phrase))
            continue
        phrase = unicodedata.normalize("NFC", phrase).strip()
        if not phrase:
            continue
        if "\n" not in phrase and phrase in text:  # mỗi cụm phải nằm gọn trong một dòng
            if phrase not in kept:
                kept.append(phrase)
        else:
            dropped.append(phrase)
    if nhan == "AN_TOAN" and kept:
        # Hướng dẫn: câu AN_TOAN để trống bằng chứng.
        dropped.extend(kept)
        kept = []

    tags_raw = raw.get("the") or []
    tags = [t for t in tags_raw if isinstance(t, str) and t in TAGS] if isinstance(tags_raw, list) else []

    do_chac = raw.get("do_chac")
    try:
        do_chac = float(do_chac)
        if not 0.0 <= do_chac <= 1.0:
            raise ValueError
    except (TypeError, ValueError):
        loi.append(f"do_chac_khong_hop_le:{raw.get('do_chac')!r}")
        do_chac = None

    ly_do = raw.get("ly_do")
    return Validated(nhan, kept, dropped, list(dict.fromkeys(tags)), do_chac,
                     ly_do if isinstance(ly_do, str) else "", loi)


def review_reasons(v: Validated, nhan_goc: str | None) -> list[str]:
    """Lý do đưa câu vào hàng chờ người duyệt (rỗng = không cần duyệt)."""
    reasons: list[str] = []
    if v.loi or v.nhan is None:
        reasons.append("loi_dau_ra")
    if v.do_chac is None or v.do_chac < CONFIDENCE_THRESHOLD:
        reasons.append("do_chac_thap")
    goc = (nhan_goc or "").upper()
    if v.nhan is not None:
        if goc in NHAN_GOC_SACH and v.nhan != "AN_TOAN":
            reasons.append("lech_nhan_goc_sach_thanh_nguy_hai")
        elif goc in NHAN_GOC_NGUY_HAI and v.nhan == "AN_TOAN":
            reasons.append("lech_nhan_goc_nguy_hai_thanh_an_toan")
    if v.bang_chung_bi_bo:
        reasons.append("co_bang_chung_bi_bo")
    return reasons


def parse_batch_response(content: str, n: int) -> dict[int, dict]:
    """Tách các dòng JSON trong phản hồi, trả về {stt: obj}. Dòng hỏng bị bỏ qua."""
    content = re.sub(r"```(?:json)?", "", content)
    results: dict[int, dict] = {}
    order = 0
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        order += 1
        stt = obj.get("stt", order if n == 1 else None)
        try:
            stt = int(stt)
        except (TypeError, ValueError):
            continue
        if 1 <= stt <= n and stt not in results:
            results[stt] = obj
    return results


# ---------------------------------------------------------------- cache và chạy theo lô

def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    """Cache dạng JSONL ghi nối, khóa theo id. Dòng cuối bị cắt dở (do ngắt giữa chừng) bị bỏ qua."""
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[entry["id"]] = entry
    return cache


def label_batch(
    batch: list[dict],
    system: str,
    complete: Callable[[str, str], str],
) -> dict[str, dict]:
    """Gán một lô; tin nào lô không trả đủ thì gọi lại từng tin một. Trả về {id: raw_obj | None}."""
    texts = [r["text"] for r in batch]
    raw = complete(system, build_user_prompt(texts))
    parsed = parse_batch_response(raw, len(batch))
    out: dict[str, dict] = {}
    for i, record in enumerate(batch, 1):
        obj = parsed.get(i)
        if obj is None and len(batch) > 1:
            single = parse_batch_response(complete(system, build_user_prompt([record["text"]])), 1)
            obj = single.get(1)
        out[record["id"]] = obj
    return out


def make_cache_entry(record: dict, obj: dict | None, meta: dict) -> dict:
    return {"id": record["id"], "text_sha256": text_key(record["text"]), "llm_raw": obj, **meta}


def finalize(record: dict, entry: dict) -> dict:
    """Gộp bản ghi gốc với kết quả LLM đã kiểm, theo định dạng v2."""
    obj = entry.get("llm_raw")
    if obj is None:
        v = Validated(None, [], [], [], None, "", ["khong_co_ket_qua"])
    else:
        v = validate_output(obj, record["text"])
    reasons = review_reasons(v, record.get("nhan_goc"))
    result = dict(record)
    result.update({
        "nhan": v.nhan or "",
        "bang_chung": v.bang_chung,
        "the": v.the,
        "cach_gan": "llm",
        "llm_do_chac": v.do_chac,
        "llm_ly_do": v.ly_do,
        "llm_bang_chung_bi_bo": v.bang_chung_bi_bo,
        "llm_loi": v.loi,
        "llm_provider": entry.get("provider"),
        "llm_model": entry.get("model"),
        "llm_prompt_version": entry.get("prompt_version"),
        "can_duyet": bool(reasons),
        "ly_do_duyet": reasons,
    })
    return result


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run(
    records: list[dict],
    *,
    system: str,
    complete: Callable[[str, str], str],
    cache_path: Path,
    meta: dict,
    batch_size: int = 10,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Gán mọi bản ghi chưa có trong cache; text trùng chỉ gọi API một lần. Trả về bản ghi đã gộp."""
    cache = load_cache(cache_path)
    by_text = {e["text_sha256"]: e for e in cache.values() if e.get("llm_raw") is not None}
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    todo: list[dict] = []
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if record["id"] in cache and cache[record["id"]].get("llm_raw") is not None:
                continue  # đã gán xong; bản ghi lỗi (llm_raw rỗng) được thử lại khi chạy lại
            same = by_text.get(text_key(record["text"]))
            if same is not None:  # text đã gán ở id khác: chép lại, không tốn API
                entry = make_cache_entry(record, same["llm_raw"], {**meta, "chep_tu_id": same["id"]})
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache[record["id"]] = entry
                continue
            todo.append(record)

        # Gom text khác nhau để không gửi trùng trong cùng lượt.
        unique: dict[str, dict] = {}
        for record in todo:
            unique.setdefault(text_key(record["text"]), record)
        pending = list(unique.values())
        log(f"Cần gọi API cho {len(pending)} text khác nhau ({len(todo)} bản ghi), lô {batch_size}.")

        batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        done = 0

        def save(batch: list[dict], results: dict[str, dict]) -> None:
            # Chỉ luồng chính ghi file và cache.
            nonlocal done
            for record in batch:
                entry = make_cache_entry(record, results[record["id"]], meta)
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache[record["id"]] = entry
                if entry["llm_raw"] is not None:
                    by_text[entry["text_sha256"]] = entry
            handle.flush()
            os.fsync(handle.fileno())
            done += len(batch)
            log(f"  xong {done}/{len(pending)}")

        failed = 0

        def safe_label(batch: list[dict]) -> dict[str, dict] | None:
            # Lô lỗi (mạng, hết tiền, nhà cung cấp ngừng) chỉ bị bỏ qua lượt này; chạy lại sẽ gán tiếp.
            nonlocal failed
            try:
                return label_batch(batch, system, complete)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log(f"  LỖI lô {batch[0]['id']}… ({type(exc).__name__}: {str(exc)[:200]}), bỏ qua, chạy lại để thử tiếp")
                return None

        if workers <= 1:
            for batch in batches:
                results = safe_label(batch)
                if results is not None:
                    save(batch, results)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(safe_label, batch): batch for batch in batches}
                for future in as_completed(futures):
                    if future.result() is not None:
                        save(futures[future], future.result())
        if failed:
            log(f"CÓ {failed} lô lỗi chưa gán. Chạy lại cùng lệnh để gán tiếp (cache giữ phần đã xong).")

        for record in todo:
            if record["id"] in cache and cache[record["id"]].get("llm_raw") is not None:
                continue
            same = by_text.get(text_key(record["text"]))
            entry = make_cache_entry(record, same["llm_raw"] if same else None,
                                     {**meta, "chep_tu_id": same["id"] if same else None})
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            cache[record["id"]] = entry

    return [finalize(r, cache[r["id"]]) for r in records if r["id"] in cache]


def sample_clean(records: list[dict], n: int | None, seed: int = 42) -> list[dict]:
    """Giữ mọi câu có nhãn gốc khác CLEAN; với câu CLEAN chỉ lấy ngẫu nhiên n text khác nhau (seed cố định).

    Lấy mẫu theo text, nên câu trùng giữa ViHSD và ViHOS được giữ hoặc bỏ cùng nhau. Text nào có ít nhất một
    bản ghi nhãn gốc nguy hại thì không tính vào phần CLEAN. Giữ nguyên thứ tự bản ghi.
    """
    if n is None:
        return records
    not_clean = {r["text"] for r in records if (r.get("nhan_goc") or "").upper() not in NHAN_GOC_SACH}
    clean_texts = sorted({r["text"] for r in records} - not_clean)
    chosen = set(random.Random(seed).sample(clean_texts, min(n, len(clean_texts))))
    return [r for r in records if r["text"] in not_clean or r["text"] in chosen]


def summarize(results: list[dict]) -> dict:
    return {
        "so_ban_ghi": len(results),
        "theo_nhan": dict(Counter(r["nhan"] or "(loi)" for r in results).most_common()),
        "theo_nhan_goc_va_nhan": {f"{g}->{n}": c for (g, n), c in sorted(
            Counter((r.get("nhan_goc"), r["nhan"] or "(loi)") for r in results).items())},
        "can_duyet": sum(r["can_duyet"] for r in results),
        "ly_do_duyet": dict(Counter(x for r in results for x in r["ly_do_duyet"]).most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="chỉ lấy N bản ghi đầu (sau lọc)")
    parser.add_argument("--nguon", nargs="*", default=None, help="chỉ lấy các nguồn này, ví dụ vihsd vihos")
    parser.add_argument("--mau-clean", type=int, default=None,
                        help="giữ mọi câu OFFENSIVE/HATE, chỉ lấy ngẫu nhiên N câu CLEAN gốc (seed 42)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1, help="số lô gọi API song song")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in prompt của lô đầu, không gọi API")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    records = [r for r in read_jsonl(args.input) if not args.nguon or r["nguon"] in args.nguon]
    kho_test = load_kho_test_ids()
    blocked = [r["id"] for r in records if r["id"] in kho_test]
    if blocked:
        print(f"Bỏ {len(blocked)} câu thuộc kho test (không gửi LLM): {blocked[:5]}...", file=sys.stderr)
        records = [r for r in records if r["id"] not in kho_test]
    records = sample_clean(records, args.mau_clean)
    if args.limit is not None:
        records = records[: args.limit]
    system = build_system_prompt()
    version = prompt_version(system)
    unique = len({r["text"] for r in records})
    print(f"Chọn {len(records)} bản ghi, {unique} text khác nhau "
          f"(khoảng {-(-unique // args.batch_size)} lượt gọi API nếu chưa có cache). "
          f"Theo nhãn gốc: {dict(Counter(r.get('nhan_goc') for r in records).most_common())}")

    if args.dry_run:
        print("===== SYSTEM =====")
        print(system)
        print("===== USER (lô đầu) =====")
        print(build_user_prompt([r["text"] for r in records[: args.batch_size]]))
        print(f"\n(prompt_version={version}, {len(records)} bản ghi được chọn, không gọi API)")
        return 0

    load_dotenv()
    config = config_from_env()
    meta = {"provider": config.provider, "model": config.model, "prompt_version": version, **config.describe()}
    tag = safe_name(f"{config.provider}_{config.model}_{version}")
    cache_path = args.out_dir / f"cache_{tag}.jsonl"

    def complete(system_prompt: str, user_prompt: str) -> str:
        return with_retries(lambda: call_llm(config, system_prompt, user_prompt))

    results = run(records, system=system, complete=complete, cache_path=cache_path,
                  meta=meta, batch_size=args.batch_size, workers=args.workers)
    print(f"Token đã dùng lượt này: {USAGE.summary()}")
    out_path = args.out_dir / f"prelabel_{tag}.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = summarize(results)
    (args.out_dir / f"prelabel_{tag}_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã ghi {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
