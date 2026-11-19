"""Sinh câu train bằng LLM theo bảng kịch bản (kich_ban.yaml): kịch bản × kênh × giọng × độ lộ.

Người nhận tin luôn là học sinh THCS, THPT hoặc sinh viên. Mỗi ô nguy hại có một ô "sinh đôi" AN_TOAN
cùng kênh, giọng. Câu sinh CHỈ vào train (nguon=sinh_llm, cach_gan=llm, split=train). Kịch bản có
giu_ngoai_train=true không được sinh.

QUY TẮC AN TOÀN (đưa nguyên vào prompt):
- Không có nội dung tình dục tường minh, và không bao giờ mô tả trẻ vị thành niên theo hướng tình dục.
  Với QR1 và DD3, câu chỉ thể hiện kiểu tiếp cận ("gửi anh tấm ảnh riêng tư", "đừng kể với ai",
  "đổi thẻ game lấy ảnh"), không mô tả chi tiết hình ảnh hay hành vi.
- Không dùng tên, số điện thoại, tài khoản, trường học có thật; link chỉ là tên miền giả đã vô hiệu hóa
  dạng xxx[.]com.
- Câu sinh chỉ là nội dung tin nhắn, không sinh hướng dẫn cách lừa đảo hay cách thực hiện hành vi.

Ô bị LLM từ chối chỉ được ghi log và bỏ qua, không cố lách.

Cấu hình nhà cung cấp, model, khóa API: như prelabel.py (xem src/v2/labeling/llm.py).

Chạy:
    python -m src.v2.labeling.generate --dry-run
    python -m src.v2.labeling.generate --kich-ban LD1 DD1 --limit-o 2      # thử nhỏ
    python -m src.v2.labeling.generate --workers 4                         # chạy thật; chạy lại để tiếp tục
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import yaml

from src.dataset_builder.dedup import group_near_duplicates
from src.v2.data.merge import make_record, normalize_text
from src.v2.labeling import llm, prelabel

ROOT = Path(__file__).resolve().parents[3]
KICH_BAN_PATH = Path(__file__).with_name("kich_ban.yaml")
OUT_DIR = ROOT / "data" / "v2" / "interim"
DATA_V2 = ROOT / "data" / "v2"
TEST_NGUOI_VIET = OUT_DIR / "test_nguoi_viet.jsonl"
DEFAULT_SO_CAU = 10
MIN_WORDS = 3

SAFETY_RULES = """\
QUY TẮC AN TOÀN:
- Không có nội dung tình dục tường minh, và không bao giờ mô tả trẻ vị thành niên theo hướng tình dục. Với QR1 và DD3, câu chỉ thể hiện kiểu tiếp cận ("gửi anh tấm ảnh riêng tư", "đừng kể với ai", "đổi thẻ game lấy ảnh"), không mô tả chi tiết hình ảnh hay hành vi.
- Không dùng tên, số điện thoại, tài khoản, trường học có thật; link chỉ là tên miền giả đã vô hiệu hóa dạng xxx[.]com.
- Câu sinh chỉ là nội dung tin nhắn, không sinh hướng dẫn cách lừa đảo hay cách thực hiện hành vi."""

GENERATION_RULES = """\
Bạn đang tạo dữ liệu huấn luyện cho SafeViChi, ứng dụng chạy trên điện thoại để phát hiện và cảnh báo tin nhắn
nguy hại gửi tới trẻ em và thanh thiếu niên Việt Nam. Dữ liệu chỉ dùng để dạy mô hình nhận ra và cảnh báo.

Cách viết:
- Viết tin nhắn giống thật như người Việt hay nhắn, đúng kênh, giọng văn và độ lộ được yêu cầu.
  Người nhận luôn là học sinh THCS, THPT hoặc sinh viên.
- Mỗi tin 1 đến 4 câu. Các tin trong một lượt phải khác nhau rõ về cách mở đầu, độ dài, chi tiết, người nhắn.
- Gán nhãn đúng theo hướng dẫn ở trên. Tin AN_TOAN phải thật sự an toàn theo hướng dẫn.
- bang_chung: các cụm chép NGUYÊN VĂN từ chính tin đó, theo mục 5 của hướng dẫn. Tin AN_TOAN để [].
- Số điện thoại ghi [SDT], số tài khoản ghi [STK], tên người dùng tên phổ biến bịa ra.
- Chỉ trả về một JSON list, không thêm lời dẫn, lời giải thích hay lưu ý nào."""

ASSISTANT_PATTERN = re.compile(
    r"dưới đây là|lưu ý\s*:|tôi không thể|tôi xin lỗi|xin lỗi,? tôi|là một (trợ lý|ai|mô hình)"
    r"|as an ai|here (is|are) |i can.?t", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(
    r"(?<![\w\[])([a-z0-9][a-z0-9-]*)\.(com|vn|net|org|xyz|top|info|io|me|cc|site|online|shop|link|ly|club|vip|app)\b",
    re.IGNORECASE)


# ---------------------------------------------------------------- bảng kịch bản

@dataclass(frozen=True)
class KichBan:
    ma: str
    nhan: str
    mo_ta: str
    sinh_doi: str
    kenh: tuple[str, ...]
    giong: tuple[str, ...]
    giu_ngoai_train: bool
    vi_du: tuple[str, ...]


@dataclass(frozen=True)
class BangKichBan:
    kenh: dict[str, str]
    giong: dict[str, str]
    do_lo: dict[str, str]
    do_lo_sinh_doi: dict[str, str]
    kich_ban: tuple[KichBan, ...]


def load_kich_ban(path: Path = KICH_BAN_PATH) -> BangKichBan:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kenh, giong, do_lo = data["kenh"], data["giong"], data["do_lo"]
    do_lo_sinh_doi = data["do_lo_sinh_doi"]
    if set(do_lo) != set(do_lo_sinh_doi):
        raise ValueError("do_lo và do_lo_sinh_doi phải có cùng mã")
    items: list[KichBan] = []
    seen: set[str] = set()
    for row in data["kich_ban"]:
        kb = KichBan(
            ma=row["ma"], nhan=row["nhan"], mo_ta=row["mo_ta"], sinh_doi=row["sinh_doi"],
            kenh=tuple(row["kenh"]), giong=tuple(row["giong"]),
            giu_ngoai_train=bool(row.get("giu_ngoai_train", False)), vi_du=tuple(row.get("vi_du") or ()),
        )
        if kb.ma in seen:
            raise ValueError(f"Trùng mã kịch bản {kb.ma}")
        seen.add(kb.ma)
        if kb.nhan not in prelabel.LABELS or kb.nhan == "AN_TOAN":
            raise ValueError(f"{kb.ma}: nhãn kịch bản phải là nhãn nguy hại, gặp {kb.nhan!r}")
        unknown = [k for k in kb.kenh if k not in kenh] + [g for g in kb.giong if g not in giong]
        if unknown:
            raise ValueError(f"{kb.ma}: kênh/giọng không có trong bảng: {unknown}")
        if len(kb.vi_du) > 5:
            raise ValueError(f"{kb.ma}: tối đa 5 câu ví dụ")
        items.append(kb)
    return BangKichBan(kenh, giong, do_lo, do_lo_sinh_doi, tuple(items))


def load_test_texts(directory: Path = DATA_V2) -> set[str]:
    """Text của mọi câu split=test trong data/v2/*.jsonl (chưa có thì rỗng)."""
    texts: set[str] = set()
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if '"test"' not in line:
                    continue
                row = json.loads(line)
                if row.get("split") == "test":
                    texts.add(normalize_text(row["text"]).lower())
    return texts


def check_examples_not_in_test(bang: BangKichBan, test_texts: set[str]) -> None:
    leaked = [(kb.ma, ex) for kb in bang.kich_ban for ex in kb.vi_du
              if normalize_text(ex).lower() in test_texts]
    if leaked:
        raise ValueError(f"Câu ví dụ trong kich_ban.yaml nằm trong tập test, phải bỏ: {leaked[:3]}")


# ---------------------------------------------------------------- ô sinh

@dataclass(frozen=True)
class O:
    o_id: str
    ma: str
    nhan: str          # nhãn mong đợi của câu sinh ra
    sinh_doi: bool
    kenh: str
    giong: str
    do_lo: str
    so_cau: int


def plan_cells(bang: BangKichBan, ma: list[str] | None = None, so_cau: int = DEFAULT_SO_CAU,
               co_sinh_doi: bool = True) -> list[O]:
    """Mọi ô theo thứ tự cố định; mỗi ô nguy hại kèm ngay ô sinh đôi AN_TOAN."""
    cells: list[O] = []
    for kb in bang.kich_ban:
        if kb.giu_ngoai_train or (ma and kb.ma not in ma):
            continue
        for kenh in kb.kenh:
            for giong in kb.giong:
                for do_lo in bang.do_lo:
                    base = f"{kb.ma}-{kenh}-{giong}-{do_lo}"
                    cells.append(O(base, kb.ma, kb.nhan, False, kenh, giong, do_lo, so_cau))
                    if co_sinh_doi:
                        cells.append(O(f"{base}-SD", kb.ma, "AN_TOAN", True, kenh, giong, do_lo, so_cau))
    return cells


def build_system_prompt() -> str:
    sections, _ = prelabel.load_guide_sections()
    return f"{sections}\n\n{GENERATION_RULES}\n\n{SAFETY_RULES}"


def build_user_prompt(cell: O, bang: BangKichBan) -> str:
    kb = next(k for k in bang.kich_ban if k.ma == cell.ma)
    lines = []
    if cell.sinh_doi:
        lines += [
            f"Đây là ô SINH ĐÔI AN TOÀN của kịch bản nguy hại {kb.ma} ({kb.mo_ta}).",
            f"Viết tin AN_TOAN theo tình huống: {kb.sinh_doi}",
            f"Mức giống tin nguy hại: {bang.do_lo_sinh_doi[cell.do_lo]}",
        ]
    else:
        lines += [
            f"Kịch bản {kb.ma}, nhãn {kb.nhan}: {kb.mo_ta}",
            f"Độ lộ: {bang.do_lo[cell.do_lo]}",
        ]
    lines += [
        f"Kênh: {bang.kenh[cell.kenh]}",
        f"Giọng văn: {bang.giong[cell.giong]}",
        "Người nhận: học sinh THCS, THPT hoặc sinh viên.",
    ]
    if kb.vi_du and not cell.sinh_doi:
        lines.append("Vài tin thật cùng kịch bản để tham khảo phong cách (không chép lại):")
        lines += [f"- {json.dumps(ex, ensure_ascii=False)}" for ex in kb.vi_du]
    template = json.dumps({"text": "...", "nhan": cell.nhan, "bang_chung": [], "kich_ban": kb.ma,
                           "kenh": cell.kenh, "giong": cell.giong, "do_lo": cell.do_lo}, ensure_ascii=False)
    lines += [
        f"Viết {cell.so_cau} tin khác nhau. Trả về một JSON list gồm {cell.so_cau} phần tử dạng:",
        template,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- phân tích và kiểm đầu ra

def parse_items(content: str) -> list[dict]:
    """Lấy JSON list trong phản hồi; nếu không có thì thử từng dòng JSON."""
    cleaned = re.sub(r"```(?:json)?", "", content)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    items = []
    for line in cleaned.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    return items


def defang(text: str) -> str:
    """Vô hiệu hóa tên miền: abc.com -> abc[.]com (bỏ http:// đi kèm)."""
    text = re.sub(r"https?://", "", text, flags=re.IGNORECASE)
    return DOMAIN_PATTERN.sub(r"\1[.]\2", text)


def check_item(item: dict, cell: O) -> tuple[dict | None, str | None]:
    """Kiểm một câu sinh. Trả về (kết quả đã kiểm, None) hoặc (None, lý do bỏ)."""
    raw_text = item.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "khong_co_text"
    text = defang(normalize_text(raw_text))
    if len(text.split()) < MIN_WORDS:
        return None, "qua_ngan"
    if ASSISTANT_PATTERN.search(text):
        return None, "giong_tro_ly"
    nhan = item.get("nhan")
    nhan = nhan.strip().upper() if isinstance(nhan, str) else nhan
    if nhan not in prelabel.LABELS:
        return None, "nhan_khong_hop_le"
    if nhan != cell.nhan:
        return None, "nhan_khong_khop_kich_ban"
    evidence = item.get("bang_chung") or []
    if isinstance(evidence, list):
        evidence = [defang(normalize_text(p)) if isinstance(p, str) else p for p in evidence]
    v = prelabel.validate_output(
        {"nhan": nhan, "bang_chung": evidence, "the": item.get("the") or [], "do_chac": 1.0}, text)
    ly_do_duyet = []
    if nhan != "AN_TOAN" and not v.bang_chung:
        ly_do_duyet.append("mat_het_bang_chung")
    return {
        "text": text, "nhan": nhan, "bang_chung": v.bang_chung, "the": v.the,
        "bang_chung_bi_bo": v.bang_chung_bi_bo, "can_duyet": bool(ly_do_duyet), "ly_do_duyet": ly_do_duyet,
    }, None


def to_record(checked: dict, cell: O, index: int, meta: dict) -> dict:
    record = make_record(text=checked["text"], nguon="sinh_llm", nguon_id=f"{cell.o_id}:{index}",
                         nhan_goc="", split_goc="")
    record.pop("bang_chung_goi_y", None)
    record.update({
        "nhan": checked["nhan"], "bang_chung": checked["bang_chung"], "the": checked["the"],
        "cach_gan": "llm", "split": "train",
        "kich_ban": cell.ma, "kenh": cell.kenh, "giong": cell.giong, "do_lo": cell.do_lo,
        "sinh_doi": cell.sinh_doi,
        "can_duyet": checked["can_duyet"], "ly_do_duyet": checked["ly_do_duyet"],
        "llm_bang_chung_bi_bo": checked["bang_chung_bi_bo"],
        "llm_provider": meta.get("provider"), "llm_model": meta.get("model"),
        "llm_prompt_version": meta.get("prompt_version"),
    })
    return record


# ---------------------------------------------------------------- gọi LLM, cache, resume

def load_raw(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # dòng cuối bị cắt khi ngắt giữa chừng
                done[entry["o_id"]] = entry
    return done


def run_cells(
    cells: list[O],
    bang: BangKichBan,
    *,
    system: str,
    complete: Callable[[str, str], str],
    raw_path: Path,
    meta: dict,
    workers: int = 1,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Gọi LLM cho các ô chưa xong. Ô lỗi mạng/API được thử lại ở lần chạy sau; ô bị từ chối thì không."""
    done = load_raw(raw_path)
    todo = [c for c in cells if c.o_id not in done or done[c.o_id].get("loi")]
    log(f"{len(cells)} ô, đã xong {len(cells) - len(todo)}, cần gọi {len(todo)}.")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def work(cell: O) -> dict:
        try:
            content, error = complete(system, build_user_prompt(cell, bang)), None
        except Exception as exc:  # noqa: BLE001
            content, error = "", f"{type(exc).__name__}: {exc}"
        items = parse_items(content)
        return {"o_id": cell.o_id, "o": asdict(cell), "raw": content, "items": items, "loi": error,
                "tu_choi": error is None and not items, **meta}

    with raw_path.open("a", encoding="utf-8", newline="\n") as handle, \
            ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, cell) for cell in todo]
        for n, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[entry["o_id"]] = entry
            if entry["loi"]:
                log(f"  {entry['o_id']}: lỗi gọi API, sẽ thử lại lần sau ({entry['loi'][:120]})")
            elif entry["tu_choi"]:
                log(f"  {entry['o_id']}: LLM từ chối hoặc trả sai định dạng, bỏ qua: {entry['raw'][:120]!r}")
            if n % 20 == 0 or n == len(futures):
                log(f"  xong {n}/{len(futures)}")
    return done


def load_test_nguoi_viet(path: Path = TEST_NGUOI_VIET) -> list[str]:
    """Câu test do người viết (nếu đã có), để loại câu sinh gần trùng khỏi train."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle if line.strip()]


def collect(cells: list[O], done: dict[str, dict], meta_default: dict | None = None,
            cau_test: list[str] | None = None) -> tuple[list[dict], dict]:
    """Kiểm từng câu, khử trùng gần đúng giữa các câu sinh và với câu test người viết (dedup.py),
    trả về (bản ghi, thống kê)."""
    dropped: Counter = Counter()
    refused: list[str] = []
    errors: list[str] = []
    records: list[dict] = []
    for cell in cells:
        entry = done.get(cell.o_id)
        if entry is None:
            continue
        if entry.get("loi"):
            errors.append(cell.o_id)
            continue
        if entry.get("tu_choi"):
            refused.append(cell.o_id)
            continue
        meta = {k: entry.get(k) for k in ("provider", "model", "prompt_version")} or (meta_default or {})
        for i, item in enumerate(entry["items"]):
            checked, reason = check_item(item, cell)
            if reason:
                dropped[reason] += 1
                continue
            records.append(to_record(checked, cell, i, meta))

    kept = records
    if records:
        cau_test = cau_test or []
        texts = list(cau_test) + [r["text"] for r in records]
        groups = group_near_duplicates([SimpleNamespace(text=t) for t in texts]).group_ids
        test_groups = set(groups[: len(cau_test)])
        first: set[str] = set()
        kept = []
        for record, group in zip(records, groups[len(cau_test):]):
            if group in test_groups:
                dropped["trung_cau_test_nguoi_viet"] += 1
                continue
            if group in first:
                dropped["trung_gan_dung"] += 1
                continue
            first.add(group)
            kept.append(record)

    def count(key: str) -> dict:
        return dict(sorted(Counter(str(r[key]) for r in kept).items()))

    stats = {
        "so_cau": len(kept),
        "theo_nhan": count("nhan"),
        "theo_kich_ban": dict(sorted(Counter(f"{r['kich_ban']}{'-SD' if r['sinh_doi'] else ''}"
                                             for r in kept).items())),
        "theo_kenh": count("kenh"),
        "theo_giong": count("giong"),
        "theo_do_lo": count("do_lo"),
        "can_duyet": sum(r["can_duyet"] for r in kept),
        "cau_bi_bo": dict(dropped),
        "so_o_da_chay": sum(1 for c in cells if c.o_id in done),
        "o_bi_tu_choi": refused,
        "o_loi_api": errors,
    }
    return kept, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kich-ban", nargs="*", default=None, help="chỉ chạy các mã này, ví dụ LD1 DD1")
    parser.add_argument("--limit-o", type=int, default=None, help="chỉ chạy N ô đầu (sau lọc)")
    parser.add_argument("--so-cau-moi-o", type=int, default=DEFAULT_SO_CAU)
    parser.add_argument("--khong-sinh-doi", action="store_true", help="bỏ các ô sinh đôi AN_TOAN")
    parser.add_argument("--workers", type=int, default=1, help="số lượt gọi song song")
    parser.add_argument("--kich-ban-file", type=Path, default=KICH_BAN_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="in prompt mẫu cho từng nhãn, không gọi API")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bang = load_kich_ban(args.kich_ban_file)
    check_examples_not_in_test(bang, load_test_texts())
    cells = plan_cells(bang, args.kich_ban, args.so_cau_moi_o, not args.khong_sinh_doi)
    if args.limit_o is not None:
        cells = cells[: args.limit_o]
    system = build_system_prompt()
    version = prelabel.prompt_version(system)

    if args.dry_run:
        print("===== SYSTEM =====")
        print(system)
        shown: set[str] = set()
        for cell in cells:
            if cell.nhan in shown:
                continue
            shown.add(cell.nhan)
            print(f"\n===== USER: {cell.nhan} (ô {cell.o_id}) =====")
            print(build_user_prompt(cell, bang))
        print("\n===== KẾ HOẠCH =====")
        per_label = Counter(c.nhan for c in cells)
        for nhan, n in sorted(per_label.items()):
            print(f"  {nhan}: {n} ô × {args.so_cau_moi_o} câu ≈ {n * args.so_cau_moi_o} câu")
        held = [kb.ma for kb in bang.kich_ban if kb.giu_ngoai_train]
        print(f"Tổng {len(cells)} lượt gọi. Để ngoài train: {', '.join(held) or 'không'}. "
              f"prompt_version={version}. Không gọi API.")
        return 0

    llm.load_dotenv()
    config = llm.config_from_env()
    config.temperature = 1.0  # cần câu đa dạng
    meta = {"provider": config.provider, "model": config.model, "prompt_version": version}
    tag = llm.safe_name(f"{config.provider}_{config.model}_{version}")

    def complete(system_prompt: str, user_prompt: str) -> str:
        return llm.with_retries(lambda: llm.call_llm(config, system_prompt, user_prompt))

    raw_path = args.out_dir / "sinh_llm_cache" / f"raw_{tag}.jsonl"
    all_cells = plan_cells(bang, None, args.so_cau_moi_o, not args.khong_sinh_doi)
    run_cells(cells, bang, system=system, complete=complete, raw_path=raw_path, meta=meta,
              workers=args.workers)
    # Gộp mọi ô đã chạy (kể cả ở lần trước) thành một file đầu ra.
    records, stats = collect(all_cells, load_raw(raw_path), meta, cau_test=load_test_nguoi_viet())
    out_path = args.out_dir / "sinh_llm.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats.update(meta)
    (args.out_dir / "sinh_llm_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Đã ghi {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
