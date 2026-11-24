"""Sinh biến dạng né lọc cho tập TEST (và validation) v2. Không bao giờ chạy trên train.

Mỗi câu test sinh ra một bản cho từng kiểu biến dạng, cộng hai bản trộn (C1 trộn, C2 trộn). Biến dạng làm theo
từng từ trong từng dòng, nên cụm `bang_chung` được biến dạng đúng như trong câu và vẫn là chuỗi con của `text`.

Kiểu biến dạng:

| Kiểu | Nhóm | Bộ chuẩn hóa v1 sửa được? |
| --- | --- | --- |
| teencode (không → ko, biết → bik) | C1 | có (từ điển) |
| bo_dau (mày → may) | C1 | không |
| keo_dai (ngu → nguuuu) | C1 | có |
| go_sai (lỗi gõ: đổi chỗ, thiếu, thừa, phím bên cạnh) | C1 | không |
| chen_ky_tu (ngu → n.g.u, n-g_u) | C1 | có |
| so_thay_chu (chết → ch3t, ngu → n9u) | C1 | có |
| dinh_tu (đánh chết → đánhchết) | C1 | không |
| cyrillic_homoglyph, zero_width, exotic_separator, space_injection, double_elongation, fullwidth | C2 | không (điểm mù v1) |

"Đã thấy" / "chưa thấy": kiểu biến dạng đã xuất hiện tự nhiên trong dữ liệu train hay chưa. Chạy với --train để
script đo tỉ lệ câu train có dấu hiệu của từng kiểu (chỉ đọc train, không sửa) và xếp kiểu nào có tỉ lệ
>= --nguong-da-thay vào "đã thấy". Kiểu không đo được bằng quy tắc (go_sai, dinh_tu) lấy theo --da-thay-mac-dinh
(mặc định go_sai, vì prompt sinh có giọng "viết vội, sai chính tả"). Kết quả đo ghi vào file thống kê.

Chạy:
    python -m src.v2.data.perturb data/v2/test_day_du.jsonl \\
        --train data/v2/train.jsonl --out data/v2/interim/test_bien_dang.jsonl

Bản gọn (mỗi câu 2 dòng bốc ngẫu nhiên trong một nhóm, không giữ bản gốc):
    python -m src.v2.data.perturb data/v2/test.jsonl --train data/v2/train.jsonl --nhom C1 --so-ban 2 \\
        --out data/v2/test_perturb_C1.jsonl
    python -m src.v2.data.perturb data/v2/test.jsonl --train data/v2/train.jsonl --nhom C2 --so-ban 2 \\
        --out data/v2/test_perturb_C2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable

from src.shared.normalization.normalizer import TEENCODE_MAP
from src.shared.variant_generator import blindspot

ROOT = Path(__file__).resolve().parents[3]
SEED = 42
C1_KIEU = ("teencode", "bo_dau", "keo_dai", "go_sai", "chen_ky_tu", "so_thay_chu", "dinh_tu")
C2_KIEU = tuple(blindspot.TECHNIQUES)
DO_DUOC = ("teencode", "bo_dau", "keo_dai", "chen_ky_tu", "so_thay_chu", *C2_KIEU)
DA_THAY_MAC_DINH = ("go_sai",)

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
DOMAIN_RE = re.compile(r"\S+\[\.\]\S+|<[A-Z_]+>|\[[A-Z_]+\]")  # tên miền đã vô hiệu, <SDT>, [STK]: không đụng
VOWELS = set("aăâeêioôơuưy")
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "g": "9", "b": "8", "s": "5"}
SEPARATORS = (".", "-", "_")
KEYBOARD = {"a": "sqw", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "g": "ftyhbv", "h": "gyujnb",
            "i": "ujko", "k": "ijolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa",
            "r": "edft", "s": "awedxz", "t": "rfgy", "u": "yhji", "v": "cfgb", "x": "zsdc", "y": "tghu"}


def bo_dau_chu(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", value) if not unicodedata.combining(c)))


def _teencode_reverse() -> dict[str, list[str]]:
    """chuẩn → các dạng teencode, lấy từ từ điển của bộ chuẩn hóa (để bộ chuẩn hóa khôi phục được)."""
    reverse: dict[str, list[str]] = {}
    for teen, standard in TEENCODE_MAP.items():
        if " " in standard or " " in teen or not teen.strip():
            continue
        reverse.setdefault(standard.lower(), []).append(teen)
    return {k: sorted(v) for k, v in reverse.items()}


TEEN_REVERSE = _teencode_reverse()


def _case_like(source: str, value: str) -> str:
    if source[:1].isupper():
        return value[:1].upper() + value[1:]
    return value


# ---- biến dạng một từ: trả về từ mới hoặc None nếu không áp được ----

def w_teencode(word: str, rng: random.Random) -> str | None:
    forms = TEEN_REVERSE.get(word.lower())
    return _case_like(word, rng.choice(forms)) if forms else None


def w_bo_dau(word: str, rng: random.Random) -> str | None:
    out = bo_dau_chu(word)
    return out if out != word else None


def w_keo_dai(word: str, rng: random.Random) -> str | None:
    idx = [i for i, c in enumerate(word) if bo_dau_chu(c).lower() in VOWELS]
    if not idx:
        return None
    i = idx[-1]
    return word[: i + 1] + word[i] * rng.randint(2, 4) + word[i + 1:]


def w_go_sai(word: str, rng: random.Random) -> str | None:
    if len(word) < 3:
        return None
    p = rng.randint(1, len(word) - 2)
    action = rng.choice(("swap", "drop", "dup", "key"))
    if action == "swap" and word[p] != word[p + 1]:
        return word[:p] + word[p + 1] + word[p] + word[p + 2:]
    if action == "drop":
        return word[:p] + word[p + 1:]
    if action == "key":
        keys = [i for i, c in enumerate(word) if c.lower() in KEYBOARD]
        if keys:
            i = rng.choice(keys)
            return word[:i] + _case_like(word[i], rng.choice(KEYBOARD[word[i].lower()])) + word[i + 1:]
    return word[:p] + word[p] + word[p:]


def w_chen_ky_tu(word: str, rng: random.Random) -> str | None:
    if len(word) < 2:
        return None
    sep = rng.choice(SEPARATORS)
    if len(word) <= 4 or rng.random() < 0.5:
        return sep.join(word)  # n.g.u
    p = rng.randint(1, len(word) - 1)
    return word[:p] + sep + word[p:]


def w_so_thay_chu(word: str, rng: random.Random) -> str | None:
    # bộ chuẩn hóa chỉ đổi khi số chữ cái > số ký tự đổi được, và token không bắt đầu bằng số
    idx = [i for i, c in enumerate(word) if i > 0 and c.lower() in LEET]
    if not idx or len(word) < 3:
        return None
    k = 1 if len(word) < 6 else 2
    out = list(word)
    for i in rng.sample(idx, min(k, len(idx))):
        out[i] = LEET[out[i].lower()]
    return "".join(out)


def _c2(name: str) -> Callable[[str, random.Random], str | None]:
    fn = blindspot.TECHNIQUES[name]

    def apply(word: str, rng: random.Random) -> str | None:
        if len(word) < 3:
            return None
        if name in ("cyrillic_homoglyph", "fullwidth"):
            out = fn(word, rng, 0.6)
        else:
            out = fn(word, rng, 1)
        return out if out != word else None
    return apply


WORD_FUNCS: dict[str, Callable[[str, random.Random], str | None]] = {
    "teencode": w_teencode, "bo_dau": w_bo_dau, "keo_dai": w_keo_dai, "go_sai": w_go_sai,
    "chen_ky_tu": w_chen_ky_tu, "so_thay_chu": w_so_thay_chu,
    **{name: _c2(name) for name in C2_KIEU},
}


# ---- áp lên cả câu, giữ vị trí bằng chứng ----

def _segments(line: str) -> list[tuple[int, int, bool]]:
    """Chia dòng thành đoạn (đầu, cuối, là_từ). Vùng tên miền/ký hiệu che thông tin coi là không phải từ."""
    protected = [(m.start(), m.end()) for m in DOMAIN_RE.finditer(line)]
    segs: list[tuple[int, int, bool]] = []
    pos = 0
    for m in WORD_RE.finditer(line):
        if any(m.start() < r and m.end() > l for l, r in protected):
            continue
        if m.start() > pos:
            segs.append((pos, m.start(), False))
        segs.append((m.start(), m.end(), True))
        pos = m.end()
    if pos < len(line):
        segs.append((pos, len(line), False))
    return segs


def _evidence_spans(line: str, evidence: list[str]) -> list[tuple[int, int]]:
    spans = []
    for phrase in evidence:
        i = line.find(phrase)
        if i >= 0:
            spans.append((i, i + len(phrase)))
    return spans


def perturb_line(line: str, kieu: list[str], rng: random.Random, ti_le: float,
                 must_hit: list[tuple[int, int]]) -> tuple[str, list[int], Counter]:
    """Biến dạng một dòng. kieu: các kiểu được chọn (mỗi từ chọn ngẫu nhiên một kiểu áp được).
    Trả về (dòng mới, vị trí mới của từng ranh giới đoạn, đếm kiểu đã dùng)."""
    segs = _segments(line)
    word_kieu = [k for k in kieu if k in WORD_FUNCS]
    outs: list[str] = []
    used: Counter = Counter()
    hit_spans: set[int] = set()
    for start, end, is_word in segs:
        piece = line[start:end]
        if is_word and word_kieu:
            in_ev = [k for k, (l, r) in enumerate(must_hit) if start >= l and end <= r]
            force = any(k not in hit_spans for k in in_ev)
            if force or rng.random() < ti_le:
                for name in rng.sample(word_kieu, len(word_kieu)):
                    new = WORD_FUNCS[name](piece, rng)
                    if new and new != piece:
                        piece = new
                        used[name] += 1
                        hit_spans.update(in_ev)
                        break
        outs.append(piece)
    # dinh_tu: bỏ khoảng trắng giữa hai từ (không vắt qua ranh giới bằng chứng)
    if "dinh_tu" in kieu:
        for j in range(1, len(segs) - 1):
            s, e, is_word = segs[j]
            if is_word or outs[j] != " " or not (segs[j - 1][2] and segs[j + 1][2]):
                continue
            crosses = any((l < e <= r) != (l <= s < r) for l, r in must_hit)
            if not crosses and rng.random() < ti_le / 2:
                outs[j] = ""
                used["dinh_tu"] += 1
    starts = [0]
    for piece in outs:
        starts.append(starts[-1] + len(piece))
    return "".join(outs), [(segs[j][0], starts[j], starts[j + 1]) for j in range(len(segs))], used


def perturb_record(record: dict, kieu: list[str], ten: str, seed: int, ti_le: float) -> tuple[dict, Counter] | None:
    rng = random.Random(int(hashlib.sha1(f"{seed}|{record['id']}|{ten}".encode()).hexdigest()[:12], 16))
    evidence = list(record.get("bang_chung") or [])
    new_lines, new_evidence, used = [], {}, Counter()
    for line in record["text"].split("\n"):
        spans = _evidence_spans(line, [p for p in evidence if p not in new_evidence])
        new_line, bounds, u = perturb_line(line, kieu, rng, ti_le, spans)
        used.update(u)
        for (l, r), phrase in zip(spans, [p for p in evidence if p not in new_evidence and p in line]):
            # bounds lưu (đầu gốc, đầu mới, cuối mới) từng đoạn; cụm bằng chứng khớp ranh giới đoạn
            mapped = _map_span_bounds(bounds, len(line), new_line, l, r)
            if mapped is not None:
                new_evidence[phrase] = mapped
        new_lines.append(new_line)
    if not used:
        return None
    if any(p not in new_evidence for p in evidence):
        return None
    out = dict(record)
    out["text"] = "\n".join(new_lines)
    out["bang_chung"] = [new_evidence[p] for p in evidence]
    out["goc_id"] = record["id"]
    out["id"] = "v2_" + hashlib.sha1(f"bien_dang|{record['id']}|{ten}".encode()).hexdigest()[:12]
    out["kieu_bien_dang"] = ten
    out["so_tu_bien_dang"] = sum(used.values())
    out["kieu_da_dung"] = dict(used)
    return out, used


def _map_span_bounds(bounds: list[tuple[int, int, int]], line_len: int, new_line: str, l: int, r: int) -> str | None:
    """Đổi khoảng [l, r) của dòng gốc sang dòng mới. Cần l, r trùng ranh giới đoạn hoặc nằm trong đoạn không phải từ."""
    new_start = new_end = None
    for j, (os_, ns, ne) in enumerate(bounds):
        oe = bounds[j + 1][0] if j + 1 < len(bounds) else line_len
        if os_ <= l < oe and new_start is None:
            new_start = ns + (l - os_) if l != os_ else ns
            if l != os_ and (ne - ns) != (oe - os_):
                return None  # cắt giữa một đoạn đã đổi độ dài
        if os_ < r <= oe:
            if r == oe:
                new_end = ne
            elif (ne - ns) == (oe - os_):
                new_end = ns + (r - os_)
            else:
                return None
            break
    if new_start is None or new_end is None:
        return None
    return new_line[new_start:new_end]


# ---- đo kiểu biến dạng đã có trong train ----

_DETECT_TOKEN = re.compile(r"\S+")
VI_DIACRITIC = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


def detect(text: str) -> set[str]:
    found: set[str] = set()
    clean = DOMAIN_RE.sub(" ", text)
    tokens = [t.strip(".,!?;:()\"'").lower() for t in _DETECT_TOKEN.findall(clean)]
    if any(t in TEENCODE_MAP for t in tokens if t):
        found.add("teencode")
    for line in clean.split("\n"):
        words = WORD_RE.findall(line)
        if len(words) >= 4 and not VI_DIACRITIC.search(line):
            found.add("bo_dau")
    if re.search(r"([^\W\d_])\1{2,}", clean):
        found.add("keo_dai")
    if re.search(r"(?<![^\W\d_])[^\W\d_]([.\-_·~][^\W\d_]){2,}(?![^\W\d_])", clean) or \
            re.search(r"[^\W\d_]{2,}[\-_][^\W\d_]{1,}", clean):
        found.add("chen_ky_tu")
    if re.search(r"[^\W\d_][0-9@][^\W\d_]", clean):
        found.add("so_thay_chu")
    if any("Ѐ" <= c <= "ӿ" for c in clean):
        found.add("cyrillic_homoglyph")
    if any(c in "\u200b\u200c\u200d\ufeff" for c in clean):
        found.add("zero_width")
    if re.search(r"[^\W\d_][*/|+~^`][^\W\d_]", clean):
        found.add("exotic_separator")
    if any("！" <= c <= "～" for c in clean):
        found.add("fullwidth")
    if re.search(r"(?<![^\W\d_])[^\W\d_]( [^\W\d_]){3,}(?![^\W\d_])", clean):
        found.add("space_injection")
    return found


def train_coverage(rows: list[dict]) -> dict[str, float]:
    counts: Counter = Counter()
    for r in rows:
        counts.update(detect(r.get("text", "")))
    n = max(1, len(rows))
    return {k: round(counts[k] / n, 4) for k in DO_DUOC}


def classify(coverage: dict[str, float] | None, nguong: float, mac_dinh: tuple[str, ...]) -> dict[str, str]:
    groups = {}
    for name in (*C1_KIEU, *C2_KIEU):
        if coverage is not None and name in coverage:
            seen = coverage[name] >= nguong
        else:
            seen = name in mac_dinh
        groups[name] = "da_thay" if seen else "chua_thay"
    return groups


def build_chon(records: list[dict], nhom: str, so_ban: int, groups: dict[str, str], seed: int = SEED,
               ti_le: float = 0.6) -> tuple[list[dict], dict]:
    """Mỗi câu gốc lấy ngẫu nhiên `so_ban` kiểu trong nhóm `nhom` (C1 hoặc C2), mỗi kiểu một dòng, không giữ bản
    gốc. Kiểu bốc ngẫu nhiên theo (seed, id câu) nên chạy lại ra đúng kết quả cũ; kiểu không áp được vào câu
    thì bốc kiểu khác."""
    kieu_nhom = list(C1_KIEU if nhom == "C1" else C2_KIEU)
    out: list[dict] = []
    thieu: Counter = Counter()
    for record in records:
        rng = random.Random(f"{seed}|{nhom}|{record['id']}")
        order = kieu_nhom[:]
        rng.shuffle(order)
        got = 0
        for ten in order:
            if got == so_ban:
                break
            res = perturb_record(record, [ten], ten, seed, ti_le)
            if res is None:
                continue
            rec, _ = res
            rec["bien_dang"] = nhom
            rec["nhom_bien_dang"] = groups[ten]
            out.append(rec)
            got += 1
        if got < so_ban:
            thieu[record["id"]] = so_ban - got
    stats = {
        "nhom": nhom,
        "so_ban_moi_cau": so_ban,
        "so_cau_goc": len(records),
        "so_ban_ghi": len(out),
        "so_cau_khong_du_ban": len(thieu),
        "theo_kieu": dict(sorted(Counter(r["kieu_bien_dang"] for r in out).items())),
        "theo_nhom": dict(sorted(Counter(r["nhom_bien_dang"] for r in out).items())),
        "nhom_kieu": {k: v for k, v in groups.items() if k in kieu_nhom},
        "ti_le_tu_bien_dang": ti_le,
        "seed": seed,
    }
    return out, stats


def build(records: list[dict], groups: dict[str, str], seed: int = SEED, ti_le: float = 0.6) -> tuple[list[dict], dict]:
    plans: list[tuple[str, list[str], str]] = []
    for name in C1_KIEU:
        plans.append((name, [name], "C1"))
    for name in C2_KIEU:
        plans.append((name, [name], "C2"))
    da_thay = [k for k in (*C1_KIEU, *C2_KIEU) if groups[k] == "da_thay"]
    chua_thay = [k for k in (*C1_KIEU, *C2_KIEU) if groups[k] == "chua_thay"]
    plans += [("tron_C1", list(C1_KIEU), "C1"), ("tron_C2", list(C2_KIEU), "C2")]
    if da_thay:
        plans.append(("tron_da_thay", da_thay, "tron"))
    if chua_thay:
        plans.append(("tron_chua_thay", chua_thay, "tron"))

    out: list[dict] = []
    skipped: Counter = Counter()
    for record in records:
        base = dict(record, bien_dang="C0", kieu_bien_dang="goc", nhom_bien_dang="goc", goc_id=record["id"])
        out.append(base)
        for ten, kieu, cap in plans:
            res = perturb_record(record, kieu, ten, seed, ti_le)
            if res is None:
                skipped[ten] += 1
                continue
            rec, _ = res
            rec["bien_dang"] = cap
            if ten.startswith("tron_da") or ten.startswith("tron_chua"):
                rec["nhom_bien_dang"] = "da_thay" if ten == "tron_da_thay" else "chua_thay"
            elif ten.startswith("tron_"):
                rec["nhom_bien_dang"] = "tron"
            else:
                rec["nhom_bien_dang"] = groups[ten]
            out.append(rec)
    stats = {
        "so_cau_goc": len(records),
        "so_ban_ghi": len(out),
        "theo_kieu": dict(sorted(Counter(r["kieu_bien_dang"] for r in out).items())),
        "theo_nhom": dict(sorted(Counter(r["nhom_bien_dang"] for r in out).items())),
        "khong_ap_duoc": dict(sorted(skipped.items())),
        "nhom_kieu": groups,
        "ti_le_tu_bien_dang": ti_le,
        "seed": seed,
    }
    return out, stats


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="jsonl test hoặc validation (không dùng cho train)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=Path, nargs="*", default=None, help="train để đo kiểu đã thấy (chỉ đọc)")
    parser.add_argument("--nguong-da-thay", type=float, default=0.01,
                        help="tỉ lệ câu train có dấu hiệu để coi một kiểu là đã thấy (mặc định 1%%)")
    parser.add_argument("--da-thay-mac-dinh", nargs="*", default=list(DA_THAY_MAC_DINH),
                        help="kiểu không đo được tự động mà coi là đã thấy")
    parser.add_argument("--ti-le", type=float, default=0.6, help="tỉ lệ từ bị biến dạng mỗi câu (mạnh: 0,6)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--nhom", choices=("C1", "C2"), default=None,
                        help="chỉ sinh nhóm này, mỗi câu --so-ban dòng bốc ngẫu nhiên theo kiểu, không giữ bản gốc")
    parser.add_argument("--so-ban", type=int, default=2, help="số dòng mỗi câu gốc khi dùng --nhom (mặc định 2)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    records = read_jsonl(args.input)
    if any(r.get("split") == "train" for r in records):
        print("Đầu vào có câu split=train. Script này chỉ dành cho test/validation.", file=sys.stderr)
        return 1
    coverage = None
    if args.train:
        rows = [r for p in args.train for r in read_jsonl(p)]
        coverage = train_coverage(rows)
    groups = classify(coverage, args.nguong_da_thay, tuple(args.da_thay_mac_dinh))
    if args.nhom:
        out, stats = build_chon(records, args.nhom, args.so_ban, groups, args.seed, args.ti_le)
    else:
        out, stats = build(records, groups, args.seed, args.ti_le)
    stats["do_phu_train"] = coverage
    stats["nguong_da_thay"] = args.nguong_da_thay
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    args.out.with_name(args.out.stem + "_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
