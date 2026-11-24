"""Kiểm tra rò rỉ giữa tập test và dữ liệu train trước khi đóng băng dataset.

Bốn phép kiểm:
1. id test trùng id train (lỗi).
2. Train có câu thuộc kịch bản giữ ngoài train (`giu_ngoai_train: true` trong kich_ban.yaml, ví dụ LD7) (lỗi).
3. Trùng nguyên văn sau chuẩn hóa (bỏ dấu, bỏ ký tự đặc biệt) (lỗi).
4. Trùng gần: so cả đoạn và từng dòng dài bằng 3-gram ký tự (Jaccard; độ chứa chỉ tính khi hai đơn vị dài gần
   bằng nhau). So từng dòng bắt trường hợp chép nguyên một dòng vào đoạn khác:
     - điểm >= --nguong-trung (mặc định 0,8): `trung_gan`, câu train đó phải bỏ khỏi train;
     - điểm >= --nguong-xem (mặc định 0,5): `can_xem`, người xem cặp câu rồi quyết định.
   Tùy chọn --embedding <model sentence-transformers> thêm cosine ngữ nghĩa (bắt câu diễn đạt lại).

Kết quả: báo cáo JSON, CSV các cặp cần xem, và danh sách id train cần bỏ. Chỉ mở CSV sau khi bộ test đã chốt
(HUONG_DAN_VIET_TEST.md: không đọc câu LLM sinh trước khi viết xong).

Chạy:
    python -m src.v2.data.kiem_tra_ro_ri --test data/v2/interim/test_nguoi_viet.jsonl \\
        --train data/v2/interim/sinh_llm_loc.jsonl
Mã thoát 1 nếu còn lỗi (1–3) hoặc còn cặp `trung_gan`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from v1.src.dataset_builder.dedup import _dedup_form, _shingles

ROOT = Path(__file__).resolve().parents[3]
KICH_BAN_PATH = ROOT / "src" / "v2" / "labeling" / "kich_ban.yaml"
OUT_DIR = ROOT / "results" / "v2" / "kiem_tra_ro_ri"
MIN_SHINGLES = 40  # dòng ngắn, câu đệm hay gặp ("có chuyện chi vậy?") không dùng để so độ chứa: dễ khớp nhầm


def held_out_codes(path: Path = KICH_BAN_PATH) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {row["ma"] for row in data.get("kich_ban", []) if row.get("giu_ngoai_train")}


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for n, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("id", f"{path.name}:{n}")
                    rows.append(row)
    return rows


def _units(text: str) -> list[set[str]]:
    """3-gram của cả đoạn và của từng dòng đủ dài."""
    whole = _dedup_form(text)
    units = [_shingles(whole)] if whole else []
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        for line in lines:
            sh = _shingles(_dedup_form(line))
            if len(sh) >= MIN_SHINGLES:
                units.append(sh)
    return units


def _scores(a: set[str], b: set[str]) -> tuple[float, float]:
    inter = len(a & b)
    if not inter:
        return 0.0, 0.0
    jac = inter / len(a | b)
    contain = inter / min(len(a), len(b)) if min(len(a), len(b)) >= MIN_SHINGLES else 0.0
    return jac, contain


class TrainIndex:
    def __init__(self, train: list[dict]):
        self.train = train
        self.units: list[tuple[int, set[str]]] = []
        self.inverted: dict[str, list[int]] = defaultdict(list)
        for ti, row in enumerate(train):
            for sh in _units(row.get("text", "")):
                ui = len(self.units)
                self.units.append((ti, sh))
                for g in sh:
                    self.inverted[g].append(ui)

    def best(self, text: str, top: int = 3) -> list[dict]:
        """Các câu train giống nhất: điểm = max(Jaccard, độ chứa) trên mọi cặp đơn vị (đoạn/dòng)."""
        best: dict[int, tuple[float, float, float]] = {}
        for sh in _units(text):
            counts: Counter = Counter()
            for g in sh:
                counts.update(self.inverted.get(g, ()))
            # chỉ tính kỹ các ứng viên chung ít nhất 30% 3-gram của đơn vị nhỏ hơn
            for ui, inter in counts.items():
                ti, other = self.units[ui]
                if inter < 0.3 * min(len(sh), len(other)):
                    continue
                jac, contain = _scores(sh, other)
                # Độ chứa 3-gram ký tự vô nghĩa khi một bên dài hơn nhiều (đoạn dài chứa gần hết 3-gram của câu
                # ngắn một cách ngẫu nhiên), nên chỉ tính khi hai đơn vị dài gần bằng nhau.
                ratio = min(len(sh), len(other)) / max(len(sh), len(other))
                score = max(jac, contain) if ratio >= 0.7 else jac
                if score > best.get(ti, (0.0,))[0]:
                    best[ti] = (score, jac, contain)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:top]
        return [{"train_idx": ti, "diem": round(s, 4), "jaccard": round(j, 4), "chua": round(c, 4)}
                for ti, (s, j, c) in ranked]


def embedding_scores(model_name: str, test: list[dict], train: list[dict]) -> list[tuple[int, float]]:
    """(chỉ số train gần nhất, cosine) cho từng câu test. Cần sentence-transformers."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 (tùy chọn)
    import numpy as np  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    a = model.encode([r["text"] for r in test], normalize_embeddings=True, batch_size=64)
    b = model.encode([r["text"] for r in train], normalize_embeddings=True, batch_size=64)
    sims = np.asarray(a) @ np.asarray(b).T
    idx = sims.argmax(axis=1)
    return [(int(i), float(sims[k, i])) for k, i in enumerate(idx)]


def check(test: list[dict], train: list[dict], held_out: set[str], nguong_trung: float = 0.8,
          nguong_xem: float = 0.5, emb: list[tuple[int, float]] | None = None,
          nguong_emb: float = 0.9) -> dict:
    errors: list[str] = []
    train_ids = Counter(str(r.get("id")) for r in train)
    for r in test:
        if str(r.get("id")) in train_ids:
            errors.append(f"id test trung id train: {r.get('id')}")

    leaked = [r for r in train if r.get("kich_ban") in held_out]
    for r in leaked[:50]:
        errors.append(f"train co cau kich ban giu ngoai train {r.get('kich_ban')}: {r.get('id')}")
    if len(leaked) > 50:
        errors.append(f"... va {len(leaked) - 50} cau khac thuoc kich ban giu ngoai train")

    exact: dict[str, list[int]] = defaultdict(list)
    for ti, r in enumerate(train):
        exact[_dedup_form(r.get("text", ""))].append(ti)

    index = TrainIndex(train)
    pairs: list[dict] = []
    drop: set[str] = set()
    muc_count: Counter = Counter()
    for k, r in enumerate(test):
        form = _dedup_form(r.get("text", ""))
        hits = index.best(r.get("text", ""))
        if form in exact:
            errors.append(f"test {r.get('id')} trung nguyen van train {train[exact[form][0]].get('id')}")
            hits = [{"train_idx": exact[form][0], "diem": 1.0, "jaccard": 1.0, "chua": 1.0}] + hits
        cos = None
        if emb is not None:
            ei, cos = emb[k]
            if not any(h["train_idx"] == ei for h in hits):
                hits.append({"train_idx": ei, "diem": 0.0, "jaccard": 0.0, "chua": 0.0})
        top_muc = "sach"
        for h in hits:
            emb_hit = emb is not None and h["train_idx"] == emb[k][0] and cos >= nguong_emb
            if h["diem"] >= nguong_trung:
                muc = "trung_gan"
            elif h["diem"] >= nguong_xem or emb_hit:
                muc = "can_xem"
            else:
                continue
            t = train[h["train_idx"]]
            if muc == "trung_gan":
                drop.add(str(t.get("id")))
            if top_muc != "trung_gan":
                top_muc = muc
            pairs.append({
                "muc": muc, "test_id": r.get("id"), "test_nhan": r.get("nhan", ""),
                "test_kich_ban": r.get("kich_ban", ""), "train_id": t.get("id"),
                "train_nhan": t.get("nhan", ""), "train_kich_ban": t.get("kich_ban", ""),
                "diem": h["diem"], "jaccard": h["jaccard"], "chua": h["chua"],
                "cosine": None if cos is None or h["train_idx"] != emb[k][0] else round(cos, 4),
                "test_text": r.get("text", ""), "train_text": t.get("text", ""),
            })
        muc_count[top_muc] += 1

    pairs.sort(key=lambda p: (p["muc"] != "trung_gan", -p["diem"]))
    report = {
        "so_cau_test": len(test),
        "so_cau_train": len(train),
        "kich_ban_giu_ngoai_train": sorted(held_out),
        "so_cau_train_thuoc_kich_ban_giu_ngoai": len(leaked),
        "test_theo_muc": dict(muc_count),
        "so_cap_trung_gan": sum(p["muc"] == "trung_gan" for p in pairs),
        "so_cap_can_xem": sum(p["muc"] == "can_xem" for p in pairs),
        "so_id_train_can_bo": len(drop),
        "nguong": {"trung_gan": nguong_trung, "can_xem": nguong_xem,
                   "embedding": nguong_emb if emb is not None else None},
        "loi": errors,
    }
    return {"report": report, "pairs": pairs, "drop": sorted(drop)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", nargs="+", type=Path, required=True)
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--kich-ban-file", type=Path, default=KICH_BAN_PATH)
    parser.add_argument("--nguong-trung", type=float, default=0.8)
    parser.add_argument("--nguong-xem", type=float, default=0.5)
    parser.add_argument("--embedding", default=None, help="tên model sentence-transformers (tùy chọn)")
    parser.add_argument("--nguong-embedding", type=float, default=0.9)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--ghi-train-sach", type=Path, default=None,
                        help="ghi bản train đã bỏ các câu trung_gan vào file này")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test, train = read_jsonl(args.test), read_jsonl(args.train)
    emb = embedding_scores(args.embedding, test, train) if args.embedding else None
    result = check(test, train, held_out_codes(args.kich_ban_file), args.nguong_trung, args.nguong_xem,
                   emb, args.nguong_embedding)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "bao_cao.json").write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "cap_can_xem.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["muc", "test_id", "test_nhan", "test_kich_ban", "train_id", "train_nhan", "train_kich_ban",
                  "diem", "jaccard", "chua", "cosine", "test_text", "train_text"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["pairs"])
    (args.out_dir / "id_train_can_bo.txt").write_text(
        "".join(f"{i}\n" for i in result["drop"]), encoding="utf-8")
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    if args.ghi_train_sach:
        drop = set(result["drop"])
        kept = [r for r in train if str(r.get("id")) not in drop]
        with args.ghi_train_sach.open("w", encoding="utf-8", newline="\n") as handle:
            for r in kept:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Đã ghi {len(kept)} câu train (bỏ {len(train) - len(kept)}) vào {args.ghi_train_sach}")
        return 1 if result["report"]["loi"] else 0
    bad = result["report"]["loi"] or result["report"]["so_cap_trung_gan"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
