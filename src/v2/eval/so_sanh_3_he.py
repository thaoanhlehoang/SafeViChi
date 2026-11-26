"""So sánh 3 hệ trên bộ test v2: SafeViChi v1, gpt-oss-120b (API) và SafeViChi v2.

Chấm từng tập C0/C1/C2, mỗi tập chia theo giọng vùng (tất cả, Bắc, Trung, Nam), mọi ô đều tính lại
trực tiếp trên tập con bằng bộ chấm v2 (`metrics.evaluate_bundle`). v1 chỉ có HATE/CLEAN nên chỉ có
các độ đo nhị phân (`cham_v1.binary_metrics`). Ghi `metrics.json` (toàn bộ số) và `KET_QUA.md`.

    python -m src.v2.eval.so_sanh_3_he
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from src.v2.eval.cham_v1 import binary_metrics
from src.v2.eval.evaluate import align_predictions, read_jsonl
from src.v2.eval.evidence import evaluate_evidence
from src.v2.eval.metrics import LABELS, SEED_DEFAULT, evaluate_bundle

HARMFUL = tuple(l for l in LABELS if l != "AN_TOAN")

SETS = {
    "C0": ("data/v2/test_day_du.jsonl", "results/v2/llm_api/test_predictions.jsonl",
           "results/v2/v1_baseline/test_v1_predictions.jsonl"),
    "C1": ("data/v2/test_perturb_C1.jsonl", "results/v2/llm_api/test_perturb_C1_predictions.jsonl",
           "results/v2/v1_baseline/test_perturb_C1_v1_predictions.jsonl"),
    "C2": ("data/v2/test_perturb_C2.jsonl", "results/v2/llm_api/test_perturb_C2_predictions.jsonl",
           "results/v2/v1_baseline/test_perturb_C2_v1_predictions.jsonl"),
}
SET_DESC = {"C0": "bản gốc", "C1": "biến dạng thường gặp (gồm cả nhóm giữ ngoài train)", "C2": "kiểu điểm mù chưa thấy"}
REGIONS = (("tat_ca", "Tất cả"), ("bac", "Bắc"), ("trung", "Trung"), ("nam", "Nam"))
SYSTEMS = (("v1", "v1 (ViHateT5)"), ("llm", "gpt-oss-120b"), ("v2", "v2"))
OUT_DIR = Path("results/v2/so_sanh_3_he")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def llm_rows(path: Path) -> list[dict[str, Any]]:
    return [{"id": r["id"], "nhan": r.get("nhan_llm"), "bang_chung": r.get("bang_chung_llm") or []} for r in read_jsonl(path)]


def score_multiclass(records, preds, n_bootstrap: int, seed: int) -> dict[str, Any]:
    rep = evaluate_bundle(records, preds, n_bootstrap=n_bootstrap, seed=seed, include_slices=False)
    c, b, e = rep["classification"], rep["classification"]["binary_harmful_vs_safe"], rep["evidence"]
    out = {
        "n": c["n"], "macro_f1": c["macro_f1"], "binary_f1": b["f1"], "precision": b["precision"],
        "recall": b["recall"], "fpr_an_toan": b["fpr_an_toan"],
        "bc_p_theo_y": e["f1_theo_y"]["precision"], "bc_r_theo_y": e["f1_theo_y"]["recall"],
        "bc_f1_theo_y": e["f1_theo_y"]["f1"], "bc_f1_muc_tu": e["f1_muc_tu"]["f1"],
        "nguy_hai_khong_bc": e["harmful_empty_evidence_rate"],
        "n_bc": e["n_harmful_evaluated"], "n_bc_bo_qua": e.get("n_harmful_skipped_pred_safe", 0),
        "theo_nhan": c["per_class"],
        "ma_tran_nham": c["confusion_matrix"],
    }
    # Bằng chứng theo từng nhãn nguy hại (tập con các câu có nhãn đúng là nhãn đó)
    bc_nhan = {}
    for lbl in HARMFUL:
        idx = [i for i, r in enumerate(records) if str(r["nhan"]) == lbl]
        ev = evaluate_evidence([records[i] for i in idx], [preds[i] for i in idx])
        bc_nhan[lbl] = {"n_bc": ev["n_harmful_evaluated"], "bc_f1_theo_y": ev["f1_theo_y"]["f1"],
                        "bc_f1_muc_tu": ev["f1_muc_tu"]["f1"]}
    out["bc_theo_nhan"] = bc_nhan
    if c.get("bootstrap_95"):
        out["ci95"] = c["bootstrap_95"]
    return out


def score_v1(records, preds) -> dict[str, Any]:
    gold_harm = [str(r["nhan"]) != "AN_TOAN" for r in records]
    pred_harm = [str(p["nhan_v1"]).upper() == "HATE" for p in preds]
    b = binary_metrics(gold_harm, pred_harm)
    # Tỉ lệ báo đúng theo nhãn thật: nhãn nguy hại = tỉ lệ bị báo HATE; AN_TOAN = tỉ lệ được giữ CLEAN
    theo_nhan = {}
    for lbl in ("AN_TOAN", *HARMFUL):
        idx = [i for i, r in enumerate(records) if str(r["nhan"]) == lbl]
        hit = sum((not pred_harm[i]) if lbl == "AN_TOAN" else pred_harm[i] for i in idx)
        theo_nhan[lbl] = {"support": len(idx), "recall": round(hit / len(idx), 4) if idx else 0.0}
    return {"n": b["n"], "binary_f1": b["f1"], "precision": b["precision"], "recall": b["recall"],
            "fpr_an_toan": b["fpr_an_toan"], "theo_nhan": theo_nhan}


# Các chiều chia thêm: (khóa, tên hiển thị, các tập áp dụng, thứ tự giá trị và nhãn hiển thị)
DIMENSIONS = (
    ("kieu_bien_dang", "Kiểu biến dạng", ("C1", "C2"), None),
    ("nhom_bien_dang", "Nhóm biến dạng (đã thấy/chưa thấy khi train v2)", ("C1",),
     {"da_thay": "đã thấy", "chua_thay": "chưa thấy"}),
    ("phan_test", "Phần test (kịch bản trong/ngoài phân phối train)", ("C0", "C1", "C2"),
     {"trong_phan_phoi": "trong phân phối", "chua_thay": "kịch bản chưa thấy"}),
    ("so_luot", "Số lượt hội thoại", ("C0", "C1", "C2"), {"1_dong": "1 dòng", "nhieu_dong": "2–8 dòng"}),
    ("do_lo", "Độ lộ của câu nguy hại", ("C0", "C1", "C2"), {"lo_lieu": "lộ liễu", "an_y": "ẩn ý"}),
    ("kenh", "Kênh", ("C0", "C1", "C2"),
     {"messenger_zalo": "Messenger/Zalo", "sms": "SMS", "binh_luan": "bình luận", "nhom_lop": "nhóm lớp",
      "chat_game": "chat game"}),
    ("nguoi_viet", "Người viết câu test", ("C0", "C1", "C2"), None),
)
FIELDS_FROM_C0 = ("so_dong", "do_lo", "kenh", "nguoi_viet", "phan_test")


def enrich(records: list[dict[str, Any]], c0_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Bổ sung các trường chỉ có ở bản gốc C0 cho câu biến dạng (qua `goc_id`) và tính `so_luot`."""
    out = []
    for r in records:
        src = c0_by_id.get(str(r.get("goc_id") or r["id"]), {})
        m = dict(r)
        for k in FIELDS_FROM_C0:
            if m.get(k) in (None, "") and src.get(k) not in (None, ""):
                m[k] = src[k]
        so_dong = m.get("so_dong")
        m["so_luot"] = None if so_dong is None else ("1_dong" if int(so_dong) == 1 else "nhieu_dong")
        out.append(m)
    return out


def run(v2_path: Path, n_bootstrap: int, seed: int) -> dict[str, Any]:
    v2_all = read_jsonl(v2_path)
    c0_by_id = {str(r["id"]): r for r in read_jsonl(Path(SETS["C0"][0]))}
    result: dict[str, Any] = {"lat_cat": {d[0]: {} for d in DIMENSIONS}}
    for set_name, (gold_p, llm_p, v1_p) in SETS.items():
        gold = enrich(read_jsonl(Path(gold_p)), c0_by_id)
        gold_ids = {str(r["id"]) for r in gold}
        preds = {
            "v1": align_predictions(gold, read_jsonl(Path(v1_p))),
            "llm": align_predictions(gold, llm_rows(Path(llm_p))),
            "v2": align_predictions(gold, [r for r in v2_all if str(r["id"]) in gold_ids]),
        }
        # Kiểm tra nhãn đúng trong file v2 khớp gold
        lech = sum(str(p.get("nhan_dung", r["nhan"])) != str(r["nhan"]) for r, p in zip(gold, preds["v2"]))
        if lech:
            raise ValueError(f"{set_name}: {lech} mẫu có nhan_dung trong file v2 lệch gold")
        result[set_name] = {}
        for reg, _ in REGIONS:
            idx = [i for i, r in enumerate(gold) if reg == "tat_ca" or r.get("giong_vung") == reg]
            sub_g = [gold[i] for i in idx]
            nb = n_bootstrap if reg == "tat_ca" else 0
            result[set_name][reg] = {
                "v1": score_v1(sub_g, [preds["v1"][i] for i in idx]),
                "llm": score_multiclass(sub_g, [preds["llm"][i] for i in idx], nb, seed),
                "v2": score_multiclass(sub_g, [preds["v2"][i] for i in idx], nb, seed),
            }
        for dim, _, sets, _ in DIMENSIONS:
            if set_name not in sets:
                continue
            values = sorted({str(r.get(dim)) for r in gold if r.get(dim) not in (None, "")})
            result["lat_cat"][dim][set_name] = {}
            for val in values:
                idx = [i for i, r in enumerate(gold) if str(r.get(dim)) == val]
                sub_g = [gold[i] for i in idx]
                result["lat_cat"][dim][set_name][val] = {
                    "n_an_toan": sum(str(r["nhan"]) == "AN_TOAN" for r in sub_g),
                    "v1": score_v1(sub_g, [preds["v1"][i] for i in idx]),
                    "llm": score_multiclass(sub_g, [preds["llm"][i] for i in idx], 0, seed),
                    "v2": score_multiclass(sub_g, [preds["v2"][i] for i in idx], 0, seed),
                }
    return result


def slice_sections(res: dict[str, Any]) -> list[str]:
    L = [
        "\n## Theo các lát cắt khác\n",
        "Mỗi ô tính lại trực tiếp trên tập con. Macro-F1 chỉ lấy trung bình trên các nhãn có mặt trong lát cắt. "
        "FPR để trống (–) khi lát cắt không có câu an toàn (vd. lộ liễu/ẩn ý chỉ gồm câu nguy hại). "
        "Thông tin số lượt, độ lộ, kênh, người viết của câu C1/C2 lấy từ bản gốc C0 qua `goc_id`. "
        "Lát cắt nhỏ (n dưới khoảng 50) dao động mạnh, cần đọc kèm n.\n",
    ]
    head = ("| Tập | Giá trị | n | n an toàn | v1 Bin-F1 | v1 FPR | LLM Macro-F1 | LLM Bin-F1 | LLM FPR | LLM BC F1 "
            "| v2 Macro-F1 | v2 Bin-F1 | v2 FPR | v2 BC F1 |")
    for dim, title, sets, names in DIMENSIONS:
        L.append(f"### {title} (`{dim}`)\n")
        L.append(head)
        L.append("|" + "|".join([" --- "] * 14) + "|")
        for s in sets:
            block = res["lat_cat"][dim].get(s, {})
            order = list(names) if names else sorted(block)
            for val in [v for v in order if v in block]:
                c = block[val]
                has_safe = c["n_an_toan"] > 0
                fpr = (lambda m: fmt(m["fpr_an_toan"], "pct") if has_safe else "–")
                v1, llm, v2 = c["v1"], c["llm"], c["v2"]
                label = names.get(val, val) if names else f"`{val}`"
                L.append(
                    f"| {s} | {label} | {v1['n']} | {c['n_an_toan']} | {fmt(v1['binary_f1'])} | {fpr(v1)} | "
                    f"{fmt(llm['macro_f1'])} | {fmt(llm['binary_f1'])} | {fpr(llm)} | {fmt(llm['bc_f1_theo_y'])} | "
                    f"{fmt(v2['macro_f1'])} | {fmt(v2['binary_f1'])} | {fpr(v2)} | {fmt(v2['bc_f1_theo_y'])} |"
                )
        L.append("")
    return L


def fmt(x: Any, kind: str = "num") -> str:
    if x is None:
        return "–"
    if kind == "int":
        return str(x)
    if kind == "pct":
        return f"{x * 100:.1f}%".replace(".", ",")
    return f"{x:.4f}".replace(".", ",")


COLUMNS = (
    ("Macro-F1", "macro_f1", "num"), ("Binary-F1", "binary_f1", "num"), ("Precision", "precision", "num"),
    ("Recall", "recall", "num"), ("FPR an toàn", "fpr_an_toan", "pct"), ("n BC", "n_bc", "int"),
    ("BC P", "bc_p_theo_y", "num"),
    ("BC R", "bc_r_theo_y", "num"), ("BC F1 theo ý", "bc_f1_theo_y", "num"), ("BC F1 mức từ", "bc_f1_muc_tu", "num"),
)


def markdown(res: dict[str, Any], meta: dict[str, Any]) -> str:
    L = [
        "# So sánh 3 hệ trên bộ test v2: v1, gpt-oss-120b và v2\n",
        "> Sinh tự động bằng `python -m src.v2.eval.so_sanh_3_he`. Không sửa tay. "
        "Số đầy đủ: [`metrics.json`](metrics.json).\n",
        "- **v1:** SafeViChi vòng loại (ViHateT5, HATE/CLEAN), dự đoán `results/v2/v1_baseline/`. Chỉ có nhãn nhị phân, "
        "không có bằng chứng, nên các cột 5 nhãn và bằng chứng để trống (–).",
        "- **gpt-oss-120b:** LLM lớn qua API, dự đoán `results/v2/llm_api/` (lượt chạy 24/11, không gọi lại).",
        f"- **v2:** dự đoán `{meta['v2_pred']}` (sha256 `{meta['v2_sha256'][:12]}…`).",
        "- Bộ chấm: `src/v2/eval/metrics.py` hiện tại (bằng chứng tính theo `he_so_phu` mặc định). "
        f"Mỗi ô tính lại trực tiếp trên tập con. Khoảng tin cậy 95% bootstrap {meta['n_bootstrap']} lần, seed {meta['seed']}.\n",
        "**Chú thích cột:** Binary-F1, Precision, Recall, FPR: nhị phân nguy hại/an toàn (FPR là tỉ lệ câu an toàn bị báo "
        "nguy hại). BC P/R/F1 theo ý: precision/recall/F1 của bằng chứng tô sáng theo ý; BC F1 mức từ: F1 bằng chứng "
        "theo từng từ.\n",
        "**Lưu ý bằng chứng:** theo bộ chấm hiện tại (commit `891945a`), điểm bằng chứng chỉ tính trên các câu nguy hại "
        "có cụm gán **và** được hệ báo một nhãn nguy hại; câu nguy hại bị đoán `AN_TOAN` đã bị phạt ở phần phân loại "
        "nên bị loại khỏi điểm bằng chứng. Cột **n BC** là số câu được chấm bằng chứng: hệ có recall thấp hơn được "
        "chấm bằng chứng trên ít câu hơn, nên so sánh cột BC cần đọc kèm Recall và n BC.\n",
    ]
    for s in SETS:
        n_all = res[s]["tat_ca"]["v2"]["n"]
        L.append(f"## {s}: {SET_DESC[s]} ({n_all} mẫu)\n")
        L.append("| Hệ | Vùng | n | " + " | ".join(c[0] for c in COLUMNS) + " |")
        L.append("|" + "|".join([" --- "] * (3 + len(COLUMNS))) + "|")
        for sys_key, sys_name in SYSTEMS:
            for reg, reg_name in REGIONS:
                m = res[s][reg][sys_key]
                cells = [fmt(m.get(k), kind) for _, k, kind in COLUMNS]
                name = f"**{sys_name}**" if reg == "tat_ca" else ""
                rn = f"**{reg_name}**" if reg == "tat_ca" else reg_name
                L.append(f"| {name} | {rn} | {m['n']} | " + " | ".join(cells) + " |")
        L.append("")
        ci_rows = []
        for sys_key, sys_name in SYSTEMS[1:]:
            ci = res[s]["tat_ca"][sys_key].get("ci95", {})
            if ci:
                ci_rows.append(f"| {sys_name} | {fmt(ci['macro_f1']['low'])} – {fmt(ci['macro_f1']['high'])} | "
                               f"{fmt(ci['binary_f1']['low'])} – {fmt(ci['binary_f1']['high'])} |")
        if ci_rows:
            L += ["Khoảng tin cậy 95% (toàn tập):\n", "| Hệ | Macro-F1 | Binary-F1 |", "| --- | --- | --- |", *ci_rows, ""]

    L.append("## Tóm tắt theo tập (toàn bộ vùng)\n")
    L.append("| Hệ | " + " | ".join(f"{s} Macro-F1 | {s} Binary-F1 | {s} BC F1 theo ý" for s in SETS) + " |")
    L.append("|" + "|".join([" --- "] * (1 + 3 * len(SETS))) + "|")
    for sys_key, sys_name in SYSTEMS:
        cells = []
        for s in SETS:
            m = res[s]["tat_ca"][sys_key]
            cells += [fmt(m.get("macro_f1")), fmt(m.get("binary_f1")), fmt(m.get("bc_f1_theo_y"))]
        L.append(f"| {sys_name} | " + " | ".join(cells) + " |")

    L.append("\n## Chênh lệch giữa vùng (cao nhất − thấp nhất)\n")
    L.append("| Hệ | Độ đo | C0 | C1 | C2 |")
    L.append("| --- | --- | --- | --- | --- |")
    for sys_key, sys_name in SYSTEMS:
        for label, key in (("Macro-F1", "macro_f1"), ("Binary-F1", "binary_f1")):
            if res["C0"]["bac"][sys_key].get(key) is None:
                continue
            gaps = []
            for s in SETS:
                vals = [res[s][r][sys_key][key] for r in ("bac", "trung", "nam")]
                gaps.append(fmt(max(vals) - min(vals)))
            L.append(f"| {sys_name} | {label} | " + " | ".join(gaps) + " |")

    L.append("\nMỗi vùng chỉ có khoảng 32 câu an toàn ở C0 và 64 ở C1/C2: một câu sai làm FPR theo vùng đổi khoảng "
             "3 (C0) hoặc 1,6 (C1/C2) điểm phần trăm, nên chênh lệch giữa vùng cần đọc kèm n.\n")
    L += per_label_sections(res)
    L += slice_sections(res)
    L += nhan_xet(res, meta)
    return "\n".join(L)


def per_label_sections(res: dict[str, Any]) -> list[str]:
    L = [
        "\n## Theo từng nhãn\n",
        "Precision/Recall/F1 là độ đo một-nhãn-so-với-phần-còn-lại trong bài toán 5 nhãn. Cột BC tính trên các câu có "
        "nhãn đúng là nhãn đó (cùng quy tắc loại câu bị đoán `AN_TOAN` như trên). v1 không có 5 nhãn nên chỉ có "
        "**tỉ lệ báo đúng nhị phân**: với nhãn nguy hại là tỉ lệ câu bị báo HATE, với `AN_TOAN` là tỉ lệ câu được giữ "
        "CLEAN (ghi ở cột Recall, không so trực tiếp được với Recall 5 nhãn).\n",
    ]
    for s in SETS:
        L.append(f"### {s}: {SET_DESC[s]}\n")
        L.append("| Hệ | Nhãn | Support | Precision | Recall | F1 | n BC | BC F1 theo ý | BC F1 mức từ |")
        L.append("|" + "|".join([" --- "] * 9) + "|")
        for sys_key, sys_name in SYSTEMS:
            m = res[s]["tat_ca"][sys_key]
            for j, lbl in enumerate(LABELS):
                t = m["theo_nhan"][lbl]
                bc = m.get("bc_theo_nhan", {}).get(lbl, {})
                name = f"**{sys_name}**" if j == 0 else ""
                L.append(f"| {name} | `{lbl}` | {t['support']} | {fmt(t.get('precision'))} | {fmt(t['recall'])} | "
                         f"{fmt(t.get('f1'))} | {fmt(bc.get('n_bc'), 'int')} | {fmt(bc.get('bc_f1_theo_y'))} | "
                         f"{fmt(bc.get('bc_f1_muc_tu'))} |")
        L.append("")
        L.append(f"F1 theo nhãn và vùng ({s}; dòng v1 là tỉ lệ báo đúng nhị phân):\n")
        L.append("| Hệ | Nhãn | Tất cả | Bắc | Trung | Nam |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for sys_key, sys_name in SYSTEMS:
            key = "recall" if sys_key == "v1" else "f1"
            for j, lbl in enumerate(LABELS):
                name = f"**{sys_name}**" if j == 0 else ""
                vals = [fmt(res[s][r][sys_key]["theo_nhan"][lbl][key]) for r, _ in REGIONS]
                L.append(f"| {name} | `{lbl}` | " + " | ".join(vals) + " |")
        L.append("")
        for sys_key, sys_name in SYSTEMS[1:]:
            cm = res[s]["tat_ca"][sys_key]["ma_tran_nham"]
            L.append(f"Ma trận nhầm lẫn {sys_name}, {s} (hàng: nhãn thật, cột: nhãn đoán):\n")
            L.append("| Thật \\ Đoán | " + " | ".join(f"`{l}`" for l in cm["labels"]) + " |")
            L.append("|" + "|".join([" --- "] * (1 + len(cm["labels"]))) + "|")
            for lbl, row in zip(cm["labels"], cm["matrix"]):
                L.append(f"| `{lbl}` | " + " | ".join(str(v) for v in row) + " |")
            L.append("")
    return L


# Phần nhận xét được viết sau khi đọc kết quả của đúng file dự đoán v2 này. Số trong nhận xét vẫn đọc từ kết quả,
# nhưng các câu so sánh (cao hơn, yếu nhất...) chỉ đúng với lần chạy này: đổi file dự đoán thì phải đọc và viết lại.
NHAN_XET_SHA256 = "dfe1d08a9b8af3cd57e1279ec50176a8d49770493e0c3344371322553ba21d3e"


def nhan_xet(res: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    if meta["v2_sha256"] != NHAN_XET_SHA256:
        return ["\n## Nhận xét\n",
                "> File dự đoán v2 đã khác file lúc viết nhận xét (sha256 khác `NHAN_XET_SHA256` trong script). "
                "Nhận xét cũ đã bị ẩn; cần đọc lại bảng và viết lại phần này.\n"]

    def a(s, sys_key, key, reg="tat_ca"):
        return res[s]["tat_ca" if reg is None else reg][sys_key][key]

    def ci(s, sys_key, key):
        c = res[s]["tat_ca"][sys_key]["ci95"][key]
        return f"[{fmt(c['low'])}; {fmt(c['high'])}]"

    def sl(dim, s, val, sys_key, key):
        return res["lat_cat"][dim][s][val][sys_key][key]

    def cm_cell(cm, true_lbl, pred_lbl):
        return cm["matrix"][cm["labels"].index(true_lbl)][cm["labels"].index(pred_lbl)]

    sms = res["lat_cat"]["kenh"]["C0"]["sms"]
    sms_cm = sms["v2"]["ma_tran_nham"]
    sms_safe = sms["n_an_toan"]
    sms_fp_ld, sms_fp_dd = cm_cell(sms_cm, "AN_TOAN", "LUA_DAO"), cm_cell(sms_cm, "AN_TOAN", "DE_DOA")
    sms_qr = sum(sms_cm["matrix"][sms_cm["labels"].index("QUAY_ROI")])
    sms_qr_ok = cm_cell(sms_cm, "QUAY_ROI", "QUAY_ROI")
    ks = res["lat_cat"]["phan_test"]
    kc = ks["C0"]["chua_thay"]["v2"]["ma_tran_nham"]
    k2 = res["lat_cat"]["kieu_bien_dang"]["C2"]
    k1 = res["lat_cat"]["kieu_bien_dang"]["C1"]
    fpr_kinds = sorted(((k, v["v2"]["fpr_an_toan"], s) for s, blk in (("C1", k1), ("C2", k2)) for k, v in blk.items()),
                       key=lambda x: -x[1])[:3]
    nb = res["lat_cat"]["nhom_bien_dang"]["C1"]
    reg_fpr = {r: [a(s, "v2", "fpr_an_toan", r) for s in SETS] for r in ("bac", "trung", "nam")}
    v1_c0 = res["C0"]["tat_ca"]["v1"]
    v1_lbl = v1_c0["theo_nhan"]
    D = res["lat_cat"]["nguoi_viet"]

    L = [
        "\n## Nhận xét\n",
        "### 1. So sánh chung\n",
        f"- **v2 tốt nhất trên cả ba tập.** Macro-F1 của v2 là {fmt(a('C0','v2','macro_f1'))} {ci('C0','v2','macro_f1')} "
        f"ở C0, {fmt(a('C1','v2','macro_f1'))} {ci('C1','v2','macro_f1')} ở C1 và {fmt(a('C2','v2','macro_f1'))} "
        f"{ci('C2','v2','macro_f1')} ở C2; của gpt-oss-120b lần lượt là {fmt(a('C0','llm','macro_f1'))} "
        f"{ci('C0','llm','macro_f1')}, {fmt(a('C1','llm','macro_f1'))} {ci('C1','llm','macro_f1')} và "
        f"{fmt(a('C2','llm','macro_f1'))} {ci('C2','llm','macro_f1')}. Khoảng tin cậy của hai hệ không chồng nhau ở cả ba tập.",
        f"- **v2 giữ điểm khi câu bị biến dạng.** Binary-F1 của v2 chỉ giảm từ {fmt(a('C0','v2','binary_f1'))} (C0) xuống "
        f"{fmt(a('C2','v2','binary_f1'))} (C2); của gpt-oss-120b giảm từ {fmt(a('C0','llm','binary_f1'))} xuống "
        f"{fmt(a('C2','llm','binary_f1'))}. Với bằng chứng, F1 theo ý của v2 từ {fmt(a('C0','v2','bc_f1_theo_y'))} xuống "
        f"{fmt(a('C2','v2','bc_f1_theo_y'))}, còn gpt-oss-120b từ {fmt(a('C0','llm','bc_f1_theo_y'))} xuống "
        f"{fmt(a('C2','llm','bc_f1_theo_y'))}.",
        f"- **v1 gần như chỉ bắt được xúc phạm.** Ở C0, v1 báo HATE cho {fmt(v1_lbl['XUC_PHAM']['recall'],'pct')} số "
        f"câu `XUC_PHAM`, nhưng chỉ {fmt(v1_lbl['DE_DOA']['recall'],'pct')} câu `DE_DOA`, "
        f"{fmt(v1_lbl['QUAY_ROI']['recall'],'pct')} câu `QUAY_ROI` và {fmt(v1_lbl['LUA_DAO']['recall'],'pct')} câu `LUA_DAO`. Precision {fmt(v1_c0['precision'])} chỉ phản "
        "ánh việc v1 hiếm khi báo HATE. Đây là lý do đổi từ HATE/CLEAN sang 5 nhãn theo tình huống.",
        f"- **Đổi lại, v2 báo nhầm câu an toàn nhiều hơn gpt-oss-120b.** FPR của v2 là {fmt(a('C0','v2','fpr_an_toan'),'pct')} "
        f"ở C0 so với {fmt(a('C0','llm','fpr_an_toan'),'pct')} của gpt-oss-120b. gpt-oss-120b thận trọng hơn: precision "
        f"nguy hại {fmt(a('C0','llm','precision'))} nhưng recall chỉ {fmt(a('C0','llm','recall'))}.\n",
        "### 2. Điểm yếu của v2 (yêu cầu 1 và 2)\n",
        f"- **Kịch bản chưa thấy khi train là điểm yếu lớn nhất.** Ở C0, Macro-F1 của v2 là "
        f"{fmt(sl('phan_test','C0','trong_phan_phoi','v2','macro_f1'))} với kịch bản trong phân phối nhưng chỉ "
        f"{fmt(sl('phan_test','C0','chua_thay','v2','macro_f1'))} với kịch bản chưa thấy; FPR tăng từ "
        f"{fmt(sl('phan_test','C0','trong_phan_phoi','v2','fpr_an_toan'),'pct')} lên "
        f"{fmt(sl('phan_test','C0','chua_thay','v2','fpr_an_toan'),'pct')}. Lỗi chủ yếu là lẫn giữa `DE_DOA` và `AN_TOAN`: "
        f"{cm_cell(kc,'DE_DOA','AN_TOAN')} câu đe dọa bị cho là an toàn và {cm_cell(kc,'AN_TOAN','DE_DOA')} câu an toàn "
        "bị báo đe dọa. Mức tụt tương tự ở C1 và C2, và gpt-oss-120b cũng tụt ở phần này, nên đây là các kịch bản khó "
        "thật chứ không chỉ do v2 học thuộc.",
        f"- **Báo nhầm câu an toàn khá cao.** FPR của v2 khoảng {fmt(a('C0','v2','fpr_an_toan'),'pct')}–"
        f"{fmt(a('C1','v2','fpr_an_toan'),'pct')} trên ba tập, cao nhất ở vùng Nam "
        f"({', '.join(fmt(x,'pct') for x in reg_fpr['nam'])} ở C0/C1/C2) và thấp nhất ở vùng Trung "
        f"({', '.join(fmt(x,'pct') for x in reg_fpr['trung'])}). Đây là chỗ cần hiệu chỉnh ngưỡng trên validation.",
        "- **Ba kiểu biến dạng gây báo nhầm nhiều nhất:** "
        + ", ".join(f"`{k}` ({s}, FPR {fmt(v,'pct')})" for k, v, s in fpr_kinds)
        + ". Văn bản bị chèn khoảng trắng, gõ sai hay chèn ký tự vô hình khiến v2 nghiêng về nhãn nguy hại.",
        f"- **Tin SMS:** Macro-F1 của v2 ở SMS (C0) là {fmt(sms['v2']['macro_f1'])}, thấp hơn gpt-oss-120b "
        f"({fmt(sms['llm']['macro_f1'])}). Trong {sms_safe} tin SMS an toàn, v2 báo nhầm {sms_fp_ld} tin thành `LUA_DAO` "
        f"và {sms_fp_dd} tin thành `DE_DOA`; trong {sms_qr} tin quấy rối qua SMS chỉ nhận đúng {sms_qr_ok}. Lát cắt này "
        f"nhỏ ({sms['v1']['n']} câu) nên cần xem lại từng câu trước khi kết luận.",
        f"- **`QUAY_ROI` và `DE_DOA` là hai nhãn yếu nhất của v2.** F1 ở C2 lần lượt là "
        f"{fmt(res['C2']['tat_ca']['v2']['theo_nhan']['QUAY_ROI']['f1'])} và "
        f"{fmt(res['C2']['tat_ca']['v2']['theo_nhan']['DE_DOA']['f1'])}, so với "
        f"{fmt(res['C2']['tat_ca']['v2']['theo_nhan']['LUA_DAO']['f1'])} của `LUA_DAO`.\n",
        "### 3. Vì sao C2 không thấp hơn C1\n",
        "C1 và C2 làm hỏng câu theo hai cách khác nhau. C1 làm hỏng chính các từ (bỏ dấu, teencode, gõ sai), nên thông "
        "tin mất thật: \"đánh\" và \"dành\" bỏ dấu đều thành \"danh\". C2 chỉ đổi lớp ký tự Unicode (fullwidth, ký tự vô "
        "hình, chữ Cyrillic giống chữ Latin), các từ vẫn còn nguyên và mô hình đọc lại được. \"Chưa thấy\" nghĩa là nằm "
        "ngoài dữ liệu train của v2, không có nghĩa là khó hơn.\n",
        f"- Với v2, nhóm C1 giữ ngoài train còn có Macro-F1 cao hơn nhóm đã thấy "
        f"({fmt(nb['chua_thay']['v2']['macro_f1'])} so với {fmt(nb['da_thay']['v2']['macro_f1'])}).",
        f"- gpt-oss-120b phân loại tốt với `fullwidth` (Binary-F1 {fmt(k2['fullwidth']['llm']['binary_f1'])}) nhưng "
        f"gần như không trả được bằng chứng đúng (BC F1 {fmt(k2['fullwidth']['llm']['bc_f1_theo_y'])}; "
        f"`zero_width` {fmt(k2['zero_width']['llm']['bc_f1_theo_y'])}): nó đọc hiểu được câu nhưng không chép lại "
        "nguyên văn chuỗi ký tự lạ.",
        f"- Với v1, Binary-F1 tăng ở C2 là ảo: đến chủ yếu từ `space_injection` (Binary-F1 "
        f"{fmt(k2['space_injection']['v1']['binary_f1'])}, FPR {fmt(k2['space_injection']['v1']['fpr_an_toan'],'pct')}). "
        "Văn bản bị băm nát khiến v1 báo HATE nhiều hơn, kể cả với câu an toàn.\n",
        "### 4. Lưu ý khi đọc bảng\n",
        f"- **Người viết D chỉ viết câu nguy hại** ({D['C0']['D']['v1']['n']} câu C0, không có câu an toàn), nên điểm gần 1 "
        "của nhóm D không so được với A, B, C.",
        "- **Cột n BC:** điểm bằng chứng chỉ tính trên câu nguy hại mà hệ đã báo nguy hại, nên hệ có recall thấp được "
        f"chấm bằng chứng trên ít câu hơn (C0: v2 {a('C0','v2','n_bc')} câu, gpt-oss-120b {a('C0','llm','n_bc')} câu).",
        "- **Lát cắt nhỏ:** vùng, kênh, người viết C, kiểu biến dạng chỉ có vài chục câu an toàn mỗi nhóm; chênh lệch vài "
        "điểm phần trăm FPR có thể chỉ là một hai câu. Khoảng tin cậy chỉ có cho toàn tập.",
        "- **Dự đoán gpt-oss-120b** là lượt chạy API ngày 24/11, không chạy lại; v1 và gpt-oss-120b chưa được dò ngưỡng "
        "riêng trên validation.\n",
    ]
    return L


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--v2-pred", type=Path,
        default=Path("results/v2/official_eval/dualorc_mmbert_base/test/real_test_all_predictions.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    res = run(args.v2_pred, args.n_bootstrap, args.seed)
    meta = {
        "v2_pred": args.v2_pred.as_posix(), "v2_sha256": sha256(args.v2_pred),
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
        "lenh": "python -m src.v2.eval.so_sanh_3_he " + " ".join(argv if argv is not None else sys.argv[1:]),
        "python": sys.version.split()[0], "may": platform.platform(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "ket_qua": res}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "KET_QUA.md").write_text(markdown(res, meta), encoding="utf-8")
    print(f"Đã ghi {args.out_dir / 'metrics.json'} và {args.out_dir / 'KET_QUA.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
