"""Sinh các file markdown kết quả baseline (gpt-oss-120b qua API và SafeViChi v1) từ file metrics.

Mọi con số trong markdown đều đọc từ `results/v2/llm_api/cham_lai/*_metrics.json` và
`results/v2/v1_baseline/cham_lai/*_metrics.json`; không gõ tay. Chạy lại toàn bộ:

    # 1. Đổi tên trường dự đoán LLM rồi chấm bằng bộ chấm v2
    python -m src.v2.eval.chuyen_du_doan_llm --src results/v2/llm_api/test_predictions.jsonl \
        --dst results/v2/llm_api/cham_lai/C0_predictions.jsonl
    python -m src.v2.eval.evaluate --gold data/v2/test_day_du.jsonl --meta data/v2/test_day_du.jsonl \
        --pred results/v2/llm_api/cham_lai/C0_predictions.jsonl --out results/v2/llm_api/cham_lai/C0_metrics.json
    # (tương tự C1, C2 với test_perturb_C1/C2_predictions.jsonl và data/v2/test_perturb_C1/C2.jsonl)

    # 2. Chấm v1 (nhị phân)
    python -m src.v2.eval.cham_v1 --pred results/v2/v1_baseline/test_v1_predictions.jsonl \
        --meta data/v2/test_day_du.jsonl --out results/v2/v1_baseline/cham_lai/C0_metrics.json
    # (tương tự C1, C2 với --meta data/v2/test_perturb_C1/C2.jsonl)

    # 3. Sinh markdown
    python -m src.v2.eval.bao_cao_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SETS = ("C0", "C1", "C2")
REGIONS = (("bac", "Bắc"), ("trung", "Trung"), ("nam", "Nam"))
HARMFUL = ("XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
LABELS = ("AN_TOAN", *HARMFUL)
C1_KINDS = ("bo_dau", "teencode", "go_sai", "dinh_tu", "so_thay_chu", "keo_dai", "chen_ky_tu")
C2_KINDS = ("fullwidth", "zero_width", "space_injection", "exotic_separator", "cyrillic_homoglyph", "double_elongation")
SET_DESC = {"C0": "bản gốc", "C1": "biến dạng đã biết", "C2": "kiểu điểm mù"}
LLM_DIR = Path("results/v2/llm_api/cham_lai")
V1_DIR = Path("results/v2/v1_baseline/cham_lai")
COMPARE_PATH = Path("results/v2/SO_SANH_BASELINE.md")
SCRIPT_NOTE = "> Tệp này sinh tự động bằng `python -m src.v2.eval.bao_cao_baseline`. Không sửa tay.\n"


def load(directory: Path) -> dict[str, dict[str, Any]]:
    return {s: json.loads((directory / f"{s}_metrics.json").read_text(encoding="utf-8")) for s in SETS}


def f(x: float, nd: int = 4) -> str:
    """Định dạng số theo kiểu Việt Nam (dấu phẩy thập phân)."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([" --- "] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def llm_readme(m: dict[str, dict[str, Any]]) -> str:
    cls = {s: m[s]["classification"] for s in SETS}
    ev = {s: m[s]["evidence"] for s in SETS}
    b = {s: cls[s]["binary_harmful_vs_safe"] for s in SETS}
    ci = {s: cls[s]["bootstrap_95"] for s in SETS}
    out = [
        "# Baseline LLM lớn qua API: gpt-oss-120b trên bộ test v2\n",
        SCRIPT_NOTE,
        "Chấm lại ngày 25/11/2026 bằng bộ chấm hiện tại (`src/v2/eval/`, sau commit `20989e4` đổi cách tính "
        "bằng chứng). Dự đoán lấy nguyên từ lượt chạy API ngày 24/11 (`results/v2/llm_api/*_predictions.jsonl`), "
        "không gọi API lại; chỉ đổi tên trường `nhan_llm`, `bang_chung_llm` thành `nhan`, `bang_chung` "
        "(`src/v2/eval/chuyen_du_doan_llm.py`). Các file `results/v2/llm_api/test*_metrics.json` cũ giữ lại vì "
        "còn log lượt gọi API, nhưng **số bằng chứng trong đó tính theo bộ chấm cũ, không dùng cho báo cáo.**\n",
        "## Phân loại và bằng chứng\n",
        table(
            ["Chỉ số"] + [f"{s} ({cls[s]['n']} mẫu)" for s in SETS],
            [
                ["Macro-F1 5 nhãn"] + [f"{f(cls[s]['macro_f1'])} [{f(ci[s]['macro_f1']['low'])}; {f(ci[s]['macro_f1']['high'])}]" for s in SETS],
                ["Binary-F1 nguy hại/an toàn"] + [f"{f(b[s]['f1'])} [{f(ci[s]['binary_f1']['low'])}; {f(ci[s]['binary_f1']['high'])}]" for s in SETS],
                ["Precision nguy hại"] + [f(b[s]["precision"]) for s in SETS],
                ["Recall nguy hại"] + [f(b[s]["recall"]) for s in SETS],
                ["FPR câu an toàn"] + [pct(b[s]["fpr_an_toan"]) for s in SETS],
                ["Bằng chứng: precision theo ý"] + [f(ev[s]["f1_theo_y"]["precision"]) for s in SETS],
                ["Bằng chứng: recall theo ý"] + [f(ev[s]["f1_theo_y"]["recall"]) for s in SETS],
                ["**Bằng chứng: F1 theo ý**"] + [f"**{f(ev[s]['f1_theo_y']['f1'])}**" for s in SETS],
                ["Bằng chứng: F1 mức từ"] + [f(ev[s]["f1_muc_tu"]["f1"]) for s in SETS],
                ["Câu nguy hại không có bằng chứng"] + [pct(ev[s]["harmful_empty_evidence_rate"]) for s in SETS],
                ["Tỉ lệ đầu ra hợp lệ"] + [pct(m[s]["output_reliability"]["valid_output_rate"]) for s in SETS],
            ],
        ),
        "\nKhoảng trong ngoặc vuông là khoảng tin cậy 95% bootstrap (1.000 lần, seed 42).\n",
        "## F1 theo nhãn\n",
        table(
            ["Nhãn"] + list(SETS),
            [[lbl] + [f(cls[s]["per_class"][lbl]["f1"]) for s in SETS] for lbl in LABELS],
        ),
        "\n## Theo giọng vùng\n",
        table(
            ["Tập", "Vùng", "n", "Macro-F1", "Binary-F1", "FPR an toàn", "BC F1 theo ý"],
            [
                [s, name, str(g["n"]), f(g["macro_f1"]), f(g["binary_f1"]), pct(g["fpr_an_toan"]), f(g["evidence_f1_theo_y"])]
                for s in SETS
                for key, name in REGIONS
                for g in [m[s]["slices"]["giong_vung"][key]]
            ],
        ),
        "\n" + table(
            ["Tập", "Vùng yếu nhất (Macro-F1)", "Chênh lệch cao nhất − thấp nhất"],
            [[s, f(m[s]["slices"]["giong_vung"]["worst_group_macro_f1"]),
              f(m[s]["slices"]["giong_vung"]["region_gap_max_min"])] for s in SETS],
        ),
        "\nMỗi vùng chỉ có khoảng 32 câu an toàn ở C0 (khoảng 64 ở C1/C2), nên một câu sai làm FPR theo vùng "
        "đổi khoảng 3 điểm phần trăm ở C0. Chênh lệch giữa vùng cần đọc kèm n.\n",
        "## Theo kiểu biến dạng\n",
        kind_table_llm(m),
        "",
    ]
    return "\n".join(out)


def kind_table_llm(m: dict[str, dict[str, Any]]) -> str:
    rows = []
    for s, kinds in (("C1", C1_KINDS), ("C2", C2_KINDS)):
        sl = m[s]["slices"]["kieu_bien_dang"]
        for k in kinds:
            g = sl[k]
            rows.append([s, f"`{k}`", str(g["n"]), f(g["macro_f1"]), f(g["binary_f1"]), pct(g["fpr_an_toan"]), f(g["evidence_f1_theo_y"])])
    return table(["Tập", "Kiểu", "n", "Macro-F1", "Binary-F1", "FPR an toàn", "BC F1 theo ý"], rows)


def v1_readme(m: dict[str, dict[str, Any]]) -> str:
    b = {s: m[s]["binary_harmful_vs_safe"] for s in SETS}
    out = [
        "# Baseline SafeViChi v1 trên bộ test v2\n",
        SCRIPT_NOTE,
        "v1 (ViHateT5 fine-tune đối kháng FGM, vòng loại) chỉ trả HATE/CLEAN và không trả bằng chứng, nên chỉ chấm "
        "**nhị phân nguy hại/an toàn**: HATE ứng với 4 nhãn nguy hại của v2, CLEAN ứng với `AN_TOAN`. "
        "Không có Macro-F1 5 nhãn và không có số bằng chứng. Script: `src/v2/eval/cham_v1.py`; dự đoán: "
        "`results/v2/v1_baseline/*_predictions.jsonl`.\n",
        "## Nhị phân nguy hại/an toàn\n",
        table(
            ["Chỉ số"] + [f"{s} ({b[s]['n']} mẫu)" for s in SETS],
            [
                ["Precision"] + [f(b[s]["precision"]) for s in SETS],
                ["**Recall**"] + [f"**{f(b[s]['recall'])}**" for s in SETS],
                ["Binary-F1"] + [f(b[s]["f1"]) for s in SETS],
                ["FPR câu an toàn"] + [pct(b[s]["fpr_an_toan"]) for s in SETS],
                ["TP / FP / FN / TN"] + [f"{b[s]['tp']} / {b[s]['fp']} / {b[s]['fn']} / {b[s]['tn']}" for s in SETS],
            ],
        ),
        f"\nv1 gán CLEAN cho {b['C0']['fn'] + b['C0']['tn']}/{b['C0']['n']} câu C0, nên precision cao "
        "chỉ phản ánh việc nó hiếm khi báo HATE.\n",
        "## Recall theo nhãn v2\n",
        table(["Nhãn"] + list(SETS), [[lbl] + [f(m[s]["recall_theo_nhan"][lbl]["recall"]) for s in SETS] for lbl in HARMFUL]),
        "\nv1 được huấn luyện cho hate speech nên chủ yếu bắt được `XUC_PHAM`; `QUAY_ROI` và `LUA_DAO` gần như bị bỏ qua.\n",
        "## Theo giọng vùng\n",
        table(
            ["Tập", "Vùng", "n", "Precision", "Recall", "Binary-F1", "FPR an toàn"],
            [
                [s, name, str(g["n"]), f(g["precision"]), f(g["recall"]), f(g["f1"]), pct(g["fpr_an_toan"])]
                for s in SETS
                for key, name in REGIONS
                for g in [m[s]["giong_vung"][key]]
            ],
        ),
        "\n" + table(
            ["Tập", "Vùng yếu nhất (Binary-F1)", "Chênh lệch cao nhất − thấp nhất"],
            [[s, f(m[s]["giong_vung"]["worst_group_binary_f1"]), f(m[s]["giong_vung"]["region_gap_binary_f1"])] for s in SETS],
        ),
        "\n## Theo kiểu biến dạng\n",
        kind_table_v1(m),
        "",
    ]
    return "\n".join(out)


def kind_table_v1(m: dict[str, dict[str, Any]]) -> str:
    rows = []
    for s, kinds in (("C1", C1_KINDS), ("C2", C2_KINDS)):
        for k in kinds:
            g = m[s]["kieu_bien_dang"][k]
            rows.append([s, f"`{k}`", str(g["n"]), f(g["recall"]), f(g["f1"]), f"{g['fp']}/{g['fp'] + g['tn']}"])
    return table(["Tập", "Kiểu", "n", "Recall", "Binary-F1", "Báo nhầm câu an toàn"], rows)


def compare_md(llm: dict[str, dict[str, Any]], v1: dict[str, dict[str, Any]]) -> str:
    lb = {s: llm[s]["classification"]["binary_harmful_vs_safe"] for s in SETS}
    vb = {s: v1[s]["binary_harmful_vs_safe"] for s in SETS}
    lk2, vk2 = llm["C2"]["slices"]["kieu_bien_dang"], v1["C2"]["kieu_bien_dang"]
    vk_si, lk_si = vk2["space_injection"], lk2["space_injection"]
    other_fp = sum(vk2[k]["fp"] for k in C2_KINDS if k != "space_injection")
    out = [
        "# So sánh baseline trên bộ test v2: gpt-oss-120b (API) và SafeViChi v1\n",
        SCRIPT_NOTE,
        "Chi tiết: [`llm_api/cham_lai/README.md`](llm_api/cham_lai/README.md), "
        "[`v1_baseline/cham_lai/README.md`](v1_baseline/cham_lai/README.md). "
        "v1 chỉ có nhãn nhị phân nên bảng dưới so sánh nhị phân nguy hại/an toàn.\n",
        "## Nhị phân nguy hại/an toàn\n",
        table(
            ["Chỉ số", "v1 C0", "v1 C1", "v1 C2", "LLM C0", "LLM C1", "LLM C2"],
            [
                ["Precision"] + [f(vb[s]["precision"]) for s in SETS] + [f(lb[s]["precision"]) for s in SETS],
                ["Recall"] + [f(vb[s]["recall"]) for s in SETS] + [f(lb[s]["recall"]) for s in SETS],
                ["Binary-F1"] + [f(vb[s]["f1"]) for s in SETS] + [f(lb[s]["f1"]) for s in SETS],
                ["FPR an toàn"] + [pct(vb[s]["fpr_an_toan"]) for s in SETS] + [pct(lb[s]["fpr_an_toan"]) for s in SETS],
            ],
        ),
        "\n## Binary-F1 theo giọng vùng\n",
        table(
            ["Hệ", "Tập", "Bắc", "Trung", "Nam"],
            [["v1", s] + [f(v1[s]["giong_vung"][k]["f1"]) for k, _ in REGIONS] for s in SETS]
            + [["LLM", s] + [f(llm[s]["slices"]["giong_vung"][k]["binary_f1"]) for k, _ in REGIONS] for s in SETS],
        ),
        "\n## Vì sao Binary-F1 ở C2 cao hơn C1\n",
        "C1 và C2 làm hỏng câu theo hai cách khác nhau; \"chưa thấy\" (C2) nghĩa là nằm ngoài dữ liệu train của "
        "mô hình v2, không có nghĩa là khó hơn. Với hai baseline này, cả C1 và C2 đều là kiểu chưa từng huấn luyện.\n",
        "- **C1 làm hỏng chính các từ** (bỏ dấu, teencode, gõ sai, số thay chữ...): thông tin mất thật, "
        "ví dụ \"đánh\" và \"dành\" bỏ dấu đều thành \"danh\".",
        "- **C2 chỉ đổi lớp ký tự Unicode** (fullwidth, ký tự vô hình, chữ Cyrillic giống chữ Latin...): từ vẫn "
        f"còn nguyên, LLM lớn đọc lại được (Binary-F1 của LLM với `fullwidth`: {f(lk2['fullwidth']['binary_f1'])}). "
        f"Riêng `space_injection` (chèn dấu cách giữa các chữ) làm LLM hụt rõ (Binary-F1 {f(lk_si['binary_f1'])}).",
        f"- **Với v1, mức tăng ở C2 là ảo:** đến chủ yếu từ `space_injection` (recall {f(vk_si['recall'])}, "
        f"báo nhầm {vk_si['fp']}/{vk_si['fp'] + vk_si['tn']} câu an toàn) và `double_elongation` "
        f"(recall {f(vk2['double_elongation']['recall'])}). Văn bản bị băm nát khiến v1 nghiêng về HATE kể cả với "
        f"câu an toàn; năm kiểu C2 còn lại cộng lại chỉ báo nhầm {other_fp} câu.\n",
        "Vì vậy báo cáo nên trình bày theo từng kiểu biến dạng thay vì một con số cho mỗi tập.\n",
        "### Binary-F1 theo kiểu biến dạng\n",
        table(
            ["Tập", "Kiểu", "v1", "LLM", "Báo nhầm an toàn v1", "FPR an toàn LLM"],
            [
                [s, f"`{k}`", f(v1[s]["kieu_bien_dang"][k]["f1"]), f(llm[s]["slices"]["kieu_bien_dang"][k]["binary_f1"]),
                 f"{v1[s]['kieu_bien_dang'][k]['fp']}/{v1[s]['kieu_bien_dang'][k]['fp'] + v1[s]['kieu_bien_dang'][k]['tn']}",
                 pct(llm[s]["slices"]["kieu_bien_dang"][k]["fpr_an_toan"])]
                for s, kinds in (("C1", C1_KINDS), ("C2", C2_KINDS))
                for k in kinds
            ],
        ),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    llm, v1 = load(LLM_DIR), load(V1_DIR)
    outputs = {
        LLM_DIR / "README.md": llm_readme(llm),
        V1_DIR / "README.md": v1_readme(v1),
        COMPARE_PATH: compare_md(llm, v1),
    }
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print(f"Đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
