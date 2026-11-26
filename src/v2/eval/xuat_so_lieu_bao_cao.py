"""Sinh số liệu cho báo cáo LaTeX vòng chung kết từ các file kết quả.

Mọi con số trong `docs/v2/bao_cao/bao_cao_v2.tex` lấy qua lệnh `\\so{khoa}`; script này đọc các file trong
`results/v2/`, `data/v2/` (thống kê) và `v1/results/` rồi ghi `docs/v2/bao_cao/so_lieu.tex`. Không sửa tay file đó.

    python -m src.v2.eval.xuat_so_lieu_bao_cao

Quy ước hiển thị: F1, precision, recall nhân 100, một chữ số thập phân (86,5); tỉ lệ ghi kèm dấu %.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.v2.eval.evaluate import align_predictions, read_jsonl
from src.v2.eval.metrics import evaluate_bundle
from src.v2.eval.so_sanh_3_he import SETS, llm_rows

OUT = Path("docs/v2/bao_cao/so_lieu.tex")
V2_PRED = Path("results/v2/official_eval/dualorc_mmbert_base/test/real_test_all_predictions.jsonl")
NHAN = ("AN_TOAN", "XUC_PHAM", "QUAY_ROI", "DE_DOA", "LUA_DAO")
MA = {"AN_TOAN": "at", "XUC_PHAM": "xp", "QUAY_ROI": "qr", "DE_DOA": "dd", "LUA_DAO": "ld"}

# Ví dụ trích trong báo cáo (id câu test C0)
VI_DU = {
    "dung": "v2_eb409bf76f11",   # DE_DOA giọng Trung, mô hình đúng, bằng chứng khớp
    "otp": "v2_017392d9824a",    # tin OTP thật bị báo LUA_DAO
    "baoco": "v2_3ac6affa681d",  # khuyên báo cô giáo bị báo DE_DOA
    "ghiam": "v2_a07f8517facc",  # dọa ẩn ý (DD6, chưa thấy) bị bỏ sót
    "vay": "v2_05b13c8b2651",    # vay online (LD7, chưa thấy) bị bỏ sót
    "nham": "v2_2fe1b96bfacd",   # chuyển nhầm tiền (kịch bản mới) bị bỏ sót
    "kiemsoat": "v2_dea58234d1b3",  # kiểm soát người yêu (QR6, chưa thấy) bị bỏ sót
}

SOURCES: list[Path] = []


def load(path: str) -> Any:
    p = Path(path)
    SOURCES.append(p)
    return json.loads(p.read_text(encoding="utf-8"))


def sc(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",")


def pc(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + r"\%"


def dl(x: float) -> str:
    s = f"{abs(x) * 100:.1f}".replace(".", ",")
    if s == "0,0":
        return s
    return ("+" if x >= 0 else "$-$") + s


def nn(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def dec(x: float, k: int = 1) -> str:
    return f"{x:.{k}f}".replace(".", ",")


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


class Store:
    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        if key in self.items:
            raise KeyError(f"trùng khóa {key}")
        self.items[key] = value


def bundle(records, preds) -> dict[str, Any]:
    return evaluate_bundle(records, preds, n_bootstrap=0, seed=42, include_slices=False)


def so_chuan_hoa_test(put, base: str) -> None:
    """Mục 8.2: bật/tắt bộ chuẩn hóa v1 trên tập test khóa (dòng #5 thô và dòng #6 có chuẩn hóa)."""
    tho = load(f"{base}/test/real_test_official_matrix.json")
    chn = load(f"{base}/test/real_test_norm_official_matrix.json")
    cap = [(t, tho["tiers"][t], chn["tiers"][t]) for t in ("C0", "C1", "C2")]
    cap.append(("chuathay", tho["strictly_unseen_perturb"], chn["strictly_unseen_perturb"]))
    for k, a, b in cap:
        put(f"cnt.{k}.n", nn(a["n"]))
        put(f"cnt.{k}.raw", sc(a["macro_f1"]))
        put(f"cnt.{k}.norm", sc(b["macro_f1"]))
        put(f"cnt.{k}.gain", dl(b["macro_f1"] - a["macro_f1"]))
        put(f"cnt.{k}.bc.raw", sc(a["evidence_f1_theo_y"]))
        put(f"cnt.{k}.bc.norm", sc(b["evidence_f1_theo_y"]))
        put(f"cnt.{k}.bc.gain", dl(b["evidence_f1_theo_y"] - a["evidence_f1_theo_y"]))
    for kieu, a in tho["by_kieu_bien_dang"].items():
        b = chn["by_kieu_bien_dang"][kieu]
        put(f"cnt.kieu.{kieu.replace('_', '')}.gain", dl(b["macro_f1"] - a["macro_f1"]))


def so_thiet_bi(put, base: str) -> None:
    """Số liệu Mục 8.3: ONNX INT8 trên laptop CPU (2 máy) và trong trình duyệt, kèm năng lượng đo từ pin."""
    # Máy A: lượt đo của nhóm bằng `python -m src.v2.bench.benchmark_cpu` (FP32/INT8/PyTorch, 100 đoạn C0)
    bc = load(f"{base}/test/bench_cpu.json")
    put("tb.a.cpu", tex_escape(bc["environment"]["cpu_name"]))
    put("tb.a.ort", bc["environment"]["onnxruntime"])
    ma = {"onnx_int8": "int8", "onnx_fp32": "fp32", "torch_fp32": "torch"}
    for r in bc["runs"]:
        k = f"tb.a.{ma[r['backend']]}.{r['threads']}"
        put(f"{k}.n", nn(r["n_samples"]))
        put(f"{k}.p50", dec(r["latency_ms"]["end_to_end_p50"], 1))
        put(f"{k}.p95", dec(r["latency_ms"]["end_to_end_p95"], 1))
        put(f"{k}.ram", nn(r["memory_MiB"]["peak_rss"]))
        put(f"{k}.nap", dec(r["cold_load_ms"] / 1000.0, 1))
    sz = bc["sizes_MiB"]
    put("tb.int8.mib", nn(sz["onnx_int8"]))
    put("tb.fp32.mib", nn(sz["onnx_fp32"]))
    put("tb.tok.mib", nn(sz["tokenizer_json"]))
    put("tb.nen", dec(sz["compression_ratio_vs_onnx_fp32"], 1))

    # INT8 so với FP32 trên toàn bộ 1.670 đoạn test (batch 8, bản khóa)
    ict = load(f"{base}/test/int8_comparison.json")["tiers"]
    for t in ("C0", "C1", "C2"):
        put(f"tb.int8.{t}.khop", pc(ict[t]["agreement"]["label_exact_rate"]))
    # Ablation lượng tử hóa FP32 -> INT8 theo từng tầng (Mục 8.2)
    for t in ("C0", "C1", "C2", "ALL"):
        x = ict[t]
        for he in ("fp32", "int8"):
            put(f"lt.{t}.{he}.macro", sc(x[he]["macro_f1"]))
            put(f"lt.{t}.{he}.bcy", sc(x[he]["evidence_f1_theo_y"]))
            put(f"lt.{t}.{he}.fpr", pc(x[he]["fpr_an_toan"]))
        put(f"lt.{t}.d.macro", dl(x["delta_int8_minus_fp32"]["macro_f1"]))
        put(f"lt.{t}.d.bcy", dl(x["delta_int8_minus_fp32"]["evidence_f1_theo_y"]))
        put(f"lt.{t}.p", dec(x["mcnemar_5class_int8_vs_fp32"]["p_value"], 2))
        put(f"lt.{t}.n", nn(x["n"]))
    ic = ict["ALL"]
    put("tb.int8.n", nn(ic["n"]))
    put("tb.int8.macro", sc(ic["int8"]["macro_f1"]))
    put("tb.fp32.macro", sc(ic["fp32"]["macro_f1"]))
    put("tb.int8.bcy", sc(ic["int8"]["evidence_f1_theo_y"]))
    put("tb.fp32.bcy", sc(ic["fp32"]["evidence_f1_theo_y"]))
    put("tb.int8.khop", pc(ic["agreement"]["label_exact_rate"]))
    put("tb.int8.mcnemar", dec(ic["mcnemar_5class_int8_vs_fp32"]["p_value"], 2))

    # Máy B: `python -m src.v2.bench.do_thiet_bi cpu` (CLI safevichi, batch 1, chạy pin, năng lượng đo từ pin)
    cb = load("results/v2/do_thiet_bi/cpu_laptop.json")
    put("tb.b.cpu", tex_escape(cb["moi_truong"]["cpu"]))
    put("tb.b.ort", cb["moi_truong"]["onnxruntime"])
    put("tb.b.n", nn(cb["cau_hinh"][0]["n"]))
    for r in cb["cau_hinh"]:
        k = f"tb.b.{r['luong']}"
        put(f"{k}.p50", dec(r["do_tre_ms"]["p50"], 1))
        put(f"{k}.p95", dec(r["do_tre_ms"]["p95"], 1))
        put(f"{k}.ram", nn(r["ram_MiB"]["dinh"]))
        put(f"{k}.nap", dec(r["nap_mo_hinh_ms"] / 1000.0, 1))
        put(f"{k}.tl", dec(r["thong_luong_doan_moi_giay"], 1))
    nl = cb["nang_luong_4_luong"]
    put("tb.b.pnghi", dec(nl["cong_suat_nghi_W"], 1))
    put("tb.b.ptai", dec(nl["cong_suat_tai_W"], 1))
    put("tb.b.mj", nn(nl["mJ_moi_doan_tang_them"]))
    put("tb.b.mjtong", nn(nl["mJ_moi_doan_toan_may"]))
    put("tb.b.wh", dec(nl["Wh_moi_1000_doan_tang_them"], 2))
    put("tb.b.giay", nn(nl["dang_tai"]["giay"]))
    put("tb.b.khop.torch", pc(cb["so_khop_pytorch_fp32"]["ti_le_nhan_trung"]))
    put("tb.b.khop.int8", pc(cb["so_khop_onnx_int8_python"]["ti_le_nhan_trung"]))
    put("tb.b.macro", sc(cb["diem"]["ALL"]["macro_f1"]))
    put("tb.b.bcy", sc(cb["diem"]["ALL"]["evidence_f1_theo_y"]))

    # Trình duyệt: `python -m src.v2.bench.do_thiet_bi trinh_duyet` (onnxruntime-web WASM, cùng máy B)
    wb = load("results/v2/do_thiet_bi/trinh_duyet.json")
    put("tb.web.trinhduyet", "Edge " + tex_escape(wb["moi_truong"]["trinh_duyet"].split("/")[-1]))
    put("tb.web.ortweb", wb["moi_truong"]["onnxruntime_web"])
    put("tb.web.backend", tex_escape(wb["nap"]["backend"]))
    put("tb.web.n", nn(wb["n"]))
    put("tb.web.p50", nn(wb["do_tre_khu_hoi_ms"]["p50"]))
    put("tb.web.p95", nn(wb["do_tre_khu_hoi_ms"]["p95"]))
    put("tb.web.tl", dec(wb["thong_luong_doan_moi_giay"], 1))
    put("tb.web.nap", dec(wb["nap"]["wall_load_ms"] / 1000.0, 1))
    put("tb.web.taive", nn(wb["dung_luong_tai_ve_MiB"]))
    put("tb.web.ram.trong", nn(wb["ram_MiB"]["trinh_duyet_trang_trong"]))
    put("tb.web.ram.dinh", nn(wb["ram_MiB"]["dinh_khi_chay"]))
    put("tb.web.ram.nap", nn(wb["ram_MiB"]["sau_khi_nap_mo_hinh"]))
    put("tb.web.ram.tang", nn(wb["ram_MiB"]["tang_them_so_voi_trang_trong"]))
    nw = wb["nang_luong"]
    put("tb.web.pnghi", dec(nw["cong_suat_nghi_W"], 1))
    put("tb.web.ptai", dec(nw["cong_suat_tai_W"], 1))
    put("tb.web.mj", nn(nw["mJ_moi_doan_tang_them"]))
    put("tb.web.mjtong", nn(nw["mJ_moi_doan_toan_may"]))
    put("tb.web.wh", dec(nw["Wh_moi_1000_doan_tang_them"], 2))
    put("tb.web.khop.torch", pc(wb["so_khop_pytorch_fp32"]["ti_le_nhan_trung"]))
    put("tb.web.khop.cli", pc(wb["so_khop_cli_cpu_batch1"]["ti_le_nhan_trung"]))
    put("tb.web.khop.cli.bc", pc(wb["so_khop_cli_cpu_batch1"]["ti_le_bang_chung_trung"]))
    put("tb.web.macro", sc(wb["diem"]["ALL"]["macro_f1"]))
    put("tb.web.bcy", sc(wb["diem"]["ALL"]["evidence_f1_theo_y"]))


def main() -> None:
    S = Store()
    put = S.put

    # ---------- So sánh 3 hệ trên test (C0/C1/C2) ----------
    m3 = load("results/v2/so_sanh_3_he/metrics.json")["ket_qua"]
    for tap in ("C0", "C1", "C2"):
        for vung in ("tat_ca", "bac", "trung", "nam"):
            pre = f"{tap}" if vung == "tat_ca" else f"{tap}.{vung}"
            for he, d in m3[tap][vung].items():
                k = f"{pre}.{he}"
                put(f"{k}.n", nn(d["n"]))
                for f in ("binary_f1", "precision", "recall"):
                    put(f"{k}.{ {'binary_f1': 'bin', 'precision': 'p', 'recall': 'r'}[f] }", sc(d[f]))
                put(f"{k}.fpr", pc(d["fpr_an_toan"]))
                if he == "v1":
                    for lbl, v in d["theo_nhan"].items():
                        put(f"{k}.{MA[lbl]}.r", pc(v["recall"]))
                    continue
                put(f"{k}.macro", sc(d["macro_f1"]))
                put(f"{k}.bcp", sc(d["bc_p_theo_y"]))
                put(f"{k}.bcr", sc(d["bc_r_theo_y"]))
                put(f"{k}.bcf", sc(d["bc_f1_theo_y"]))
                put(f"{k}.bctu", sc(d["bc_f1_muc_tu"]))
                put(f"{k}.nbc", nn(d["n_bc"]))
                for lbl, v in d["theo_nhan"].items():
                    for f in ("precision", "recall", "f1"):
                        put(f"{k}.{MA[lbl]}.{f[0] if f != 'f1' else 'f'}", sc(v[f]))
                    put(f"{k}.{MA[lbl]}.n", nn(v["support"]))
                if vung == "tat_ca":
                    ci = d["ci95"]
                    put(f"{k}.macro.lo", sc(ci["macro_f1"]["low"]))
                    put(f"{k}.macro.hi", sc(ci["macro_f1"]["high"]))
                    put(f"{k}.bin.lo", sc(ci["binary_f1"]["low"]))
                    put(f"{k}.bin.hi", sc(ci["binary_f1"]["high"]))
                    labels = d["ma_tran_nham"]["labels"]
                    for i, a in enumerate(labels):
                        for j, b in enumerate(labels):
                            put(f"{k}.cm.{MA[a]}.{MA[b]}", str(d["ma_tran_nham"]["matrix"][i][j]))
                    for lbl, v in d["bc_theo_nhan"].items():
                        put(f"{k}.{MA[lbl]}.bcf", sc(v["bc_f1_theo_y"]))
        # chênh lệch vùng (cao nhất - thấp nhất)
        for he in ("llm", "v2"):
            for f, ten in (("macro_f1", "macro"), ("binary_f1", "bin")):
                vals = [m3[tap][v][he][f] for v in ("bac", "trung", "nam")]
                put(f"{tap}.{he}.gap.{ten}", sc(max(vals) - min(vals)))
            fprs = [m3[tap][v][he]["fpr_an_toan"] for v in ("bac", "trung", "nam")]
            put(f"{tap}.{he}.gap.fpr", pc(max(fprs) - min(fprs)))
        vals = [m3[tap][v]["v1"]["binary_f1"] for v in ("bac", "trung", "nam")]
        put(f"{tap}.v1.gap.bin", sc(max(vals) - min(vals)))

    # số câu nguy hại bị đoán an toàn (C0, v2) và số câu an toàn bị báo nguy hại
    cm = m3["C0"]["tat_ca"]["v2"]["ma_tran_nham"]["matrix"]
    put("C0.v2.fp.tong", str(sum(cm[0][1:])))
    put("C0.v2.fn.tong", str(sum(r[0] for r in cm[1:])))
    put("C0.at.n", str(sum(cm[0])))

    # lát cắt
    lc = m3["lat_cat"]
    for nhom, ten in (("kieu_bien_dang", "kieu"), ("nhom_bien_dang", "nhom"), ("phan_test", "phan"),
                      ("so_luot", "luot"), ("do_lo", "dolo"), ("kenh", "kenh"), ("nguoi_viet", "nv")):
        for tap, gia_tri in lc[nhom].items():
            for gt, d in gia_tri.items():
                gtk = gt.replace("_", "")
                k = f"{ten}.{tap}.{gtk}"
                put(f"{k}.n", nn(d["v2"]["n"]))
                put(f"{k}.nat", nn(d["n_an_toan"]))
                put(f"{k}.v1.bin", sc(d["v1"]["binary_f1"]))
                put(f"{k}.v1.fpr", pc(d["v1"]["fpr_an_toan"]) if d["n_an_toan"] else "--")
                for he in ("llm", "v2"):
                    put(f"{k}.{he}.macro", sc(d[he]["macro_f1"]))
                    put(f"{k}.{he}.bin", sc(d[he]["binary_f1"]))
                    put(f"{k}.{he}.fpr", pc(d[he]["fpr_an_toan"]) if d["n_an_toan"] else "--")
                    put(f"{k}.{he}.bcf", sc(d[he]["bc_f1_theo_y"]))

    # ---------- Bằng chứng: các độ đo phụ (tính lại bằng bộ chấm v2) ----------
    v2_all = read_jsonl(V2_PRED)
    SOURCES.append(V2_PRED)
    for tap, (gold_path, llm_path, _v1) in SETS.items():
        gold = read_jsonl(Path(gold_path))
        for he, rows in (("v2", v2_all), ("llm", llm_rows(Path(llm_path)))):
            ev = bundle(gold, align_predictions(gold, rows))["evidence"]
            put(f"{tap}.{he}.toan", pc(ev["safe_false_highlight_rate"]))
            put(f"{tap}.{he}.rong", pc(ev["harmful_empty_evidence_rate"]))
            put(f"{tap}.{he}.khongnguyenvan", pc(ev["non_verbatim_phrase_rate"]))
            put(f"{tap}.{he}.socum", nn(ev["total_pred_phrases"]))

    # ---------- Ví dụ ----------
    gold_c0 = {r["id"]: r for r in read_jsonl(Path("data/v2/test_day_du.jsonl"))}
    pred_c0 = {r["id"]: r for r in v2_all if r["bien_dang"] == "C0"}
    llm_c0 = {r["id"]: r for r in llm_rows(Path(SETS["C0"][1]))}
    for ten, i in VI_DU.items():
        g, p = gold_c0[i], pred_c0[i]
        dong = [tex_escape(x) for x in g["text"].split("\n")]
        put(f"vd.{ten}.text", r"\newline ".join(dong) if len(dong) > 1 else dong[0])
        put(f"vd.{ten}.dong", r"\\ ".join(rf"\textit{{{d}}}" for d in dong))
        put(f"vd.{ten}.gold", rf"\nhan{{{g['nhan']}}}")
        put(f"vd.{ten}.pred", rf"\nhan{{{p['nhan']}}}")
        put(f"vd.{ten}.llm", rf"\nhan{{{llm_c0[i]['nhan']}}}")
        put(f"vd.{ten}.tincay", pc(p["tin_cay"]))
        put(f"vd.{ten}.kichban", tex_escape(g["kich_ban"]))
        put(f"vd.{ten}.vung", {"bac": "Bắc", "trung": "Trung", "nam": "Nam"}[g["giong_vung"]])
        put(f"vd.{ten}.bc", ", ".join(rf"``{tex_escape(b)}''" for b in p["bang_chung"]) or "(rỗng)")
        put(f"vd.{ten}.bcgold", ", ".join(rf"``{tex_escape(b)}''" for b in g["bang_chung"]) or "(rỗng)")
        put(f"vd.{ten}.json", tex_escape(json.dumps({"nhan": p["nhan"], "bang_chung": p["bang_chung"]},
                                                    ensure_ascii=False)))

    # ---------- Dữ liệu ----------
    mg = load("data/v2/interim/merged_stats.json")
    put("dl.goc.tong", nn(mg["truoc_khi_gop_text"]["tong"]))
    for nguon, v in mg["truoc_khi_gop_text"]["theo_nguon"].items():
        put(f"dl.goc.{nguon}", nn(v))
    put("dl.khacnhau", nn(mg["tong"]))
    cg = load("data/v2/interim/chon_gan_lai_stats.json")
    put("dl.khotest", nn(cg["kho_test"]["so_cau"]))
    put("dl.khotrain", nn(cg["kho_train"]["so_cau"]))
    put("dl.ganlai", nn(cg["can_gan_lai"]["so_cau"]))
    ly = cg["can_gan_lai"]["theo_ly_do"]
    put("dl.ganlai.nguyhai", nn(sum(ly["nhan_goc_nguy_hai"].values())))
    put("dl.ganlai.blacklist", nn(ly["clean_co_tu_blacklist"]["CLEAN"]))
    put("dl.ganlai.tukhoa", nn(ly["ung_vien_tu_khoa"]["CLEAN"]))
    put("dl.ganlai.ngaunhien", nn(ly["clean_ngau_nhien"]["CLEAN"]))
    tg = load("data/v2/interim/that_gan_lai_stats.json")
    put("dl.that.giu", nn(tg["giu"]))
    put("dl.that.bo", nn(tg["theo_ly_do"]["lech_moi_nguoi_gan"]))
    put("dl.that.khopgoc", nn(tg["theo_ly_do"]["khop_nhan_goc"]))
    for k, v in tg["chuyen_nhan_bi_bo"].items():
        put(f"dl.that.bo.{k.replace('->', '.').replace('_', '').lower()}", nn(v))
    sl = load("data/v2/interim/sinh_llm_loc_stats.json")
    put("dl.sinh.tong", nn(sl["tong"]))
    put("dl.sinh.giu", nn(sl["giu"]))
    put("dl.sinh.loai", nn(sl["loai"]))
    put("dl.sinh.tilegiu", pc(sl["giu"] / sl["tong"]))
    for lbl, v in sl["theo_nhan"].items():
        put(f"dl.sinh.{MA[lbl.split('(')[0]]}.tilegiu", pc(v["ti_le_giu"]))
    put("dl.sinh.loai.raxp", nn(sl["ly_do_loai"]["kiem_lai_ra_XUC_PHAM"]))
    put("dl.sinh.qrba.tilegiu", pc(sl["theo_kich_ban"]["QR3"]["ti_le_giu"]))
    ss = load("data/v2/interim/sinh_llm_stats.json")
    put("dl.sinh.dongtb", dec(ss["so_dong_trung_binh"], 1))

    tr = load("data/v2/train_stats.json")
    put("tr.tong", nn(tr["tong"]))
    tong_that = tong_sinh = 0
    for lbl in NHAN:
        that = tr["theo_nhan_va_loai"].get(f"{lbl}|that", 0)
        sinh = tr["theo_nhan_va_loai"].get(f"{lbl}|sinh", 0)
        tong_that += that
        tong_sinh += sinh
        put(f"tr.{MA[lbl]}.that", nn(that))
        put(f"tr.{MA[lbl]}.sinh", nn(sinh))
        put(f"tr.{MA[lbl]}", nn(tr["theo_nhan"][lbl]))
    put("tr.that", nn(tong_that))
    put("tr.sinh", nn(tong_sinh))
    put("tr.motdong", nn(tr["theo_so_dong"]["1"]))

    te = load("data/v2/test_stats.json")
    put("te.tong", nn(te["so_cau"]))
    for lbl in NHAN:
        put(f"te.{MA[lbl]}", nn(te["theo_nhan"][lbl]))
    for vung in ("bac", "trung", "nam"):
        put(f"te.{vung}", nn(sum(v for k, v in te["theo_giong_vung_va_nhan"].items() if k.startswith(vung + "/"))))
    put("te.chuathay", nn(sum(v for k, v in te["theo_phan_va_nhan"].items() if k.startswith("chua_thay/"))))
    put("te.trongpp", nn(sum(v for k, v in te["theo_phan_va_nhan"].items() if k.startswith("trong_phan_phoi/"))))
    put("te.chuathay.xp", nn(te["theo_phan_va_nhan"]["chua_thay/XUC_PHAM"]))
    put("te.chon.trung", nn(te["theo_nhan_chon_theo"]["trung"]))
    put("te.chon.nguoi", nn(te["theo_nhan_chon_theo"]["nguoi"]))
    put("te.chon.llm", nn(te["theo_nhan_chon_theo"]["llm"]))
    put("te.nguoiviet", str(len(te["theo_nguoi_viet"])))
    for nhom in ("bien_dang_C1", "bien_dang_C2"):
        b = te[nhom]
        tap = nhom[-2:]
        put(f"te.{tap}.n", nn(b["so_ban_ghi"]))
        put(f"te.{tap}.sokieu", str(len(b["theo_kieu"])))
        put(f"te.{tap}.tiletu", pc(b["ti_le_tu_bien_dang"]))
        for kieu, v in b["theo_kieu"].items():
            put(f"te.{tap}.{kieu.replace('_', '')}", nn(v))
    put("te.C1.dathay", nn(te["bien_dang_C1"]["theo_nhom"]["da_thay"]))
    put("te.C1.chuathay", nn(te["bien_dang_C1"]["theo_nhom"]["chua_thay"]))

    rr = load("results/v2/kiem_tra_ro_ri/bao_cao.json")
    put("rr.trunggan", nn(rr["so_cap_trung_gan"]))
    put("rr.nguong", dec(rr["nguong"]["trung_gan"], 1))
    put("rr.kichban", ", ".join(sorted(rr["kich_ban_giu_ngoai_train"])))
    put("rr.trainkichban", nn(rr["so_cau_train_thuoc_kich_ban_giu_ngoai"]))

    dt = load("results/v2/dong_thuan_nhan/metrics.json")
    g = dt["nhom_giu"]["llm_dung_5_nhan"]
    put("dt.giu.dung", nn(g["dung"]))
    put("dt.giu.tong", nn(g["tong"]))
    put("dt.giu.tile", pc(g["ti_le"]))
    put("dt.giu.lo", pc(g["khoang_tin_cay_95"][0]))
    put("dt.giu.hi", pc(g["khoang_tin_cay_95"][1]))
    b = dt["nhom_bo"]["nguoi_dong_y_llm_sach_nguy_hai"]
    put("dt.bo.llm", pc(b["ti_le"]))
    put("dt.mau", nn(dt["so_cau_mau"]))

    # ---------- Chi phí dữ liệu ----------
    cp = load("results/v2/chi_phi_du_lieu/chi_phi.json")
    put("cp.model", tex_escape(cp["gia"]["model"]))
    ten_buoc = {"gan_lai_nhan": "ganlai", "sinh_doan": "sinh", "loc_lai_doan_sinh": "loc"}
    phut = 0.0
    for b in cp["cac_buoc"]:
        k = ten_buoc[b["buoc"]]
        put(f"cp.{k}.goi", nn(b["luot_goi"]))
        put(f"cp.{k}.usd", dec(b["usd"], 2))
        put(f"cp.{k}.phut", dec(b["thoi_gian_phut"], 1))
        phut += b["thoi_gian_phut"]
    put("cp.tong.goi", nn(cp["tong"]["luot_goi"]))
    put("cp.tong.usd", dec(cp["tong"]["usd"], 2))
    put("cp.tong.vnd", nn(cp["tong"]["vnd"]))
    put("cp.tong.phut", dec(phut, 0))
    put("cp.tong.trieutoken", dec((cp["tong"]["token_vao"] + cp["tong"]["token_ra"]) / 1e6, 1))

    # ---------- LLM qua API (chi phí, độ trễ) ----------
    tong_usd = 0.0
    for tap, f in (("C0", "test_metrics.json"), ("C1", "test_perturb_C1_metrics.json"),
                   ("C2", "test_perturb_C2_metrics.json")):
        d = load(f"results/v2/llm_api/{f}")
        ms = d["measurement_summary"]
        put(f"api.{tap}.p50", dec(ms["api_latency_ms"]["p50"] / 1000, 2))
        put(f"api.{tap}.p95", dec(ms["api_latency_ms"]["p95"] / 1000, 2))
        put(f"api.{tap}.usd1000", dec(ms["cost_usd_per_1000_samples"], 3))
        tong_usd += ms["total_cost_usd"]
        if tap == "C0":
            put("api.songsong", str(ms["concurrency"]))
            put("api.provider", tex_escape(d["config"]["provider_order"][0]))
            put("api.reasoning", tex_escape(d["config"]["reasoning"]))
    put("api.tong.usd", dec(tong_usd, 2))

    # ---------- Mô hình v2: cấu hình, hiệu chuẩn, chi phí ----------
    base = "results/v2/official_eval/dualorc_mmbert_base"
    bm = load(f"{base}/benchmark.json")
    put("v2.p50", dec(bm["model_and_collate_p50_ms"], 1))
    put("v2.p95", dec(bm["model_and_collate_p95_ms"], 1))
    put("v2.trongso", nn(bm["model_weights_cuda_MiB"]))
    put("v2.dinh", nn(bm["session_peak_cuda_MiB"]))
    cal = load(f"{base}/calibration.json")
    put("v2.T", dec(cal["temperature"], 2))
    put("v2.nguongbc", dec(cal["marginal_threshold"], 2))
    put("v2.nguongmottu", dec(cal["single_word_threshold"], 2))
    cfg = load(f"{base}/config.json")
    put("v2.maxtok", str(cfg["max_tokens"]))
    put("v2.epoch", str(cfg["epochs"]))
    put("v2.accent", pc(cfg["accent_aug"]))
    put("v2.surface", pc(cfg["surface_aug"]))
    put("v2.fgm", dec(cfg["fgm_eps"], 1))

    # ---------- v2 trên thiết bị đích (ONNX INT8): laptop CPU, trình duyệt, năng lượng ----------
    so_thiet_bi(put, base)
    so_chuan_hoa_test(put, base)
    env = load(f"{base}/environment.json")
    put("v2.gpu", tex_escape(env["device"]))
    put("v2.torch", tex_escape(env["torch"]))
    put("v2.transformers", tex_escape(env["transformers"]))
    put("v2.datarev", cfg["dataset"][:7])

    ds = load("results/v2/dev_selection/dev_summary.json")
    for k, v in ds["split_sizes"].items():
        put(f"dev.chia.{k}", nn(v))
    ten_mo_hinh = {"dualorc_mmbert_base": "mmbert", "dualorc_bamibert": "bamibert",
                   "dualorc_visobert": "visobert", "guardmerge_v6": "guard"}
    for m, k in ten_mo_hinh.items():
        d = ds["models"][m]
        put(f"dev.{k}.macro", sc(d["dev"]["macro_f1"]))
        put(f"dev.{k}.bin", sc(d["dev"]["binary_f1"]))
        put(f"dev.{k}.fpr", pc(d["dev"]["fpr_an_toan"]))
        put(f"dev.{k}.bcf", sc(d["dev"]["evidence"]["f1_theo_y"]))
        put(f"dev.{k}.toan", pc(d["dev"]["evidence"]["safe_mark_rate"]))
        put(f"dev.{k}.gap", sc(d["dev"]["region_gap_max_min"]))
        put(f"dev.{k}.tieng", sc(d["dev"]["slices"]["tieng_long"]["macro_present"]))
        put(f"dev.{k}.tieng.fpr", pc(d["dev"]["slices"]["tieng_long"]["fpr_an_toan"]))
        put(f"dev.{k}.trongso", nn(d["benchmark_gpu"]["model_weights_cuda_MiB"]))
        put(f"dev.{k}.p50", dec(d["benchmark_gpu"]["p50_ms"], 1))
        put(f"dev.{k}.trainphut", dec(d["total_train_seconds"] / 60, 0))
        put(f"dev.{k}.trainvram", nn(d["train_peak_cuda_MiB"]))
    put("dev.tieng.n", nn(ds["models"]["dualorc_mmbert_base"]["dev"]["slices"]["tieng_long"]["n"]))
    mc = ds["mcnemar_dev"]
    put("dev.mc.guard", dec(mc["dualorc_mmbert_base_vs_guardmerge_v6"]["exact_p"], 3))
    put("dev.mc.visobert", dec(mc["dualorc_mmbert_base_vs_dualorc_visobert"]["exact_p"], 2))
    put("dev.mc.bamibert", dec(mc["dualorc_mmbert_base_vs_dualorc_bamibert"]["exact_p"], 2))

    ctx = load(f"{base}/dev_context_checks.json")
    put("ctx.n", nn(ctx["multi_n"]))
    put("ctx.full", sc(ctx["full_context_multi_macro_f1"]))
    put("ctx.last", sc(ctx["last_turn_multi_macro_f1"]))
    put("ctx.gain", dl(ctx["full_context_multi_macro_f1"] - ctx["last_turn_multi_macro_f1"]))
    put("ctx.lenonly", sc(ctx["length_only_macro_f1"]))

    raw = load(f"{base}/dev_stress_tests.json")
    norm = load(f"{base}/dev_norm_stress_tests.json")
    for suite, d in norm.items():
        k = suite.replace("_", "")
        put(f"cn.{k}.n", nn(d["n"]))
        put(f"cn.{k}.raw", sc(d["perturbed_raw_f1"]))
        put(f"cn.{k}.norm", sc(d["perturbed_norm_f1"]))
        put(f"cn.{k}.gain", dl(d["gain_vs_raw_f1"]))
        if suite in raw:
            put(f"cn.{k}.goc", sc(raw[suite]["original_f1"]))
            put(f"cn.{k}.tut", dl(raw[suite]["delta_f1"]))

    # ---------- v1 ----------
    v1m = load("v1/results/step3/test/matrix.json")["results"]["b3_t5ft_norm"]
    for tap, k in (("C0_clean", "C0"), ("C1_perturbed", "C1"), ("C2_blindspot", "C2")):
        put(f"v1goc.{k}", sc(v1m[tap]["hate_f1"]))
    put("v1goc.n", nn(v1m["C0_clean"]["n"]))
    mf = load("demo/model-manifest.json")
    put("v1.onnx", nn(mf["totalBytes"] / 2**20))

    # ---------- Biểu đồ: Binary-F1 của 3 hệ ----------
    for he in ("v1", "llm", "v2"):
        pts = " ".join(f"({t},{m3[t]['tat_ca'][he]['binary_f1'] * 100:.1f})" for t in ("C0", "C1", "C2"))
        put(f"plot.bin.{he}", pts)

    ve_hinh(m3, ds, tr)

    # ---------- Ghi file ----------
    lines = [
        "% Tệp sinh tự động bằng `python -m src.v2.eval.xuat_so_lieu_bao_cao`. Không sửa tay.",
        "% Nguồn (sha256 12 ký tự đầu):",
    ]
    for p in sorted(set(SOURCES)):
        lines.append(f"%   {p.as_posix()}  {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}")
    for k, v in S.items.items():
        lines.append(rf"\soset{{{k}}}{{{v}}}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {len(S.items)} số liệu vào {OUT}")


HINH = OUT.parent / "hinh"
TEN_KIEU = {
    "bo_dau": "Bỏ dấu", "teencode": "Teencode", "go_sai": "Gõ sai", "keo_dai": "Kéo dài ký tự",
    "chen_ky_tu": "Chèn ký tự", "so_thay_chu": "Số thay chữ", "dinh_tu": "Dính từ",
    "fullwidth": "Chữ toàn khổ", "zero_width": "Ký tự vô hình", "space_injection": "Chèn dấu cách",
    "exotic_separator": "Dấu phân cách lạ", "cyrillic_homoglyph": "Chữ Cyrillic giống Latin",
    "double_elongation": "Kéo dài kép",
}


def v1d(x: float) -> str:
    """Số hiển thị trong hình: nhân 100, 1 chữ số, dấu phẩy."""
    return f"{x * 100:.1f}".replace(".", ",")


def ghi_hinh(ten: str, noi_dung: str) -> None:
    HINH.mkdir(parents=True, exist_ok=True)
    dau = "% Sinh tự động bằng `python -m src.v2.eval.xuat_so_lieu_bao_cao`. Không sửa tay.\n"
    (HINH / f"{ten}.tex").write_text(dau + noi_dung.strip() + "\n", encoding="utf-8")
    # Khung standalone để xuất từng hình ra PDF riêng
    (HINH / "pdf").mkdir(exist_ok=True)
    (HINH / "pdf" / f"{ten}.tex").write_text(KHUNG_PDF.replace("@TEN@", ten), encoding="utf-8")


def xuat_pdf() -> None:
    """Biên dịch từng hình thành PDF riêng trong hinh/pdf/ (cần pdflatex)."""
    import subprocess
    thu_muc = HINH / "pdf"
    for tex in sorted(thu_muc.glob("*.tex")):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=thu_muc, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(("OK   " if r.returncode == 0 else "LỖI  ") + str(thu_muc / (tex.stem + ".pdf")))
        for duoi in (".aux", ".log"):
            (thu_muc / (tex.stem + duoi)).unlink(missing_ok=True)


KHUNG_PDF = r"""\documentclass[border=6pt]{standalone}
% Sinh tự động. Biên dịch: pdflatex <ten>.tex (trong thư mục hinh/pdf/)
\usepackage[utf8]{inputenc}
\usepackage[T5]{fontenc}
\usepackage[vietnamese]{babel}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{navy}{HTML}{17365D}
\definecolor{mauvmot}{HTML}{257DE0}
\definecolor{maullm}{HTML}{F47230}
\definecolor{mauvhai}{HTML}{00B894}
\newcommand{\nhan}[1]{\texttt{\detokenize{#1}}}
\setlength{\textwidth}{16.3cm}
\begin{document}
\setlength{\linewidth}{\textwidth}
\input{../@TEN@.tex}
\end{document}
"""


def ve_hinh(m3: dict, ds: dict, tr: dict) -> None:
    # 0. Binary-F1 của 3 hệ trên C0/C1/C2
    def cot_bin(he):
        return " ".join(f"({t},{m3[t]['tat_ca'][he]['binary_f1'] * 100:.1f})" for t in ("C0", "C1", "C2"))
    ghi_hinh("binary_ba_he", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ybar=3pt, bar width=18pt, width=\linewidth, height=6cm, ymin=0, ymax=100,
  symbolic x coords={{C0,C1,C2}}, xtick=data, enlarge x limits=0.25, ylabel={{Binary-F1}},
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={{draw=black!12}},
  tick label style={{font=\small}}, label style={{font=\small}},
  nodes near coords={{\pgfmathprintnumber[fixed,fixed zerofill,precision=1,use comma]{{\pgfplotspointmeta}}}},
  every node near coord/.append style={{font=\scriptsize, color=black!75}},
  legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=3, draw=none, font=\small,
    /tikz/every even column/.append style={{column sep=0.4cm}}}}]
\addplot[fill=mauvmot, draw=white, area legend] coordinates {{{cot_bin('v1')}}};
\addlegendentry{{v1}}
\addplot[fill=maullm, draw=white, area legend] coordinates {{{cot_bin('llm')}}};
\addlegendentry{{gpt-oss-120b (API)}}
\addplot[fill=mauvhai, draw=white, area legend] coordinates {{{cot_bin('v2')}}};
\addlegendentry{{v2}}
\end{{axis}}
\end{{tikzpicture}}""")

    # 1. Phân bố train: câu thật / đoạn sinh theo nhãn (thanh ngang chồng)
    nhan_ve = ("LUA_DAO", "DE_DOA", "QUAY_ROI", "XUC_PHAM", "AN_TOAN")
    that = " ".join(f"({tr['theo_nhan_va_loai'].get(l + '|that', 0)},{MA[l]})" for l in nhan_ve)
    sinh = " ".join(f"({tr['theo_nhan_va_loai'].get(l + '|sinh', 0)},{MA[l]})" for l in nhan_ve)
    nhan_y = ",".join(r"\nhan{" + l + "}" for l in nhan_ve)
    ghi_hinh("train_phan_bo", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[xbar stacked, bar width=11pt, width=0.9\linewidth, height=5.2cm,
  symbolic y coords={{{",".join(MA[l] for l in nhan_ve)}}}, ytick=data, yticklabels={{{nhan_y}}},
  xmin=0, xlabel={{Số mẫu train}}, axis x line*=bottom, axis y line*=left, xmajorgrids,
  grid style={{draw=black!12}}, tick label style={{font=\small}}, label style={{font=\small}},
  scaled x ticks=false, xticklabel style={{/pgf/number format/.cd, 1000 sep={{.}}}},
  legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=2, draw=none, font=\small,
    /tikz/every even column/.append style={{column sep=0.4cm}}}}]
\addplot[fill=navy, draw=white, area legend] coordinates {{{that}}};
\addlegendentry{{Câu thật (gán lại)}}
\addplot[fill=maullm, draw=white, area legend] coordinates {{{sinh}}};
\addlegendentry{{Đoạn LLM sinh}}
\end{{axis}}
\end{{tikzpicture}}""")

    # 2. Ma trận nhầm lẫn v2 C0, tô theo tỉ lệ trong hàng
    d = m3["C0"]["tat_ca"]["v2"]["ma_tran_nham"]
    labels, mat = d["labels"], d["matrix"]
    o = 1.45
    dong = []
    for i, a in enumerate(labels):
        tong = sum(mat[i])
        for j, _b in enumerate(labels):
            v = mat[i][j]
            pct = round(100 * v / tong) if tong else 0
            mau = "white" if pct >= 50 else "black"
            dong.append(rf"\fill[mauvmot!{max(pct, 0)}] ({j * o},{-i * o}) rectangle ++({o},{-o});"
                        rf"\node[font=\small\bfseries,text={mau}] at ({j * o + o / 2},{-i * o - o / 2}) {{{v}}};")
        dong.append(rf"\node[anchor=east,font=\footnotesize] at (-0.1,{-i * o - o / 2}) {{\nhan{{{a}}}}};")
    for j, b in enumerate(labels):
        dong.append(rf"\node[anchor=west,rotate=40,font=\footnotesize] at ({j * o + o / 2 - 0.2},0.15) {{\nhan{{{b}}}}};")
    n = len(labels)
    dong.append(rf"\draw[black!25] (0,0) grid[step={o}] ({n * o},{-n * o});")
    dong.append(rf"\node[font=\small,rotate=90] at (-3.0,{-n * o / 2}) {{Nhãn đúng}};")
    dong.append(rf"\node[font=\small] at ({n * o / 2},2.0) {{Nhãn v2 đoán}};")
    ghi_hinh("ma_tran_nham", "\\begin{tikzpicture}\n" + "\n".join(dong) + "\n\\end{tikzpicture}")

    # 3. F1 bằng chứng qua C0 -> C1 -> C2
    def duong(he):
        return " ".join(f"({t},{m3[t]['tat_ca'][he]['bc_f1_theo_y'] * 100:.1f})" for t in ("C0", "C1", "C2"))
    nhan_cuoi = {he: v1d(m3["C2"]["tat_ca"][he]["bc_f1_theo_y"]) for he in ("llm", "v2")}
    nhan_dau = {he: v1d(m3["C0"]["tat_ca"][he]["bc_f1_theo_y"]) for he in ("llm", "v2")}
    ghi_hinh("bang_chung_bien_dang", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=0.75\linewidth, height=5.8cm, ymin=0, ymax=100,
  symbolic x coords={{C0,C1,C2}}, xtick=data, enlarge x limits=0.15, ylabel={{F1 bằng chứng (theo ý)}},
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={{draw=black!12}},
  tick label style={{font=\small}}, label style={{font=\small}}, clip=false,
  xticklabels={{C0 (gốc),C1 (biến dạng thường),C2 (điểm mù)}}]
\addplot[mauvhai, line width=2pt, mark=*, mark size=3pt] coordinates {{{duong('v2')}}};
\addplot[maullm, line width=2pt, mark=*, mark size=3pt] coordinates {{{duong('llm')}}};
\node[anchor=west,font=\small] at (axis cs:C2,{m3['C2']['tat_ca']['v2']['bc_f1_theo_y'] * 100:.1f}) {{\ v2: {nhan_cuoi['v2']}}};
\node[anchor=west,font=\small] at (axis cs:C2,{m3['C2']['tat_ca']['llm']['bc_f1_theo_y'] * 100:.1f}) {{\ gpt-oss-120b: {nhan_cuoi['llm']}}};
\node[anchor=south,font=\small] at (axis cs:C0,{m3['C0']['tat_ca']['v2']['bc_f1_theo_y'] * 100:.1f}) {{{nhan_dau['v2']}}};
\node[anchor=north,font=\small] at (axis cs:C0,{m3['C0']['tat_ca']['llm']['bc_f1_theo_y'] * 100:.1f}) {{{nhan_dau['llm']}}};
\end{{axis}}
\end{{tikzpicture}}""")

    # 4. Macro-F1 theo 13 kiểu biến dạng: mỗi hàng nối điểm LLM và v2
    lc = m3["lat_cat"]["kieu_bien_dang"]
    xmin, xmax, rong = 40.0, 95.0, 9.0
    def X(v):
        return (v * 100 - xmin) / (xmax - xmin) * rong
    dong, y = [], 0.0
    for tap, ten_nhom in (("C1", "C1: hỏng chính con chữ"), ("C2", "C2: đổi lớp ký tự (chưa thấy khi train)")):
        dong.append(rf"\node[anchor=west,font=\small\bfseries] at (-5.2,{y:.2f}) {{{ten_nhom}}};")
        y -= 0.55
        for kieu in sorted(lc[tap], key=lambda k: -lc[tap][k]["v2"]["macro_f1"]):
            a, b = lc[tap][kieu]["llm"]["macro_f1"], lc[tap][kieu]["v2"]["macro_f1"]
            dong.append(rf"\node[anchor=east,font=\footnotesize] at (-0.15,{y:.2f}) {{{TEN_KIEU[kieu]}}};")
            dong.append(rf"\draw[black!30, line width=1.2pt] ({X(a):.2f},{y:.2f}) -- ({X(b):.2f},{y:.2f});")
            dong.append(rf"\fill[maullm] ({X(a):.2f},{y:.2f}) circle (3.2pt);")
            dong.append(rf"\fill[mauvhai] ({X(b):.2f},{y:.2f}) circle (3.2pt);")
            dong.append(rf"\node[anchor=west,font=\scriptsize] at ({X(b) + 0.12:.2f},{y:.2f}) {{{v1d(b)}}};")
            dong.append(rf"\node[anchor=east,font=\scriptsize,text=black!70] at ({X(a) - 0.12:.2f},{y:.2f}) {{{v1d(a)}}};")
            y -= 0.5
        y -= 0.2
    for t in range(int(xmin), int(xmax) + 1, 10):
        dong.append(rf"\draw[black!12] ({X(t / 100):.2f},0.3) -- ({X(t / 100):.2f},{y + 0.2:.2f});")
        dong.append(rf"\node[font=\scriptsize,anchor=north] at ({X(t / 100):.2f},{y + 0.2:.2f}) {{{t}}};")
    dong.append(rf"\node[font=\small,anchor=north] at ({rong / 2:.2f},{y - 0.25:.2f}) {{Macro-F1 5 nhãn}};")
    dong.append(rf"\fill[mauvhai] (1.0,0.75) circle (3.2pt); \node[anchor=west,font=\small] at (1.15,0.75) {{v2}};"
                rf"\fill[maullm] (2.6,0.75) circle (3.2pt); \node[anchor=west,font=\small] at (2.75,0.75) {{gpt-oss-120b (API)}};")
    # các đường lưới vẽ trước để nằm dưới chấm
    luoi = [l for l in dong if "black!12" in l or "anchor=north] at" in l]
    con_lai = [l for l in dong if l not in luoi]
    ghi_hinh("kieu_bien_dang", "\\begin{tikzpicture}\n" + "\n".join(luoi + con_lai) + "\n\\end{tikzpicture}")

    # 5. Kịch bản trong phân phối / chưa thấy
    pt = m3["lat_cat"]["phan_test"]
    def cot(he, phan):
        return " ".join(f"({t},{pt[t][phan][he]['macro_f1'] * 100:.1f})" for t in ("C0", "C1", "C2"))
    ghi_hinh("kich_ban_chua_thay", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[ybar=2pt, bar width=13pt, width=\linewidth, height=6cm, ymin=0, ymax=100,
  symbolic x coords={{C0,C1,C2}}, xtick=data, enlarge x limits=0.22, ylabel={{Macro-F1 5 nhãn}},
  axis x line*=bottom, axis y line*=left, ymajorgrids, grid style={{draw=black!12}},
  tick label style={{font=\small}}, label style={{font=\small}},
  nodes near coords={{\pgfmathprintnumber[fixed,fixed zerofill,precision=1,use comma]{{\pgfplotspointmeta}}}},
  every node near coord/.append style={{font=\tiny, color=black!75}}, legend cell align=left,
  legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=2, draw=none, font=\footnotesize,
    /tikz/every even column/.append style={{column sep=0.5cm}}}}]
\addplot[fill=mauvhai, draw=white, area legend] coordinates {{{cot('v2', 'trong_phan_phoi')}}};
\addlegendentry{{v2, kịch bản đã có trong train}}
\addplot[fill=mauvhai!35, draw=mauvhai, area legend] coordinates {{{cot('v2', 'chua_thay')}}};
\addlegendentry{{v2, kịch bản chưa thấy}}
\addplot[fill=maullm, draw=white, area legend] coordinates {{{cot('llm', 'trong_phan_phoi')}}};
\addlegendentry{{gpt-oss-120b, kịch bản đã có trong train v2}}
\addplot[fill=maullm!35, draw=maullm, area legend] coordinates {{{cot('llm', 'chua_thay')}}};
\addlegendentry{{gpt-oss-120b, kịch bản chưa thấy}}
\end{{axis}}
\end{{tikzpicture}}""")

    # 6. Bốn ứng viên trên dev: Macro-F1 theo độ trễ GPU
    ten = {"dualorc_mmbert_base": "mmBERT-base (chọn)", "dualorc_bamibert": "BamiBERT",
           "dualorc_visobert": "ViSoBERT", "guardmerge_v6": "GuardMerge 0,6B"}
    vi_tri = {"dualorc_mmbert_base": "west,xshift=4pt", "dualorc_bamibert": "north,yshift=-4pt",
              "dualorc_visobert": "south,yshift=4pt", "guardmerge_v6": "south,yshift=4pt"}
    diem = []
    for m, nm in ten.items():
        dm = ds["models"][m]
        x, yv = dm["benchmark_gpu"]["p50_ms"], dm["dev"]["macro_f1"] * 100
        mau = "mauvhai" if m == "dualorc_mmbert_base" else "navy!70"
        diem.append(rf"\addplot[only marks, mark=*, mark size=4pt, {mau}] coordinates {{({x:.2f},{yv:.2f})}};")
        diem.append(rf"\node[anchor={vi_tri[m]},font=\footnotesize] at (axis cs:{x:.2f},{yv:.2f}) "
                    rf"{{{nm}: {v1d(yv / 100)}; {nn(dm['benchmark_gpu']['model_weights_cuda_MiB'])} MiB}};")
    ghi_hinh("ung_vien_dev", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[width=0.85\linewidth, height=6cm, xmin=0, xmax=100, ymin=89, ymax=93.6,
  xlabel={{Độ trễ p50 mỗi đoạn trên GPU (ms, batch 1)}}, ylabel={{Macro-F1 trên dev}},
  axis x line*=bottom, axis y line*=left, grid=major, grid style={{draw=black!12}},
  tick label style={{font=\small}}, label style={{font=\small}}, clip=false,
  yticklabel style={{/pgf/number format/.cd, use comma, fixed}}]
{chr(10).join(diem)}
\end{{axis}}
\end{{tikzpicture}}""")


if __name__ == "__main__":
    import sys
    main()
    if "--pdf" in sys.argv:
        xuat_pdf()
