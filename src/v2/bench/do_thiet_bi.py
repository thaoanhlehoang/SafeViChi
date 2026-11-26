"""Đo v2 (ONNX INT8) trên môi trường đích: laptop CPU và trình duyệt, kèm năng lượng đo từ pin.

    python -m src.v2.bench.do_thiet_bi cpu           # onnxruntime trên CPU (1, 4, 8 luồng) + năng lượng (4 luồng)
    python -m src.v2.bench.do_thiet_bi trinh_duyet   # onnxruntime-web trong Edge headless + năng lượng

Cả hai lệnh chạy toàn bộ 1.670 đoạn test (334 C0 + 668 C1 + 668 C2), batch 1, tuần tự như khi người dùng nhận tin,
và so khớp nhãn, bằng chứng với dự đoán đã khóa của bản PyTorch FP32 và bản ONNX INT8 (Python).

Năng lượng: Windows không cho đọc bộ đếm RAPL khi không có quyền admin, nên đo công suất toàn máy bằng tốc độ xả pin
(WMI `root/wmi:BatteryStatus.DischargeRate`, mW). Máy phải chạy bằng pin, không cắm sạc. Đo một cửa sổ nghỉ trước và
sau, một cửa sổ đang suy luận (mỗi cửa sổ vài phút vì cảm biến pin chỉ cập nhật khoảng 8–9 giây một lần). Năng lượng
mỗi đoạn = công suất tăng thêm so với lúc nghỉ × thời gian / số đoạn; kèm số theo công suất toàn máy.

Kết quả ghi vào `results/v2/do_thiet_bi/`: `cpu_laptop.json`, `trinh_duyet.json`, dự đoán `*_predictions.jsonl`
(lượt đầu, 4 luồng) và `env.txt`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import psutil

SEED = 42
ONNX_DIR = Path("models/dualorc_mmbert_base/onnx")
GOLD = [Path("data/v2/test_day_du.jsonl"), Path("data/v2/test_perturb_C1.jsonl"), Path("data/v2/test_perturb_C2.jsonl")]
META = Path("data/v2/test_day_du.jsonl")
REF_DIR = Path("results/v2/official_eval/dualorc_mmbert_base/test")
REF_TORCH = REF_DIR / "real_test_all_predictions.jsonl"        # PyTorch FP32, đã khóa
REF_INT8 = REF_DIR / "real_test_int8_all_predictions.jsonl"    # ONNX INT8 chạy bằng Python
OUT_DIR = Path("results/v2/do_thiet_bi")
WEB_DIR = Path("web-v2")


# ---------------------------------------------------------------------------------------------------------------
# Pin
# ---------------------------------------------------------------------------------------------------------------

_PS_PIN = r"""
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
  $b = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | Select-Object -First 1
  [Console]::Out.WriteLine("$($b.DischargeRate) $($b.RemainingCapacity) $($b.PowerOnline)")
  [Console]::Out.Flush()
  Start-Sleep -Milliseconds 1000
}
"""


class PinSampler:
    """Đọc tốc độ xả pin mỗi giây trong một tiến trình PowerShell chạy nền (chạy suốt cả cửa sổ nghỉ và tải)."""

    def __init__(self) -> None:
        self.mau: list[tuple[float, float, float, bool]] = []  # (epoch_s, mW, mWh còn lại, đang cắm sạc)
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_PIN],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
        )
        self._thread = threading.Thread(target=self._doc, daemon=True)
        self._thread.start()

    def _doc(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                self.mau.append((time.time(), float(parts[0]), float(parts[1]), parts[2].lower() == "true"))
            except ValueError:
                continue

    def stop(self) -> None:
        self._proc.kill()

    def kiem_tra_chay_pin(self, cho_s: float = 15.0) -> None:
        t0 = time.time()
        while not self.mau and time.time() - t0 < cho_s:
            time.sleep(0.5)
        if not self.mau:
            raise RuntimeError("Không đọc được WMI BatteryStatus: máy không có pin hoặc không hỗ trợ.")
        if self.mau[-1][3]:
            raise RuntimeError("Máy đang cắm sạc. Rút sạc rồi chạy lại để đo năng lượng bằng pin.")

    def thong_ke(self, t0: float, t1: float) -> dict[str, Any]:
        w = [m for m in self.mau if t0 <= m[0] <= t1]
        if len(w) < 10:
            raise RuntimeError(f"Cửa sổ đo pin quá ngắn ({len(w)} mẫu).")
        if any(m[3] for m in w):
            raise RuntimeError("Máy bị cắm sạc trong lúc đo năng lượng; kết quả không dùng được.")
        rate_w = float(np.mean([m[1] for m in w])) / 1000.0
        cap = [m[2] for m in w]
        # Chỉ lấy dung lượng tại các lần cảm biến cập nhật để tính độ giảm theo thời gian
        doi = [(m[0], m[2]) for i, m in enumerate(w) if i == 0 or m[2] != w[i - 1][2]]
        cap_w = None
        if len(doi) >= 3:
            dt_h = (doi[-1][0] - doi[1][0]) / 3600.0
            if dt_h > 0:
                cap_w = (doi[1][1] - doi[-1][1]) / dt_h / 1000.0
        return {
            "giay": round(t1 - t0, 1),
            "so_mau": len(w),
            "so_lan_cam_bien_cap_nhat": len({m[1] for m in w}),
            "cong_suat_tb_W": round(rate_w, 3),
            "cong_suat_theo_dung_luong_W": None if cap_w is None else round(cap_w, 3),
            "dung_luong_dau_cuoi_mWh": [cap[0], cap[-1]],
        }


def tinh_nang_luong(nghi_truoc: dict, tai: dict, nghi_sau: dict, n_doan: int) -> dict[str, Any]:
    p_nghi = (nghi_truoc["cong_suat_tb_W"] + nghi_sau["cong_suat_tb_W"]) / 2.0
    p_tai = tai["cong_suat_tb_W"]
    t = tai["giay"]
    tang = max(0.0, p_tai - p_nghi)
    return {
        "phuong_phap": "Tốc độ xả pin toàn máy (WMI BatteryStatus.DischargeRate), chạy pin, màn hình bật; "
                       "nghỉ trước/sau và đang tải; năng lượng tăng thêm = (P_tai - P_nghi) × t / số đoạn.",
        "nghi_truoc": nghi_truoc,
        "dang_tai": tai,
        "nghi_sau": nghi_sau,
        "cong_suat_nghi_W": round(p_nghi, 3),
        "cong_suat_tai_W": round(p_tai, 3),
        "cong_suat_tang_them_W": round(tang, 3),
        "so_doan_trong_cua_so_tai": n_doan,
        "mJ_moi_doan_tang_them": round(tang * t / n_doan * 1000.0, 1),
        "mJ_moi_doan_toan_may": round(p_tai * t / n_doan * 1000.0, 1),
        "Wh_moi_1000_doan_tang_them": round(tang * t / n_doan * 1000.0 / 3600.0, 4),
        "Wh_moi_1000_doan_toan_may": round(p_tai * t / n_doan * 1000.0 / 3600.0, 4),
    }


def cho_nghi(pin: PinSampler, giay: float, nhan: str) -> dict[str, Any]:
    print(f"    Cửa sổ nghỉ {nhan}: {giay:.0f} s (đừng dùng máy)...", flush=True)
    t0 = time.time()
    time.sleep(giay)
    return pin.thong_ke(t0, time.time())


# ---------------------------------------------------------------------------------------------------------------
# Dữ liệu, so khớp, chấm điểm
# ---------------------------------------------------------------------------------------------------------------


def doc_test() -> list[dict[str, Any]]:
    from src.v2.eval.evaluate import enrich_metadata, read_jsonl

    records = [row for p in GOLD for row in read_jsonl(p)]
    return enrich_metadata(records, META)


def so_khop(preds: list[dict[str, Any]], ref_path: Path) -> dict[str, Any]:
    from src.v2.eval.evaluate import read_jsonl

    ref = {r["id"]: r for r in read_jsonl(ref_path)}
    n = len(preds)
    nhan = sum(1 for p in preds if p["nhan"] == ref[p["id"]]["nhan"])
    bc = sum(1 for p in preds if list(p["bang_chung"]) == list(ref[p["id"]]["bang_chung"]))
    lech = [{"id": p["id"], "ref": ref[p["id"]]["nhan"], "do": p["nhan"]} for p in preds if p["nhan"] != ref[p["id"]]["nhan"]]
    return {
        "tham_chieu": str(ref_path).replace("\\", "/"),
        "n": n,
        "nhan_trung": nhan,
        "ti_le_nhan_trung": round(nhan / n, 4),
        "bang_chung_trung_hoan_toan": bc,
        "ti_le_bang_chung_trung": round(bc / n, 4),
        "cac_ca_lech_nhan": lech[:50],
    }


def cham_diem(records: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    from src.v2.eval.evaluate import align_predictions
    from src.v2.eval.metrics import evaluate_bundle

    aligned = align_predictions(records, preds)
    out: dict[str, Any] = {}
    for tang in ("ALL", "C0", "C1", "C2"):
        idx = [i for i, r in enumerate(records) if tang == "ALL" or str(r.get("bien_dang", "C0")) == tang]
        if not idx:
            continue
        rep = evaluate_bundle([records[i] for i in idx], [aligned[i] for i in idx], n_bootstrap=0, include_slices=False)
        cls, ev = rep["classification"], rep["evidence"]
        out[tang] = {
            "n": len(idx),
            "macro_f1": cls["macro_f1"],
            "binary_f1": cls["binary_harmful_vs_safe"]["f1"],
            "fpr_an_toan": cls["binary_harmful_vs_safe"]["fpr_an_toan"],
            "evidence_f1_theo_y": ev["f1_theo_y"]["f1"],
        }
    return out


def phan_vi(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
    }


def ghi_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def kich_thuoc_mib(*paths: Path) -> float:
    return round(sum(p.stat().st_size for p in paths) / 2**20, 2)


def moi_truong() -> dict[str, Any]:
    import onnxruntime as ort

    cpu = platform.processor()
    try:
        cpu = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True
        ).strip() or cpu
    except Exception:
        pass
    return {
        "cpu": cpu,
        "nhan_vat_ly": psutil.cpu_count(logical=False),
        "luong_logic": psutil.cpu_count(logical=True),
        "ram_GiB": round(psutil.virtual_memory().total / 2**30, 2),
        "he_dieu_hanh": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "che_do_nguon": _che_do_nguon(),
    }


def _che_do_nguon() -> str:
    pin = psutil.sensors_battery()
    if pin is None:
        return "không có pin"
    return "cắm sạc" if pin.power_plugged else "chạy pin"


# ---------------------------------------------------------------------------------------------------------------
# CPU (onnxruntime Python)
# ---------------------------------------------------------------------------------------------------------------


def cpu_worker(args: argparse.Namespace) -> None:
    """Chạy trong tiến trình riêng để RAM đỉnh của mỗi cấu hình không lẫn nhau. In JSON ở dòng cuối.

    Dùng đúng bộ suy luận của CLI `safevichi` (`SafeViChiPredictor`: onnxruntime + tokenizers, không cần PyTorch).
    """
    from src.v2.predict import SafeViChiPredictor

    proc = psutil.Process(os.getpid())
    records = doc_test()
    rss0 = proc.memory_info().rss / 2**20
    t0 = time.perf_counter()
    model = SafeViChiPredictor(onnx_dir=ONNX_DIR, threads=args.threads)
    nap_ms = (time.perf_counter() - t0) * 1000.0
    rss_nap = proc.memory_info().rss / 2**20

    def mot_doan(r: dict[str, Any]) -> tuple[dict[str, Any], float]:
        a = time.perf_counter()
        out = model.predict(r["text"])
        p = {"id": r["id"], "nhan": out["nhan"], "bang_chung": out["bang_chung"], "probs": out["probs"]}
        return p, (time.perf_counter() - a) * 1000.0

    for r in records[:10]:
        mot_doan(r)
    preds, ms = [], []
    t_start = time.time()
    n_tong, luot = 0, 0
    while True:
        for r in records:
            p, d = mot_doan(r)
            n_tong += 1
            if luot == 0:
                preds.append({"id": p["id"], "nhan": p["nhan"], "bang_chung": p["bang_chung"], "probs": p["probs"]})
                ms.append(d)
        luot += 1
        if time.time() - t_start >= args.min_giay:
            break
    t_end = time.time()
    mi = proc.memory_info()
    out = {
        "luong": args.threads,
        "n": len(preds),
        "nap_mo_hinh_ms": round(nap_ms, 1),
        "do_tre_ms": phan_vi(ms),
        "thong_luong_doan_moi_giay": round(1000.0 / float(np.mean(ms)), 2),
        "ram_MiB": {
            "truoc_khi_nap": round(rss0, 1),
            "sau_khi_nap": round(rss_nap, 1),
            "dinh": round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1),
        },
        "t_start": t_start,
        "t_end": t_end,
        "n_tong": n_tong,
        "luot": luot,
    }
    if args.ghi_du_doan:
        ghi_jsonl(Path(args.ghi_du_doan), preds)
    print(json.dumps(out, ensure_ascii=False))


def chay_cpu_worker(threads: int, min_giay: float, ghi_du_doan: Path | None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "src.v2.bench.do_thiet_bi", "_cpu_worker", "--threads", str(threads),
           "--min-giay", str(min_giay)]
    if ghi_du_doan:
        cmd += ["--ghi-du-doan", str(ghi_du_doan)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TRANSFORMERS_VERBOSITY": "error"}
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", env=env)
    return json.loads([ln for ln in out.splitlines() if ln.strip().startswith("{")][-1])


def do_cpu(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = doc_test()
    pin = PinSampler()
    try:
        pin.kiem_tra_chay_pin()
        env = moi_truong()
        print(f"CPU: {env['cpu']}, {len(records)} đoạn test", flush=True)
        nghi1 = cho_nghi(pin, args.nghi, "trước")
        pred_path = OUT_DIR / "cpu_laptop_predictions.jsonl"
        print(f"    4 luồng: một lượt đủ 1.670 đoạn, lặp tới khi đủ {args.tai:.0f} s để đo năng lượng...", flush=True)
        r4 = chay_cpu_worker(4, args.tai, pred_path)
        tai = pin.thong_ke(r4["t_start"], r4["t_end"])
        nghi2 = cho_nghi(pin, args.nghi, "sau")
        nang_luong = tinh_nang_luong(nghi1, tai, nghi2, r4["n_tong"])
        runs = [r4]
        for th in (8, 1):
            print(f"    {th} luồng: một lượt 1.670 đoạn...", flush=True)
            runs.append(chay_cpu_worker(th, 0, None))
    finally:
        pin.stop()

    from src.v2.eval.evaluate import read_jsonl

    preds = read_jsonl(pred_path)
    ket_qua = {
        "mo_hinh": "safevichi_v2 ONNX INT8",
        "lenh": "python -m src.v2.bench.do_thiet_bi cpu",
        "seed": SEED,
        "moi_truong": env,
        "dung_luong_MiB": {
            "onnx_int8": kich_thuoc_mib(ONNX_DIR / "safevichi_v2_int8.onnx"),
            "tokenizer": kich_thuoc_mib(ONNX_DIR / "tokenizer" / "tokenizer.json"),
        },
        "cau_hinh": [{k: v for k, v in r.items() if k not in ("t_start", "t_end")} for r in runs],
        "nang_luong_4_luong": nang_luong,
        "so_khop_pytorch_fp32": so_khop(preds, REF_TORCH),
        "so_khop_onnx_int8_python": so_khop(preds, REF_INT8),
        "diem": cham_diem(records, preds),
    }
    (OUT_DIR / "cpu_laptop.json").write_text(json.dumps(ket_qua, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ghi_env(env)
    print(json.dumps({k: ket_qua[k] for k in ("cau_hinh", "nang_luong_4_luong")}, ensure_ascii=False, indent=1)[:3000])


# ---------------------------------------------------------------------------------------------------------------
# Trình duyệt (onnxruntime-web trong Edge headless, điều khiển qua DevTools Protocol)
# ---------------------------------------------------------------------------------------------------------------


class CDP:
    def __init__(self, ws_url: str) -> None:
        import websocket

        self.ws = websocket.create_connection(ws_url, timeout=None, suppress_origin=True)
        self._id = 0

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"CDP {method}: {msg['error']}")
                return msg["result"]

    def eval(self, expr: str) -> Any:
        res = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in res:
            raise RuntimeError(f"Lỗi trong trang: {json.dumps(res['exceptionDetails'], ensure_ascii=False)[:2000]}")
        return res["result"].get("value")


def tim_trinh_duyet() -> Path:
    for p in (
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    ):
        if p.exists():
            return p
    raise RuntimeError("Không tìm thấy Edge hoặc Chrome.")


def cho_url(url: str, giay: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < giay:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Không mở được {url}")


class RamSampler:
    """Lấy mẫu bộ nhớ của toàn bộ tiến trình trình duyệt (cây tiến trình) mỗi 0,5 s."""

    def __init__(self, root_pid: int) -> None:
        self.root = psutil.Process(root_pid)
        self.dinh_tong = 0.0
        self.dinh_tien_trinh = 0.0
        self._stop = False
        self._t = threading.Thread(target=self._chay, daemon=True)
        self._t.start()

    def hien_tai(self) -> tuple[float, float]:
        tong, lon_nhat = 0.0, 0.0
        for p in [self.root, *self.root.children(recursive=True)]:
            try:
                priv = p.memory_info().private / 2**20
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            tong += priv
            lon_nhat = max(lon_nhat, priv)
        return tong, lon_nhat

    def _chay(self) -> None:
        while not self._stop:
            tong, lon = self.hien_tai()
            self.dinh_tong = max(self.dinh_tong, tong)
            self.dinh_tien_trinh = max(self.dinh_tien_trinh, lon)
            time.sleep(0.5)

    def reset(self) -> None:
        self.dinh_tong, self.dinh_tien_trinh = self.hien_tai()

    def stop(self) -> None:
        self._stop = True


def do_trinh_duyet(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = doc_test()
    if args.gioi_han:
        records = records[: args.gioi_han]
    rows = [{"id": r["id"], "text": r["text"]} for r in records]
    port, cdp_port = 5179, 9339
    vite = subprocess.Popen(
        ["node", "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
        cwd=WEB_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    profile = Path(tempfile.mkdtemp(prefix="safevichi_bench_"))
    browser = None
    pin = None
    ram = None
    try:
        url = f"http://127.0.0.1:{port}/bench.html"
        cho_url(url)
        exe = tim_trinh_duyet()
        browser = subprocess.Popen([
            str(exe), "--headless=new", f"--remote-debugging-port={cdp_port}", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check", "--disable-extensions", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cho_url(f"http://127.0.0.1:{cdp_port}/json/version")
        with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/version") as r:
            phien_ban = json.loads(r.read())
        with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json/list") as r:
            tab = next(t for t in json.loads(r.read()) if t["type"] == "page")
        cdp = CDP(tab["webSocketDebuggerUrl"])
        ram = RamSampler(browser.pid)
        time.sleep(3)
        ram_trong, _ = ram.hien_tai()

        cdp.call("Page.enable")
        cdp.call("Page.navigate", url=url)
        t0 = time.time()
        while not cdp.eval("window.__benchReady === true"):
            if time.time() - t0 > 60:
                raise RuntimeError("Trang đo không sẵn sàng.")
            time.sleep(0.5)
        print("    Nạp mô hình trong trình duyệt...", flush=True)
        nap = cdp.eval("window.__bench.load()")
        time.sleep(2)
        ram_sau_nap, _ = ram.hien_tai()
        print(f"    {nap['backend']}, nạp {nap['wall_load_ms']:.0f} ms", flush=True)

        pin = PinSampler()
        pin.kiem_tra_chay_pin()
        env = moi_truong()
        nghi1 = cho_nghi(pin, args.nghi, "trước")
        ram.reset()
        print(f"    Chạy 1.670 đoạn, lặp tới khi đủ {args.tai:.0f} s...", flush=True)
        kq = cdp.eval(f"window.__bench.run({json.dumps(rows, ensure_ascii=False)}, {{minSeconds: {args.tai}, warmup: 10}})")
        tai = pin.thong_ke(kq["t_start_ms"] / 1000.0, kq["t_end_ms"] / 1000.0)
        dinh_tong, dinh_tt = ram.dinh_tong, ram.dinh_tien_trinh
        nghi2 = cho_nghi(pin, args.nghi, "sau")
        nang_luong = tinh_nang_luong(nghi1, tai, nghi2, kq["n_total"])
    finally:
        if pin:
            pin.stop()
        if ram:
            ram.stop()
        if browser:
            for p in psutil.Process(browser.pid).children(recursive=True) if psutil.pid_exists(browser.pid) else []:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            browser.kill()
        vite.kill()
        # Tắt nốt tiến trình con còn sót (crashpad của Edge dùng profile tạm, node của vite)
        for p in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                la_node_vite = (p.info["name"] or "").lower().startswith("node") and f"--port {port}" in cmd
                if profile.name in cmd or la_node_vite:
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)

    preds = kq["preds"]
    ghi_jsonl(OUT_DIR / "trinh_duyet_predictions.jsonl",
              [{"id": p["id"], "nhan": p["nhan"], "bang_chung": p["bang_chung"]} for p in preds])
    wasm = WEB_DIR / "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"
    tai_ve = [ONNX_DIR / "safevichi_v2_int8.onnx", ONNX_DIR / "tokenizer" / "tokenizer.json",
              ONNX_DIR / "crf_params.json", ONNX_DIR / "export_manifest.json", wasm]
    pkg = json.loads((WEB_DIR / "node_modules/onnxruntime-web/package.json").read_text(encoding="utf-8"))
    env["trinh_duyet"] = phien_ban.get("Browser")
    env["onnxruntime_web"] = pkg["version"]
    ket_qua = {
        "mo_hinh": "safevichi_v2 ONNX INT8",
        "lenh": "python -m src.v2.bench.do_thiet_bi trinh_duyet",
        "seed": SEED,
        "moi_truong": env,
        "cach_chay": "Edge headless, vite dev server cục bộ, trang web-v2/bench.html dùng đúng src/model.worker.js "
                     "của ứng dụng (onnxruntime-web WASM SIMD); mỗi đoạn gửi tuần tự, batch 1.",
        "nap": nap,
        "dung_luong_tai_ve_MiB": kich_thuoc_mib(*tai_ve),
        "n": len(preds),
        "do_tre_khu_hoi_ms": phan_vi([p["ms_khu_hoi"] for p in preds]),
        "do_tre_mo_hinh_ms": phan_vi([p["ms_mo_hinh"] for p in preds]),
        "thong_luong_doan_moi_giay": round(1000.0 / float(np.mean([p["ms_khu_hoi"] for p in preds])), 2),
        "ram_MiB": {
            "cach_do": "Private bytes của cả cây tiến trình trình duyệt (lấy mẫu 0,5 s)",
            "trinh_duyet_trang_trong": round(ram_trong, 1),
            "sau_khi_nap_mo_hinh": round(ram_sau_nap, 1),
            "dinh_khi_chay": round(dinh_tong, 1),
            "tang_them_so_voi_trang_trong": round(dinh_tong - ram_trong, 1),
            "dinh_tien_trinh_lon_nhat": round(dinh_tt, 1),
        },
        "so_luot": kq["passes"],
        "nang_luong": nang_luong,
        "so_khop_pytorch_fp32": so_khop(preds, REF_TORCH),
        "so_khop_onnx_int8_python": so_khop(preds, REF_INT8),
        "diem": cham_diem(records, preds),
    }
    cli_pred = OUT_DIR / "cpu_laptop_predictions.jsonl"
    if cli_pred.exists() and len(preds) == len(records):
        ket_qua["so_khop_cli_cpu_batch1"] = so_khop(preds, cli_pred)
    (OUT_DIR / "trinh_duyet.json").write_text(json.dumps(ket_qua, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ghi_env(env)
    print(json.dumps({k: ket_qua[k] for k in ("nap", "do_tre_khu_hoi_ms", "ram_MiB", "nang_luong")}, ensure_ascii=False, indent=1)[:3000])


def ghi_env(env: dict[str, Any]) -> None:
    p = OUT_DIR / "env.txt"
    cu = p.read_text(encoding="utf-8") if p.exists() else ""
    dong = [f"{k}: {v}" for k, v in env.items()]
    moi = "\n".join(dict.fromkeys(cu.splitlines() + dong)) + "\n"
    p.write_text(moi, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="lenh", required=True)
    for ten in ("cpu", "trinh_duyet"):
        s = sub.add_parser(ten)
        s.add_argument("--nghi", type=float, default=120.0, help="Giây mỗi cửa sổ nghỉ (trước và sau)")
        s.add_argument("--tai", type=float, default=180.0, help="Giây tối thiểu của cửa sổ tải")
        s.add_argument("--gioi-han", type=int, default=0, help="Chỉ chạy N đoạn đầu (chạy thử, không dùng cho báo cáo)")
    w = sub.add_parser("_cpu_worker")
    w.add_argument("--threads", type=int, required=True)
    w.add_argument("--min-giay", type=float, default=0.0)
    w.add_argument("--ghi-du-doan", type=str, default=None)
    args = ap.parse_args(argv)
    if args.lenh in ("cpu", "trinh_duyet") and sys.platform == "win32":
        # Giữ máy và màn hình luôn thức suốt lúc đo để công suất nền không đổi giữa cửa sổ nghỉ và cửa sổ tải
        import ctypes

        es_continuous, es_system, es_display = 0x80000000, 0x00000001, 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system | es_display)
    if args.lenh == "cpu":
        do_cpu(args)
    elif args.lenh == "trinh_duyet":
        do_trinh_duyet(args)
    else:
        cpu_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
