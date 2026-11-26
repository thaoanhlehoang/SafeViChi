"""Unit test cho mô-đun tự động tải mô hình từ Hugging Face (`src/v2/download_model.py`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.v2 import download_model as dm
from src.v2 import predict as pr


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_required_and_missing_onnx_files(tmp_path):
    assert dm.INT8_ONNX_FILENAME in dm.required_onnx_files(use_fp32=False)
    assert dm.FP32_ONNX_FILENAME not in dm.required_onnx_files(use_fp32=False)
    assert dm.FP32_ONNX_FILENAME in dm.required_onnx_files(use_fp32=True)
    assert len(dm.missing_onnx_files(tmp_path, use_fp32=False)) == len(dm.required_onnx_files(use_fp32=False))


def test_ensure_onnx_model_downloads_only_int8_and_verifies_sha256(tmp_path, monkeypatch):
    fake_int8 = b"fake-int8-onnx-bytes"
    fake_crf = b'{"labels": ["AN_TOAN"]}'
    fake_manifest = json.dumps(
        {
            "model_name": "safevichi_v2",
            "sha256": {
                "onnx_int8": _sha256_bytes(fake_int8),
                "crf_params": _sha256_bytes(fake_crf),
            },
        }
    ).encode("utf-8")
    payloads = {
        "onnx/safevichi_v2_int8.onnx": fake_int8,
        "onnx/crf_params.json": fake_crf,
        "onnx/export_manifest.json": fake_manifest,
        "onnx/tokenizer/tokenizer.json": b"{}",
        "onnx/tokenizer/tokenizer_config.json": b"{}",
        "onnx/tokenizer/backbone_config.json": b"{}",
    }
    calls: list[str] = []

    def fake_download(repo_id: str, remote_path: str, local_dest: Path, revision: str = "main") -> Path:
        calls.append(remote_path)
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        local_dest.write_bytes(payloads[remote_path])
        return local_dest

    monkeypatch.setattr(dm, "_download_one_hf", fake_download)

    out_dir = dm.ensure_onnx_model(tmp_path / "onnx", use_fp32=False, quiet=True)
    assert out_dir == tmp_path / "onnx"
    assert set(calls) == set(payloads.keys())
    assert "onnx/safevichi_v2_fp32.onnx" not in calls
    assert dm.missing_onnx_files(out_dir, use_fp32=False) == []

    # Lần gọi thứ 2 không tải lại gì (hoàn toàn offline)
    calls.clear()
    dm.ensure_onnx_model(out_dir, use_fp32=False, quiet=True)
    assert calls == []


def test_ensure_onnx_model_rejects_corrupted_sha256(tmp_path, monkeypatch):
    fake_manifest = json.dumps(
        {
            "model_name": "safevichi_v2",
            "sha256": {"onnx_int8": "0" * 64},
        }
    ).encode("utf-8")

    def fake_download(repo_id: str, remote_path: str, local_dest: Path, revision: str = "main") -> Path:
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        local_dest.write_bytes(fake_manifest if remote_path.endswith("export_manifest.json") else b"corrupt")
        return local_dest

    monkeypatch.setattr(dm, "_download_one_hf", fake_download)

    with pytest.raises(RuntimeError, match="SHA-256"):
        dm.ensure_onnx_model(tmp_path / "onnx", use_fp32=False, quiet=True)
    assert not (tmp_path / "onnx" / dm.INT8_ONNX_FILENAME).exists()


def test_cli_download_flag_and_offline_guard(tmp_path, monkeypatch, capsys):
    downloaded_dirs: list[Path] = []

    def fake_ensure(onnx_dir, use_fp32=False, repo_id="", revision="main", force=False, quiet=False):
        downloaded_dirs.append(Path(onnx_dir))
        return Path(onnx_dir)

    monkeypatch.setattr(pr, "ensure_onnx_model", fake_ensure)

    # --download không kèm tin nhắn: tải xong trả 0
    target = tmp_path / "custom_onnx"
    rc = pr.main(["--download", "--onnx-dir", str(target), "--no-color"])
    assert rc == 0
    assert downloaded_dirs == [target]

    # --download cùng --no-download báo lỗi rõ ràng
    rc_bad = pr.main(["--download", "--no-download", "--no-color"])
    assert rc_bad == 1
    assert "Lỗi:" in capsys.readouterr().err

    # Chế độ mặc định (không truyền --onnx-dir) khi thiếu tệp sẽ bật auto_download=True
    captured_kwargs: dict = {}

    class DummyPredictor:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            raise FileNotFoundError("stop_after_init")

    monkeypatch.setattr(pr, "SafeViChiPredictor", DummyPredictor)
    monkeypatch.delenv("SAFEVICHI_OFFLINE", raising=False)
    pr.main(["--no-color", "xin chào"])
    assert captured_kwargs["auto_download"] is True
    assert captured_kwargs["hf_repo_id"] == dm.DEFAULT_HF_REPO_ID

    # Truyền --no-download -> auto_download=False
    captured_kwargs.clear()
    pr.main(["--no-download", "--no-color", "xin chào"])
    assert captured_kwargs["auto_download"] is False
