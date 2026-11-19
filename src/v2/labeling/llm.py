"""Lớp gọi LLM dùng chung cho prelabel.py và generate.py.

Cấu hình qua biến môi trường (hoặc file .env ở gốc repo, không commit):
    SAFEVICHI_LLM_PROVIDER   anthropic | openai | gemini   (bắt buộc, không có mặc định)
    SAFEVICHI_LLM_MODEL      tên model của nhà cung cấp     (bắt buộc)
    SAFEVICHI_LLM_BASE_URL   tùy chọn, cho API tương thích OpenAI (OpenRouter, DeepSeek, ...)
    ANTHROPIC_API_KEY | OPENAI_API_KEY | GEMINI_API_KEY   khóa tương ứng
    (hoặc SAFEVICHI_LLM_API_KEY dùng chung cho mọi nhà cung cấp)
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]

def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Đọc .env đơn giản (KEY=VALUE); không ghi đè biến đã có."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0


def config_from_env() -> LLMConfig:
    provider = os.environ.get("SAFEVICHI_LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("SAFEVICHI_LLM_MODEL", "").strip()
    if provider not in KEY_ENV:
        raise SystemExit(
            "Chưa chọn nhà cung cấp: đặt SAFEVICHI_LLM_PROVIDER là một trong "
            f"{', '.join(KEY_ENV)} (trong biến môi trường hoặc .env)."
        )
    if not model:
        raise SystemExit("Chưa đặt SAFEVICHI_LLM_MODEL.")
    api_key = os.environ.get("SAFEVICHI_LLM_API_KEY") or os.environ.get(KEY_ENV[provider], "")
    if not api_key:
        raise SystemExit(f"Thiếu khóa API: đặt {KEY_ENV[provider]} hoặc SAFEVICHI_LLM_API_KEY.")
    return LLMConfig(provider, model, api_key, os.environ.get("SAFEVICHI_LLM_BASE_URL") or None)


def call_llm(config: LLMConfig, system: str, user: str) -> str:
    """Gọi API qua HTTP, trả về văn bản phản hồi. Không log khóa."""
    import requests

    if config.provider == "anthropic":
        url = (config.base_url or "https://api.anthropic.com") + "/v1/messages"
        response = requests.post(
            url,
            headers={"x-api-key": config.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": config.model, "max_tokens": config.max_tokens, "temperature": config.temperature,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            timeout=config.timeout,
        )
        response.raise_for_status()
        return "".join(b.get("text", "") for b in response.json()["content"] if b.get("type") == "text")
    if config.provider == "openai":
        url = (config.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            json={"model": config.model, "temperature": config.temperature,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""
    if config.provider == "gemini":
        base = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        response = requests.post(
            f"{base}/models/{config.model}:generateContent",
            headers={"x-goog-api-key": config.api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": config.temperature}},
            timeout=config.timeout,
        )
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    raise ValueError(f"Nhà cung cấp không hỗ trợ: {config.provider}")


def with_retries(fn: Callable[[], str], attempts: int = 4) -> str:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - lỗi mạng/HTTP đều thử lại
            if attempt == attempts - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"Lỗi gọi API ({type(exc).__name__}), thử lại sau {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
