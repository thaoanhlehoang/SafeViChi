"""Lớp gọi LLM dùng chung cho prelabel.py và generate.py.

Cấu hình qua biến môi trường (hoặc file .env ở gốc repo, không commit):
    SAFEVICHI_LLM_PROVIDER   anthropic | openai | gemini   (bắt buộc, không có mặc định)
    SAFEVICHI_LLM_MODEL      tên model của nhà cung cấp     (bắt buộc)
    SAFEVICHI_LLM_BASE_URL   tùy chọn, cho API tương thích OpenAI (OpenRouter, DeepSeek, ...)
    ANTHROPIC_API_KEY | OPENAI_API_KEY | GEMINI_API_KEY   khóa tương ứng
    (hoặc SAFEVICHI_LLM_API_KEY dùng chung cho mọi nhà cung cấp)

Chỉ áp dụng cho OpenRouter (provider=openai, base_url là openrouter.ai):
    SAFEVICHI_LLM_REASONING        mức suy luận: none (tắt thinking) | low | high | ...; để trống = mặc định của model
    SAFEVICHI_LLM_PROVIDER_ORDER   endpoint được phép dùng, ngăn bằng dấu phẩy (vd. wafer/us);
                                   đặt thì tắt chuyển sang nhà cung cấp khác (allow_fallbacks=false)
    SAFEVICHI_LLM_MAX_PRICE        giá trần USD/1 triệu token "vào,ra" (vd. 0.04,0.5); vượt thì OpenRouter từ chối
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
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
    reasoning: str | None = None
    provider_order: list[str] = field(default_factory=list)
    max_price: tuple[float, float] | None = None

    @property
    def is_openrouter(self) -> bool:
        return self.provider == "openai" and "openrouter.ai" in (self.base_url or "")

    def describe(self) -> dict:
        """Thông tin cấu hình để ghi kèm kết quả (không có khóa)."""
        return {"reasoning": self.reasoning, "provider_order": self.provider_order, "max_price": self.max_price}


class UsageCounter:
    """Đếm token đã dùng (an toàn khi gọi song song)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls = self.prompt = self.completion = self.reasoning = 0

    def add(self, usage: dict) -> None:
        with self.lock:
            self.calls += 1
            self.prompt += int(usage.get("prompt_tokens") or 0)
            self.completion += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning += int(details.get("reasoning_tokens") or 0)

    def summary(self) -> str:
        return (f"{self.calls} lượt gọi, {self.prompt:,} token vào, {self.completion:,} token ra "
                f"(trong đó {self.reasoning:,} token suy luận)")


USAGE = UsageCounter()
_warned_reasoning = False


def _parse_max_price(value: str) -> tuple[float, float] | None:
    value = value.strip()
    if not value:
        return None
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 2:
        raise SystemExit("SAFEVICHI_LLM_MAX_PRICE phải có dạng 'vào,ra', ví dụ 0.04,0.5")
    return parts[0], parts[1]


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
    config = LLMConfig(
        provider, model, api_key, os.environ.get("SAFEVICHI_LLM_BASE_URL") or None,
        reasoning=os.environ.get("SAFEVICHI_LLM_REASONING", "").strip().lower() or None,
        provider_order=[x.strip() for x in os.environ.get("SAFEVICHI_LLM_PROVIDER_ORDER", "").split(",") if x.strip()],
        max_price=_parse_max_price(os.environ.get("SAFEVICHI_LLM_MAX_PRICE", "")),
    )
    if (config.reasoning or config.provider_order or config.max_price) and not config.is_openrouter:
        raise SystemExit("SAFEVICHI_LLM_REASONING / PROVIDER_ORDER / MAX_PRICE chỉ dùng được với OpenRouter "
                         "(SAFEVICHI_LLM_PROVIDER=openai, SAFEVICHI_LLM_BASE_URL=https://openrouter.ai/api/v1).")
    return config


def openai_body(config: LLMConfig, system: str, user: str) -> dict:
    """Thân request kiểu OpenAI; thêm tham số riêng của OpenRouter nếu có cấu hình."""
    body: dict = {"model": config.model, "temperature": config.temperature, "max_tokens": config.max_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if config.is_openrouter:
        body["usage"] = {"include": True}
        if config.reasoning:
            body["reasoning"] = {"effort": config.reasoning, "exclude": True}
        provider: dict = {}
        if config.provider_order:
            provider["order"] = config.provider_order
            provider["allow_fallbacks"] = False
        if config.max_price:
            provider["max_price"] = {"prompt": config.max_price[0], "completion": config.max_price[1]}
        if provider:
            body["provider"] = provider
    return body


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
            json=openai_body(config, system, user),
            timeout=config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Lỗi từ API: {data['error']}")
        usage = data.get("usage") or {}
        USAGE.add(usage)
        global _warned_reasoning
        reasoning_tokens = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        if config.reasoning == "none" and reasoning_tokens and not _warned_reasoning:
            _warned_reasoning = True
            print(f"CẢNH BÁO: đã đặt REASONING=none nhưng vẫn có {reasoning_tokens} token suy luận "
                  f"(nhà cung cấp: {data.get('provider')}). Dừng lại kiểm tra chi phí.", file=sys.stderr)
        return data["choices"][0]["message"]["content"] or ""
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
