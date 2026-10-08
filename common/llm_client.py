"""OpenRouter LLM client: request mapping, response parsing, retries, endpoint metadata."""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Protocol

from .config import RunConfig, load_api_key

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is needed but not set (Req 1.6)."""

    def __init__(self, var: str = "OPENROUTER_API_KEY") -> None:
        self.var = var
        super().__init__(f"{var} is not set. Export it or add it to assignment1/.env.")


@dataclass
class LLMResult:
    status: Literal["ok", "error"]
    text: str | None = None
    reasoning: str | None = None
    reasoning_source: Literal["details_text", "reasoning_field", "inline_think"] | None = None
    reasoning_detail_types: list[str] = field(default_factory=list)
    reasoning_missing: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float = 0.0
    provider: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    generation_id: str | None = None
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], cfg: RunConfig) -> LLMResult: ...


# --------------------------------------------------------------------------- pure helpers

def build_request(messages: list[dict], cfg: RunConfig) -> dict:
    """Map Run_Config to chat.completions.create kwargs (Req 6.2-6.5). Pure."""
    req: dict[str, Any] = {
        "model": cfg.model_id,
        "messages": list(messages),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.top_p is not None:
        req["top_p"] = cfg.top_p
    if cfg.seed is not None:
        req["seed"] = cfg.seed
    extra: dict[str, Any] = {}
    if cfg.top_k is not None:
        extra["top_k"] = cfg.top_k
    if cfg.provider_order:
        extra["provider"] = {
            "order": list(cfg.provider_order),
            "allow_fallbacks": bool(cfg.allow_fallbacks),
            "require_parameters": bool(cfg.require_parameters),
        }
    if cfg.reasoning_enabled:
        extra["reasoning"] = {"enabled": True, "exclude": False}
    if extra:
        req["extra_body"] = extra
    return req


def _get(d: Any, key: str) -> Any:
    if isinstance(d, dict):
        return d.get(key)
    return getattr(d, key, None)


def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def parse_response(raw: dict, latency_s: float, attempts: int, cfg: RunConfig) -> LLMResult:
    """Parse an OpenRouter chat completion dict into an LLMResult (Req 6.1, 6.5, 6.6). Pure."""
    raw = raw or {}
    choices = _get(raw, "choices") or []
    choice = choices[0] if choices else {}
    message = _get(choice, "message") or {}
    text = _get(message, "content")
    if text is not None and not isinstance(text, str):
        text = str(text)

    details = _get(message, "reasoning_details") or []
    if not isinstance(details, list):
        details = []
    detail_types = [str(_get(d, "type")) for d in details if _get(d, "type") is not None]
    detail_texts = [
        _get(d, "text") for d in details
        if _get(d, "type") == "reasoning.text" and isinstance(_get(d, "text"), str)
    ]

    reasoning: str | None = None
    source = None
    if detail_texts and "".join(detail_texts).strip():
        reasoning, source = "".join(detail_texts), "details_text"
    else:
        rf = _get(message, "reasoning")
        if isinstance(rf, str) and rf.strip():
            reasoning, source = rf, "reasoning_field"
        elif isinstance(text, str):
            blocks = _THINK_RE.findall(text)
            if blocks and "".join(blocks).strip():
                reasoning, source = "\n".join(b.strip() for b in blocks), "inline_think"
    if isinstance(text, str) and _THINK_RE.search(text):
        text = _THINK_RE.sub("", text).strip()

    usage = _get(raw, "usage") or {}
    ctd = _get(usage, "completion_tokens_details") or {}
    return LLMResult(
        status="ok",
        text=text,
        reasoning=reasoning,
        reasoning_source=source,
        reasoning_detail_types=detail_types,
        reasoning_missing=bool(cfg.reasoning_enabled and not (reasoning or "").strip()),
        prompt_tokens=_int_or_none(_get(usage, "prompt_tokens")),
        completion_tokens=_int_or_none(_get(usage, "completion_tokens")),
        reasoning_tokens=_int_or_none(_get(ctd, "reasoning_tokens")),
        latency_s=float(latency_s),
        provider=_get(raw, "provider"),
        model=_get(raw, "model"),
        finish_reason=_get(choice, "finish_reason"),
        generation_id=_get(raw, "id"),
        attempts=attempts,
        error=None,
    )


def backoff_delay(attempt: int, base: float = 2.0, cap: float = 60.0, jitter: float = 0.25) -> float:
    """Exponential backoff for retry `attempt` (1-based): min(cap, base**attempt) * (1 ± jitter)."""
    d = min(cap, base ** max(0, attempt))
    if jitter:
        d *= 1.0 + random.uniform(-jitter, jitter)
    return max(0.0, min(cap, d))


# --------------------------------------------------------------------------- client

class _RetryableResponse(Exception):
    """200 response with an error body or empty choices."""


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = OPENROUTER_BASE_URL,
                 sleep: Callable[[float], None] = time.sleep, client: Any = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.sleep = sleep
        self._client = client  # injectable transport (tests); built lazily otherwise

    def _get_client(self, timeout: float) -> Any:
        if self._client is None:
            key = self.api_key or load_api_key(required=False)
            if not key:
                raise MissingAPIKeyError("OPENROUTER_API_KEY")
            from openai import OpenAI
            self.api_key = key
            self._client = OpenAI(api_key=key, base_url=self.base_url,
                                  max_retries=0, timeout=timeout)
        return self._client

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, _RetryableResponse):
            return True
        try:
            import openai
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                            openai.APITimeoutError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            code = getattr(exc, "status_code", None)
            return code == 429 or (code is not None and code >= 500)
        return False

    def complete(self, messages: list[dict], cfg: RunConfig) -> LLMResult:
        client = self._get_client(cfg.request_timeout_s)
        req = build_request(messages, cfg)
        max_attempts = max(1, int(cfg.max_attempts))
        last_err = "unknown error"
        t_total = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = client.chat.completions.create(**req)
                latency = time.perf_counter() - t0
                raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                if raw.get("error"):
                    raise _RetryableResponse(f"error body: {json.dumps(raw['error'], default=str)}")
                if not raw.get("choices"):
                    raise _RetryableResponse("empty choices in 200 response")
                return parse_response(raw, latency, attempt, cfg)
            except MissingAPIKeyError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                last_err = f"{type(exc).__name__}: {exc}"
                if not self._is_retryable(exc):
                    return LLMResult(status="error", error=last_err, attempts=attempt,
                                     latency_s=time.perf_counter() - t_total,
                                     model=cfg.model_id)
                if attempt < max_attempts:
                    self.sleep(backoff_delay(attempt))
        return LLMResult(status="error", error=last_err, attempts=max_attempts,
                         latency_s=time.perf_counter() - t_total, model=cfg.model_id)


def fetch_endpoints(model_id: str, timeout: float = 30.0) -> list[dict]:
    """GET /models/{model_id}/endpoints (no key needed). Network call."""
    url = f"{OPENROUTER_BASE_URL}/models/{model_id}/endpoints"
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    eps = (payload.get("data") or {}).get("endpoints") or []
    return [
        {
            "provider_name": e.get("provider_name"),
            "quantization": e.get("quantization"),
            "context_length": e.get("context_length"),
            "pricing": e.get("pricing"),
        }
        for e in eps
    ]
