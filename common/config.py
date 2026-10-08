"""Run configuration, paths, prices, marker list, and LIMIT resolution."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent  # assignment1/
OFFICIAL_DIR: Path = Path(
    os.environ.get("SPIDER_EVAL_DIR", "/home/nguyen/School/CISC_451/test-suite-sql-eval")
)
DATA_DIR: Path = ROOT_DIR / "data" / "spider"

SELF_CORRECTION_MARKERS: tuple[str, ...] = (
    "wait", "actually", "let me check", "double-check", "alternatively", "hmm",
)

# USD per million tokens. Fill in from the OpenRouter model page and record source/date.
PRICES: dict[str, dict] = {
    "qwen/qwen-2.5-7b-instruct": {
        "input_per_m": None, "output_per_m": None,
        "source": "https://openrouter.ai/qwen/qwen-2.5-7b-instruct", "date": None,
    },
    "qwen/qwen-2.5-72b-instruct": {
        "input_per_m": None, "output_per_m": None,
        "source": "https://openrouter.ai/qwen/qwen-2.5-72b-instruct", "date": None,
    },
    "qwen/qwen3-32b": {
        "input_per_m": None, "output_per_m": None,
        "source": "https://openrouter.ai/qwen/qwen3-32b", "date": None,
    },
}

MODEL_META: dict[str, dict] = {
    "qwen/qwen-2.5-7b-instruct": {"params": "7B", "family": "Qwen2.5", "type": "instruct", "reasoning": False},
    "qwen/qwen-2.5-72b-instruct": {"params": "72B", "family": "Qwen2.5", "type": "instruct", "reasoning": False},
    "qwen/qwen3-32b": {"params": "32B", "family": "Qwen3", "type": "hybrid-thinking", "reasoning": True},
}

_NON_FINGERPRINT_FIELDS = frozenset({"threads", "limit", "request_timeout_s"})


def resolve_limit(value: int | None) -> int | None:
    """Notebook value if set, else A1_LIMIT env var, else None."""
    if value is not None:
        return int(value)
    env = os.environ.get("A1_LIMIT", "").strip()
    return int(env) if env else None


def out_root(limit: int | None) -> Path:
    return ROOT_DIR / ("outputs" if limit is None else f"outputs_limit{limit}")


def results_root(limit: int | None) -> Path:
    return ROOT_DIR / ("results" if limit is None else f"results_limit{limit}")


def load_api_key(required: bool) -> str | None:
    """OPENROUTER_API_KEY from env, then from .env via python-dotenv."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:
            from dotenv import dotenv_values
            for p in (ROOT_DIR / ".env", Path.cwd() / ".env"):
                if p.is_file():
                    key = dotenv_values(p).get("OPENROUTER_API_KEY")
                    if key:
                        break
        except ImportError:
            pass
    if not key and required:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it or add it to assignment1/.env."
        )
    return key or None


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    model_id: str
    temperature: float
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 512
    reasoning_enabled: bool = False
    provider_order: tuple[str, ...] | None = None
    allow_fallbacks: bool = False
    require_parameters: bool = True
    seed: int | None = 0
    prompt_version: str = "v1"
    threads: int = 8
    limit: int | None = None
    max_attempts: int = 6
    request_timeout_s: float = 300.0
    preflight_n: int = 8
    retry_on_empty: bool = False
    egsc_max_retries: int = 2
    egsc_exec_timeout_s: float = 5.0
    sample_rows: int = 3
    value_trunc: int = 64

    def __post_init__(self) -> None:
        if self.provider_order is not None and not isinstance(self.provider_order, tuple):
            object.__setattr__(self, "provider_order", tuple(self.provider_order))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if d["provider_order"] is not None:
            d["provider_order"] = list(d["provider_order"])
        return d

    def fingerprint(self) -> str:
        d = {k: v for k, v in self.to_dict().items() if k not in _NON_FINGERPRINT_FIELDS}
        canon = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def output_dir(self) -> Path:
        return out_root(self.limit) / self.run_name
