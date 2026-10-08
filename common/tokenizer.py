"""Token counting for reasoning traces (Task 2).

Provider-reported `reasoning_tokens` is an estimate on some providers (DeepInfra) and can exceed
`completion_tokens`, so trace length is measured here from the trace text with the Qwen3 tokenizer.
"""
from __future__ import annotations

from functools import lru_cache

TOKENIZER_NAME = "Qwen/Qwen3-32B"


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer  # lightweight (Rust) library; no torch/transformers needed

    return Tokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Number of Qwen3 tokens in `text` (no special tokens added)."""
    if not text:
        return 0
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


def tokenizer_info() -> dict:
    import tokenizers

    return {"tokenizer": TOKENIZER_NAME, "tokenizers_version": tokenizers.__version__}
