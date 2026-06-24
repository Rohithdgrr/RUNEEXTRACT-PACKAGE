"""Provider registry for AI processor backends.

Each provider module exports ``call(proc, system, user, ...)``,
and optionally ``call_stream``, ``embed``, and ``create_client``.
"""

import importlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_REGISTRY = {
    "openai": "openai_compat",
    "openrouter": "openai_compat",
    "azure": "openai_compat",
    "ollama": "openai_compat",
    "groq": "openai_compat",
    "together": "openai_compat",
    "deepseek": "openai_compat",
    "mistral": "openai_compat",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "bedrock": "bedrock",
    "local": "local",
}


def _load(provider: str):
    mod = _REGISTRY.get(provider)
    if mod is None:
        raise ValueError(f"Unknown provider: {provider}")
    return importlib.import_module(f"runeextract.processors.providers.{mod}")


def call(provider: str, proc, system: str, user: str,
         response_format=None, max_tokens=None) -> str:
    return _load(provider).call(proc, system, user,
                                response_format=response_format, max_tokens=max_tokens)


def call_stream(provider: str, proc, system: str, user: str, max_tokens=None):
    mod = _load(provider)
    if hasattr(mod, "call_stream") and not getattr(mod, '_NO_STREAM', False):
        yield from mod.call_stream(proc, system, user, max_tokens=max_tokens)
    else:
        yield mod.call(proc, system, user, max_tokens=max_tokens)


def embed(provider: str, proc, texts, model=None):
    mod = _load(provider)
    if hasattr(mod, "embed"):
        return mod.embed(proc, texts, model=model)
    raise ValueError(f"Embedding not supported for provider '{provider}'")


def create_client(provider: str, proc):
    mod = _load(provider)
    if hasattr(mod, "create_client"):
        return mod.create_client(proc)
    raise ValueError(f"Client creation not supported for provider '{provider}'")
