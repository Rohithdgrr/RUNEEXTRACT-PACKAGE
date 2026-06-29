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


def supports_vision(provider: str, model: str) -> bool:
    """Return True if the provider+model supports image inputs."""
    mod = _load(provider)
    if hasattr(mod, "supports_vision"):
        try:
            return mod.supports_vision(model)
        except Exception:
            return False
    # Default: check common vision model name patterns
    vision_patterns = ["gpt-4o", "gpt-4-vision", "claude-3-5", "claude-3-opus",
                       "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2",
                       "llava", "vision"]
    return any(p in model.lower() for p in vision_patterns)


def vision_call(provider: str, proc, system: str, user: str,
                images: list, max_tokens=None) -> str:
    """Call the LLM with image content blocks.

    Args:
        provider: Provider name.
        proc: AIProcessor instance.
        system: System prompt.
        user: User text prompt.
        images: List of (image_bytes: bytes, image_format: str) tuples.
        max_tokens: Optional max output tokens.

    Returns:
        The model's text response.
    """
    mod = _load(provider)
    if hasattr(mod, "vision_call"):
        return mod.vision_call(proc, system, user, images, max_tokens=max_tokens)
    # Fallback: describe images inline
    descriptions = []
    for img_bytes, img_fmt in images:
        import base64
        b64 = base64.b64encode(img_bytes).decode("ascii")
        descriptions.append(f"[Image: data:image/{img_fmt};base64,{b64[:80]}...]")
    desc_text = "\n".join(descriptions) if descriptions else ""
    combined = f"{user}\n\n{desc_text}" if desc_text else user
    return call(provider, proc, system, combined, max_tokens=max_tokens)
