"""OpenAI-compatible provider: OpenAI, OpenRouter, Azure, Ollama, Groq, Together, DeepSeek, Mistral."""

import os
import logging

from runeextract.exceptions import ExtractionError, DependencyMissingError


logger = logging.getLogger(__name__)

_NO_STREAM = False


def create_client(proc):
    p = proc.provider
    if p in ("openai", "openrouter"):
        from openai import OpenAI
        return OpenAI(
            api_key=proc.api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    elif p == "ollama":
        from openai import OpenAI as OllamaClient
        return OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama")
    elif p == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=proc.api_key,
            api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-02-01"),
            azure_endpoint=getattr(proc, "_azure_endpoint", "") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        )
    elif p == "groq":
        from groq import Groq
        return Groq(api_key=proc.api_key)
    elif p == "together":
        from openai import OpenAI as TogetherClient
        return TogetherClient(api_key=proc.api_key, base_url="https://api.together.xyz/v1")
    elif p == "deepseek":
        from openai import OpenAI as DeepSeekClient
        return DeepSeekClient(api_key=proc.api_key, base_url="https://api.deepseek.com")
    elif p == "mistral":
        from mistralai import Mistral
        return Mistral(api_key=proc.api_key)
    raise DependencyMissingError(f"(ai)", "openai")


def call(proc, system, user, response_format=None, max_tokens=None):
    kwargs = {
        "model": proc.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": proc.temperature,
        "max_tokens": max_tokens or proc.max_tokens,
        "timeout": getattr(proc, "_request_timeout", 60),
    }
    if response_format:
        kwargs["response_format"] = response_format
    resp = proc._call_with_retry(
        lambda: proc.client.chat.completions.create(**kwargs),
        lambda r: (r.usage.prompt_tokens if r.usage else 0,
                   r.usage.completion_tokens if r.usage else 0),
        provider_label=proc.provider.title(),
    )
    return resp.choices[0].message.content.strip()


def call_stream(proc, system, user, max_tokens=None):
    kwargs = {
        "model": proc.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": proc.temperature,
        "max_tokens": max_tokens or proc.max_tokens,
        "stream": True,
        "timeout": getattr(proc, "_request_timeout", 60),
    }
    collected = []
    try:
        stream = proc.client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                collected.append(delta.content)
                yield delta.content
        if hasattr(chunk, "usage") and chunk.usage:
            proc._track_cost(
                input_tokens=chunk.usage.prompt_tokens or 0,
                output_tokens=chunk.usage.completion_tokens or 0,
            )
        proc._record_success()
    except Exception as exc:
        if collected:
            yield "".join(collected)
        proc._record_failure()
        safe_msg = proc._sanitize_error(exc)
        logger.error(f"Streaming call failed: {safe_msg[:200]}")
        raise ExtractionError(f"AI streaming failed: {safe_msg[:500]}", error_code="E031")


def embed(proc, texts, model=None):
    model = model or ("text-embedding-3-small" if proc.provider == "openai" else "nomic-embed-text")
    try:
        resp = proc.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in resp.data]
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        raise ExtractionError(f"Embedding failed: {exc}", error_code="E031")
