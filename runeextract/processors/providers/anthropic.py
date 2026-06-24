"""Anthropic Claude provider."""

import logging

from runeextract.exceptions import DependencyMissingError


logger = logging.getLogger(__name__)

_NO_STREAM = True


def create_client(proc):
    import anthropic
    return anthropic.Anthropic(api_key=proc.api_key)


def call(proc, system, user, response_format=None, max_tokens=None):
    resp = proc._call_with_retry(
        lambda: proc.client.messages.create(
            model=proc.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=proc.temperature,
            max_tokens=max_tokens or proc.max_tokens,
            timeout=getattr(proc, "_request_timeout", 60),
        ),
        lambda r: (
            getattr(r, "usage", None).input_tokens if getattr(r, "usage", None) else 0,
            getattr(r, "usage", None).output_tokens if getattr(r, "usage", None) else 0,
        ),
        provider_label="Anthropic",
    )
    return resp.content[0].text.strip()
