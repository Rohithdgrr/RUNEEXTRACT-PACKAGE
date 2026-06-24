"""Google Gemini provider."""

import logging

logger = logging.getLogger(__name__)

_NO_STREAM = True


def create_client(proc):
    import google.generativeai as genai
    genai.configure(api_key=proc.api_key)
    return genai.GenerativeModel(proc.model)


def call(proc, system, user, response_format=None, max_tokens=None):
    generation_config = {
        "temperature": proc.temperature,
        "max_output_tokens": max_tokens or proc.max_tokens,
    }
    resp = proc._call_with_retry(
        lambda: proc.client.generate_content(
            f"{system}\n\n{user}",
            generation_config=generation_config,
            request_options={"timeout": getattr(proc, "_request_timeout", 60)},
        ),
        lambda r: (0, 0),
        provider_label="Gemini",
    )
    return resp.text.strip()
