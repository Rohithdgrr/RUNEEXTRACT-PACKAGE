"""Google Gemini provider."""

import logging

logger = logging.getLogger(__name__)

_NO_STREAM = True


def create_client(proc):
    import google.generativeai as genai
    genai.configure(api_key=proc.api_key)
    return genai.GenerativeModel(proc.model)


_VISION_MODELS = {"gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0"}


def supports_vision(model: str) -> bool:
    return any(v in model.lower() for v in _VISION_MODELS)


def vision_call(proc, system, user, images, max_tokens=None):
    import google.generativeai as genai
    contents = [f"{system}\n\n{user}"]
    for img_bytes, img_fmt in images:
        import base64
        img_data = base64.b64decode(base64.b64encode(img_bytes).decode("ascii"))
        contents.append(genai.upload_file(img_data, mime_type=f"image/{img_fmt}"))
    generation_config = {
        "temperature": proc.temperature,
        "max_output_tokens": max_tokens or proc.max_tokens,
    }
    resp = proc._call_with_retry(
        lambda: proc.client.generate_content(
            contents,
            generation_config=generation_config,
            request_options={"timeout": getattr(proc, "_request_timeout", 60)},
        ),
        lambda r: (0, 0),
        provider_label="Gemini",
    )
    return resp.text.strip()


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
