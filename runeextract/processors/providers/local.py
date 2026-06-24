"""Local transformers provider."""

import logging

from runeextract.exceptions import ExtractionError, DependencyMissingError

logger = logging.getLogger(__name__)

_NO_STREAM = True


def create_client(proc):
    return None


def call(proc, system, user, response_format=None, max_tokens=None):
    try:
        from transformers import pipeline
    except ImportError:
        raise DependencyMissingError("(local AI)", "transformers")

    if proc._local_pipeline is None:
        try:
            proc._local_pipeline = pipeline(
                "text-generation",
                model=proc.model,
                max_new_tokens=proc.max_tokens,
            )
        except Exception as exc:
            raise ExtractionError(
                f"Failed to load local model '{proc.model}': {exc}",
                error_code="E032",
            )

    prompt = f"{system}\n\n{user}\n\nAnswer:"
    result = proc._local_pipeline(
        prompt,
        max_new_tokens=proc.max_tokens,
        temperature=proc.temperature,
        do_sample=True,
    )
    proc._track_cost(input_tokens=0, output_tokens=0)
    return result[0]["generated_text"].replace(prompt, "").strip()


def embed(proc, texts, model=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise DependencyMissingError("(embeddings)", "sentence-transformers")
    model_name = model or "all-MiniLM-L6-v2"
    try:
        encoder = SentenceTransformer(model_name)
        embeddings = encoder.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    except Exception as exc:
        logger.error(f"Local embedding failed: {exc}")
        raise ExtractionError(f"Local embedding failed: {exc}", error_code="E032")
