"""
Embedding model auto-selection.

Maps human-readable quality levels (``"fast"``, ``"balanced"``, ``"accurate"``)
to concrete ``provider:model`` strings, with domain-aware overrides.
"""

from runeextract.rag.templates import DomainTemplates

_QUALITY_MAP = {
    "auto": "openai:text-embedding-3-large",
    "fast": "openai:text-embedding-3-small",
    "balanced": "openai:text-embedding-3-large",
    "accurate": "openai:text-embedding-3-large",
}

_DOMAIN_OVERRIDE = {
    "medical": "openai:text-embedding-3-large",
}


def resolve_embedding(spec: str) -> str:
    """Resolve a quality level or domain name to a concrete embedding spec.

    Args:
        spec: ``"fast"``, ``"balanced"``, ``"accurate"``, or any valid
            ``provider:model`` string (passed through unchanged).

    Returns:
        A ``provider:model`` string.
    """
    if spec in _QUALITY_MAP:
        return _QUALITY_MAP[spec]
    return spec


def get_domain_embedding(domain: str) -> str:
    """Return the recommended embedding spec for a domain.

    Falls back to the domain template's ``embedding`` field, then to
    ``"balanced"`` if the domain is unknown.
    """
    if domain in _DOMAIN_OVERRIDE:
        return _DOMAIN_OVERRIDE[domain]
    config = DomainTemplates.get(domain)
    return resolve_embedding(config.embedding)
