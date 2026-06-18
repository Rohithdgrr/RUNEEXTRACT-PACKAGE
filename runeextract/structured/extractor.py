"""
StructuredExtractor: extract structured data matching Pydantic schemas from documents.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type, Union

from runeextract.exceptions import ExtractionError, StructuredExtractionError
from runeextract.models.document import Document

logger = logging.getLogger(__name__)


def _model_to_json_schema(model: Type) -> Dict[str, Any]:
    """Return a JSON schema dict for a Pydantic model (v1 or v2)."""
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    if hasattr(model, "schema"):
        return model.schema()
    raise TypeError(f"{model.__name__} is not a Pydantic BaseModel")


def _model_schema_to_prompt(model: Type) -> str:
    """Build a human-readable type description from a Pydantic model's JSON schema."""
    schema = _model_to_json_schema(model)
    model_name = schema.get("title", model.__name__)
    desc = schema.get("description", "")
    defs = schema.get("$defs", {})
    parts = [f"Output must be a single JSON object conforming to the schema titled \"{model_name}\"."]
    if desc:
        parts.append(desc)

    def _resolve_ref(ref: str) -> dict:
        """Resolve a $ref string to the actual definition dict."""
        key = ref.split("/")[-1]
        return defs.get(key, {})

    def _describe_properties(props: Dict, required: List[str], indent: int = 0) -> List[str]:
        lines = []
        prefix = "  " * indent
        for name, prop in props.items():
            if "$ref" in prop:
                prop = _resolve_ref(prop["$ref"])
            typ = prop.get("type", "any")
            desc = prop.get("description", "")
            req = " (required)" if name in required else " (optional)"
            if typ == "array":
                items = prop.get("items", {})
                if "$ref" in items:
                    items = _resolve_ref(items["$ref"])
                item_type = items.get("type", "any")
                if "properties" in items:
                    sub_lines = _describe_properties(items.get("properties", {}), items.get("required", []), indent + 1)
                    lines.append(f"{prefix}- \"{name}\": array of objects{req}")
                    lines.extend(sub_lines)
                else:
                    lines.append(f"{prefix}- \"{name}\": array of {item_type}{req}")
            elif typ == "object":
                sub_lines = _describe_properties(prop.get("properties", {}), prop.get("required", []), indent + 1)
                lines.append(f"{prefix}- \"{name}\": object{req}")
                lines.extend(sub_lines)
            else:
                enum_vals = prop.get("enum")
                if enum_vals:
                    lines.append(f"{prefix}- \"{name}\": {typ} (one of {enum_vals}){req}")
                else:
                    hint = f" — {desc}" if desc else ""
                    lines.append(f"{prefix}- \"{name}\": {typ}{hint}{req}")
        return lines

    props = schema.get("properties", {})
    required = schema.get("required", [])
    if props:
        parts.append("Fields:")
        parts.extend(_describe_properties(props, required))
    return "\n".join(parts)


class StructuredExtractor:
    """Extract structured data matching a Pydantic schema using an AI model.

    Usage::

        from pydantic import BaseModel
        from runeextract.structured import StructuredExtractor

        class Person(BaseModel):
            name: str
            age: int

        extractor = StructuredExtractor()
        person = extractor.extract("Alice is 30 years old.", Person)
        # person.name == "Alice", person.age == 30
    """

    def __init__(
        self,
        ai_processor: Optional[Any] = None,
        model: Optional[str] = None,
        provider: str = "openai",
    ):
        if ai_processor is not None:
            self._ai = ai_processor
        else:
            from runeextract.processors.ai import AIProcessor
            try:
                self._ai = AIProcessor(model=model, provider=provider)
            except ExtractionError:
                self._ai = None

    @property
    def ai(self) -> Any:
        if self._ai is None:
            raise ExtractionError(
                "StructuredExtractor has no AI backend configured. "
                "Provide an AIProcessor or set OPENAI_API_KEY.",
                error_code="E030",
            )
        return self._ai

    def extract(
        self,
        source: Union[str, Document],
        schema: Type,
        max_retries: int = 3,
        raise_on_failure: bool = True,
    ) -> Any:
        """Extract structured data matching *schema* from *source*.

        Args:
            source: Document object or raw text string.
            schema: A Pydantic ``BaseModel`` subclass describing the desired shape.
            max_retries: How many times to retry on parse/validation failure.
            raise_on_failure: If True (default), raise on failure; otherwise return None.

        Returns:
            An instance of *schema* with the extracted fields populated.
        """
        text = source.text if isinstance(source, Document) else str(source)

        schema_prompt = _model_schema_to_prompt(schema)
        system = (
            "You are a structured data extraction assistant. "
            "Extract the requested information from the provided text exactly as specified. "
            "Return only valid JSON — no explanation, no markdown fences.\n\n"
            f"{schema_prompt}"
        )
        user = f"Extract the requested fields from this text:\n\n{text}"

        for attempt in range(max_retries):
            try:
                raw = self.ai._call(
                    system,
                    user,
                    response_format={"type": "json_object"},
                )
                data = json.loads(raw)
                return schema(**data)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning(
                    "Structured extraction attempt %d/%d failed: %s",
                    attempt + 1, max_retries, exc,
                )
                if attempt == max_retries - 1:
                    if raise_on_failure:
                        raise StructuredExtractionError(
                            f"Structured extraction failed after {max_retries} attempts: {exc}",
                        )
                    return None

    def extract_many(
        self,
        sources: List[Union[str, Document]],
        schema: Type,
        max_retries: int = 3,
        raise_on_failure: bool = True,
    ) -> List[Any]:
        """Extract structured data from multiple documents in sequence."""
        return [
            self.extract(s, schema, max_retries=max_retries, raise_on_failure=raise_on_failure)
            for s in sources
        ]


def extract_structured(
    source: Union[str, Document],
    schema: Type,
    ai_processor: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Convenience function for one-off structured extraction.

    Usage::

        from pydantic import BaseModel
        from runeextract.structured import extract_structured

        class Invoice(BaseModel):
            invoice_number: str
            total: float

        inv = extract_structured("Invoice #1234: $99.99", Invoice)
        # inv.invoice_number == "1234", inv.total == 99.99
    """
    extractor = StructuredExtractor(ai_processor=ai_processor)
    return extractor.extract(source, schema, **kwargs)
