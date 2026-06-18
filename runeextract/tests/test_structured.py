"""Tests for structured extraction with Pydantic schemas."""

import json
from unittest.mock import patch, MagicMock

import pytest

from runeextract.structured import StructuredExtractor, extract_structured
from runeextract.structured.extractor import _model_to_json_schema, _model_schema_to_prompt
from runeextract.exceptions import ExtractionError, StructuredExtractionError
from runeextract.models.document import Document
from runeextract.processors.ai import AIProcessor


# --- Pydantic test models ---

@pytest.fixture(scope="session")
def pydantic():
    try:
        from pydantic import BaseModel
        return BaseModel
    except ImportError:
        pytest.skip("pydantic not installed")


def make_model(model_name, **fields):
    from pydantic import create_model
    return create_model(model_name, **fields)


class TestModelToJsonSchema:
    def test_flat_model(self, pydantic):
        M = make_model("Person", full_name=(str, ...), age=(int, 0))
        schema = _model_to_json_schema(M)
        assert "properties" in schema
        assert schema["properties"]["full_name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"

    def test_optional_field_not_required(self, pydantic):
        M = make_model("WithOptional", full_name=(str, ...), nick=(str, None))
        schema = _model_to_json_schema(M)
        assert "full_name" in schema["required"]
        assert "nick" not in schema["required"]

    def test_model_with_enum(self, pydantic):
        from enum import Enum
        from pydantic import BaseModel

        class Color(str, Enum):
            RED = "red"
            GREEN = "green"

        class Item(BaseModel):
            color: Color

        schema = _model_to_json_schema(Item)
        prop = schema["properties"]["color"]
        if "type" in prop:
            assert prop["type"] == "string"
        assert "$ref" in prop or "enum" in str(prop)


class TestModelSchemaToPrompt:
    def test_flat_model_prompt(self, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, 0))
        prompt = _model_schema_to_prompt(M)
        assert "Person" in prompt
        assert '"person_name": string' in prompt
        assert '(required)' in prompt
        assert '"age": integer' in prompt
        assert '(optional)' in prompt or '(Optional)' in prompt or 'optional' in prompt

    def test_nested_model_prompt(self, pydantic):
        from pydantic import BaseModel

        class Address(BaseModel):
            city: str
            zipcode: str = ""

        class Person(BaseModel):
            name: str
            address: Address

        prompt = _model_schema_to_prompt(Person)
        assert "name" in prompt
        assert "address" in prompt
        assert "array of" not in prompt

    def test_array_field_prompt(self, pydantic):
        from typing import List
        from pydantic import BaseModel

        class Invoice(BaseModel):
            items: List[str]

        prompt = _model_schema_to_prompt(Invoice)
        assert "items" in prompt
        assert "array" in prompt


class TestStructuredExtractorInit:
    def test_init_with_ai_processor(self):
        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        assert ex._ai is ai

    def test_init_without_ai_no_key_raises_on_call(self):
        ex = StructuredExtractor(provider="openai")
        with pytest.raises(ExtractionError, match="no AI backend configured"):
            ex.ai


class TestStructuredExtractorExtract:
    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_flat_model(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, ...))
        mock_call.return_value = json.dumps({"person_name": "Alice", "age": 30})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        result = ex.extract("Alice is 30 years old.", M)

        assert result.person_name == "Alice"
        assert result.age == 30

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_from_document(self, mock_call, pydantic):
        M = make_model("Title", title=(str, ...))
        mock_call.return_value = json.dumps({"title": "Hello World"})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        doc = Document(text="Hello World document", source_type="text", source_path="test.txt")
        result = ex.extract(doc, M)

        assert result.title == "Hello World"

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_with_defaults(self, mock_call, pydantic):
        M = make_model("Config", cfg_name=(str, ...), version=(int, 1))
        mock_call.return_value = json.dumps({"cfg_name": "app"})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        result = ex.extract("Config: app", M)

        assert result.cfg_name == "app"
        assert result.version == 1

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_retry_on_bad_json(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, ...))
        mock_call.side_effect = [
            "not json",
            "also not json",
            json.dumps({"person_name": "Bob", "age": 25}),
        ]

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        result = ex.extract("Bob is 25.", M, max_retries=3)

        assert result.person_name == "Bob"
        assert result.age == 25
        assert mock_call.call_count == 3

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_retry_then_raise(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, ...))
        mock_call.return_value = "invalid json forever"

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        with pytest.raises(StructuredExtractionError, match="after 3 attempts"):
            ex.extract("Bob is 25.", M, max_retries=3)

        assert mock_call.call_count == 3

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_retry_then_return_none(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, ...))
        mock_call.return_value = "invalid json"

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        result = ex.extract("Bob is 25.", M, max_retries=2, raise_on_failure=False)

        assert result is None

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_with_list_field(self, mock_call, pydantic):
        from typing import List
        M = make_model("Tags", tags=(List[str], ...))
        mock_call.return_value = json.dumps({"tags": ["ai", "ml", "nlp"]})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        result = ex.extract("AI, ML, NLP topics.", M)

        assert result.tags == ["ai", "ml", "nlp"]

    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_includes_response_format(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...), age=(int, ...))
        mock_call.return_value = json.dumps({"person_name": "Alice", "age": 30})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        ex.extract("Alice is 30.", M)

        _, kwargs = mock_call.call_args
        assert kwargs.get("response_format") == {"type": "json_object"}


class TestExtractStructuredFunction:
    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_convenience_function(self, mock_call, pydantic):
        M = make_model("Result", value=(str, ...))
        mock_call.return_value = json.dumps({"value": "hello"})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        result = extract_structured("hello world", M, ai_processor=ai)

        assert result.value == "hello"


class TestStructuredExtractorExtractMany:
    @patch("runeextract.processors.ai.AIProcessor._call")
    def test_extract_many(self, mock_call, pydantic):
        M = make_model("Person", person_name=(str, ...))
        mock_call.return_value = json.dumps({"person_name": "Alice"})

        ai = AIProcessor(provider="openai", api_key="sk-test")
        ex = StructuredExtractor(ai_processor=ai)
        results = ex.extract_many(["text A", "text B"], M)

        assert len(results) == 2
        assert results[0].person_name == "Alice"
        assert results[1].person_name == "Alice"
        assert mock_call.call_count == 2


class TestModelSchemaToPromptEdgeCases:
    def test_model_with_description_in_schema(self, pydantic):
        from pydantic import BaseModel, Field

        class DescribedModel(BaseModel):
            name: str = Field(description="The full name")

        prompt = _model_schema_to_prompt(DescribedModel)
        assert "full name" in prompt

    def test_non_pydantic_raises(self, pydantic):
        class NotAModel:
            pass

        with pytest.raises(TypeError, match="is not a Pydantic"):
            _model_to_json_schema(NotAModel)

    def test_empty_model_prompt(self, pydantic):
        class EmptyModel(pydantic):
            pass
        prompt = _model_schema_to_prompt(EmptyModel)
        assert "EmptyModel" in prompt
