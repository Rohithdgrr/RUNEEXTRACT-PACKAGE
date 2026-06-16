"""
AI processor for summarization, keyword extraction, entities,
Q&A, flashcards, structured output, embeddings, and multi-provider support.

Providers: OpenAI (default), Anthropic, Google Gemini, Ollama.
"""

import json
import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from runeextract.exceptions import DependencyMissingError, ExtractionError
from runeextract.config import get_config
from runeextract.models.document import _get_token_encoding
from runeextract.utils.rate_limiter import RateLimiter

# Cost per 1K tokens (USD) for common models — approximate prices
MODEL_COST_MAP = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4-turbo": {"input": 0.0100, "output": 0.0300},
    "claude-3-5-sonnet-20241022": {"input": 0.0030, "output": 0.0150},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.0050},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.00030},
    "deepseek-chat": {"input": 0.00027, "output": 0.00110},
    "mistral-large-latest": {"input": 0.0020, "output": 0.0060},
    "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
}

logger = logging.getLogger(__name__)


class AIProcessor:
    """Client for LLM-powered document processing.

    Supports OpenAI (default), Anthropic, Google Gemini, and local models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "openai",
        use_local: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.provider = provider.lower()
        self.use_local = use_local
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0
        self.rate_limiter = rate_limiter

        if use_local:
            self.api_key = None
        elif self.provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        elif self.provider == "ollama":
            self.api_key = api_key or ""  # Ollama doesn't need a key
        elif self.provider == "azure":
            self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            self._azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        elif self.provider == "bedrock":
            self.api_key = api_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        elif self.provider == "groq":
            self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        elif self.provider == "together":
            self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        elif self.provider == "deepseek":
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        elif self.provider == "mistral":
            self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        else:
            raise ExtractionError(
                f"Unknown provider '{provider}'. Options: openai, anthropic, gemini, ollama, "
                f"azure, bedrock, groq, together, deepseek, mistral",
                error_code="E033"
            )

        no_key_providers = {"ollama", "bedrock"}
        if not use_local and not self.api_key and self.provider not in no_key_providers:
            key_env = self.provider.upper() + ("_API_KEY" if self.provider not in ("azure", "bedrock") else ("_API_KEY" if self.provider == "azure" else "_ACCESS_KEY_ID"))
            raise ExtractionError(
                f"{self.provider.title()}: API key not set. "
                f"Set the {key_env} env var or pass api_key.",
                error_code="E030"
            )

        cfg = get_config()
        self.model = model or cfg.extra.get("ai_model", "gpt-4o-mini")
        self.temperature = float(cfg.extra.get("ai_temperature", 0.3))
        self.max_tokens = int(cfg.extra.get("ai_max_tokens", 2048))
        self._client = None
        self._local_pipeline = None

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if self.use_local:
            return None
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise DependencyMissingError("(ai)", "openai")
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise DependencyMissingError("(ai)", "anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise DependencyMissingError("(ai)", "google-generativeai")
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        elif self.provider == "ollama":
            try:
                from openai import OpenAI as OllamaClient
            except ImportError:
                raise DependencyMissingError("(ai)", "openai")
            self._client = OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama")
        elif self.provider == "azure":
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise DependencyMissingError("(ai)", "openai")
            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=os.environ.get("AZURE_OPENAI_VERSION", "2024-02-01"),
                azure_endpoint=self._azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            )
        elif self.provider == "bedrock":
            try:
                import boto3
            except ImportError:
                raise DependencyMissingError("(ai)", "boto3")
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
        elif self.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise DependencyMissingError("(ai)", "groq")
            self._client = Groq(api_key=self.api_key)
        elif self.provider == "together":
            try:
                from openai import OpenAI as TogetherClient
            except ImportError:
                raise DependencyMissingError("(ai)", "openai")
            self._client = TogetherClient(
                api_key=self.api_key,
                base_url="https://api.together.xyz/v1",
            )
        elif self.provider == "deepseek":
            try:
                from openai import OpenAI as DeepSeekClient
            except ImportError:
                raise DependencyMissingError("(ai)", "openai")
            self._client = DeepSeekClient(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
            )
        elif self.provider == "mistral":
            try:
                from mistralai import Mistral
            except ImportError:
                raise DependencyMissingError("(ai)", "mistralai")
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def _call(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.rate_limiter:
            self.rate_limiter()
        if self.use_local:
            return self._call_local(system, user)
        if self.provider == "openai":
            return self._call_openai(system, user, response_format=response_format, max_tokens=max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(system, user, max_tokens=max_tokens)
        elif self.provider == "gemini":
            return self._call_gemini(system, user, max_tokens=max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(system, user, response_format=response_format, max_tokens=max_tokens)
        elif self.provider == "azure":
            return self._call_azure(system, user, response_format=response_format, max_tokens=max_tokens)
        elif self.provider == "bedrock":
            return self._call_bedrock(system, user, max_tokens=max_tokens)
        elif self.provider in ("groq", "together", "deepseek", "mistral"):
            return self._call_openai_compat(system, user, max_tokens=max_tokens)
        raise ExtractionError(f"Unknown provider: {self.provider}", error_code="E033")

    def _call_openai(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._track_cost(
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"OpenAI call failed: {exc}")
                raise ExtractionError(f"AI processing failed: {exc}", error_code="E031")

    def _call_anthropic(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                usage = getattr(resp, "usage", None)
                self._track_cost(
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                )
                return resp.content[0].text.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"Anthropic call failed: {exc}")
                raise ExtractionError(f"Anthropic processing failed: {exc}", error_code="E031")

    def _call_gemini(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        for attempt in range(3):
            try:
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": max_tokens or self.max_tokens,
                }
                resp = self.client.generate_content(
                    f"{system}\n\n{user}",
                    generation_config=generation_config,
                )
                self._track_cost(input_tokens=0, output_tokens=0)
                return resp.text.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"Gemini call failed: {exc}")
                raise ExtractionError(f"Gemini processing failed: {exc}", error_code="E031")

    def _call_ollama(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._track_cost(
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"Ollama call failed: {exc}")
                raise ExtractionError(f"Ollama processing failed: {exc}", error_code="E031")

    def _call_azure(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._track_cost(
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"Azure call failed: {exc}")
                raise ExtractionError(f"Azure AI processing failed: {exc}", error_code="E031")

    def _call_bedrock(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.temperature,
        })
        for attempt in range(3):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
                data = json.loads(resp["body"].read())
                self._track_cost(input_tokens=0, output_tokens=0)
                return data["content"][0]["text"].strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"Bedrock call failed: {exc}")
                raise ExtractionError(f"Bedrock AI processing failed: {exc}", error_code="E031")

    def _call_openai_compat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._track_cost(
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                logger.error(f"{self.provider.title()} call failed: {exc}")
                raise ExtractionError(f"{self.provider.title()} AI processing failed: {exc}", error_code="E031")

    def _call_local(self, system: str, user: str) -> str:
        try:
            from transformers import pipeline
        except ImportError:
            raise DependencyMissingError("(local AI)", "transformers")

        if self._local_pipeline is None:
            try:
                self._local_pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    max_new_tokens=self.max_tokens,
                )
            except Exception as exc:
                raise ExtractionError(
                    f"Failed to load local model '{self.model}': {exc}",
                    error_code="E032",
                )

        prompt = f"{system}\n\n{user}\n\nAnswer:"
        result = self._local_pipeline(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
        )
        self._track_cost(input_tokens=0, output_tokens=0)
        return result[0]["generated_text"].replace(prompt, "").strip()

    def _call_chunked(
        self,
        system: str,
        text: str,
        chunk_size: int = 50_000,
        response_format: Optional[Dict[str, str]] = None,
        use_token_count: bool = False,
    ) -> str:
        """Process very long text in chunks, then merge results.

        Args:
            system: System prompt
            text: Text to process
            chunk_size: Target chunk size (chars by default, tokens if use_token_count=True)
            response_format: Optional response format
            use_token_count: If True, split by token count instead of character count
        """
        if use_token_count:
            try:
                token_count = self._estimate_token_count(text)
                if token_count <= chunk_size:
                    return self._call(system, text, response_format=response_format)
                estimated_chars = int(len(text) * (chunk_size / max(token_count, 1)))
                chunks_text = []
                for i in range(0, len(text), max(estimated_chars, 1)):
                    part = text[i:i + max(estimated_chars, 1)]
                    if part:
                        chunks_text.append(part)
                if len(chunks_text) <= 1:
                    return self._call(system, text, response_format=response_format)
                chunks = chunks_text
            except Exception:
                use_token_count = False

        if not use_token_count:
            if len(text) <= chunk_size:
                return self._call(system, text, response_format=response_format)
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        partials = []
        for i, chunk in enumerate(chunks):
            result = self._call(
                system,
                f"Part {i + 1} of {len(chunks)}:\n\n{chunk}",
                response_format=response_format,
            )
            partials.append(result)

        merge_prompt = (
            f"Combine the following {len(partials)} partial summaries "
            f"into one coherent summary. Remove redundancy."
        )
        merged = "\n\n".join(
            f"--- Part {i + 1} ---\n{p}"
            for i, p in enumerate(partials)
        )
        return self._call(merge_prompt, merged, response_format=response_format)

    def summarize(self, text: str, max_words: int = 200) -> str:
        """Summarize the document text."""
        return self._call_chunked(
            "You are a document summarization assistant. "
            "Summarize the following text concisely.",
            f"Summarize this in {max_words} words or fewer:\n\n{text}"
        )

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text."""
        if len(text) > 100_000:
            text = text[:100_000]
        result = self._call(
            "You extract keywords and key phrases from documents. "
            "Return them as a comma-separated list.",
            f"Extract the top {top_n} keywords from:\n\n{text}"
        )
        return [kw.strip() for kw in result.split(",") if kw.strip()]

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities (people, orgs, locations, dates)."""
        if len(text) > 100_000:
            text = text[:100_000]
        result = self._call(
            "You extract named entities from documents. "
            "Return a JSON array of objects with 'type' and 'name' keys.",
            f"Extract all named entities (people, organizations, locations, dates, etc.):\n\n{text}",
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed["entities"]
        except json.JSONDecodeError:
            pass
        return [{"type": "text", "name": line.strip()}
                for line in result.split("\n") if line.strip()]

    @staticmethod
    def extract_entities_tools() -> List[Dict[str, Any]]:
        """Return tool definitions for structured entity extraction via function calling.

        Use with `AIProcessor.call_with_tools(... tools=..., tool_choice=...)`
        to get structured JSON output.
        """
        return [{
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract named entities from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["person", "organization", "location", "date", "event", "other"]},
                                    "name": {"type": "string"},
                                },
                                "required": ["type", "name"],
                            },
                        },
                    },
                    "required": ["entities"],
                },
            },
        }]

    def generate_questions(self, text: str, n: int = 5) -> List[str]:
        """Generate questions based on document content."""
        if len(text) > 100_000:
            text = text[:100_000]
        result = self._call(
            "You generate questions based on document content.",
            f"Generate {n} questions that test understanding of:\n\n{text}"
        )
        return [q.strip() for q in result.split("\n") if q.strip() and q.strip().endswith("?")]

    def generate_flashcards(self, text: str, n: int = 10) -> List[Dict[str, str]]:
        """Generate flashcards (front/back pairs) from document content."""
        if len(text) > 100_000:
            text = text[:100_000]
        result = self._call(
            "You create flashcards from document content. "
            "Return a JSON array of objects with 'front' and 'back' keys.",
            f"Create {n} flashcards from:\n\n{text}",
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "flashcards" in parsed:
                return parsed["flashcards"]
        except json.JSONDecodeError:
            pass
        return [{"front": line, "back": ""}
                for line in result.split("\n") if line.strip() and "?" in line]

    @staticmethod
    def generate_flashcards_tools() -> List[Dict[str, Any]]:
        """Return tool definitions for structured flashcard generation via function calling."""
        return [{
            "type": "function",
            "function": {
                "name": "generate_flashcards",
                "description": "Generate flashcards from document content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flashcards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "front": {"type": "string", "description": "The question or prompt"},
                                    "back": {"type": "string", "description": "The answer or explanation"},
                                },
                                "required": ["front", "back"],
                            },
                        },
                    },
                    "required": ["flashcards"],
                },
            },
        }]

    def answer_question(self, question: str, context: str, max_words: int = 500) -> str:
        """Answer a question based on the provided context document."""
        return self._call(
            "You are a helpful assistant. Answer the user's question based solely on the provided context. "
            "If the context doesn't contain the answer, say so.",
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            f"Answer in {max_words} words or fewer.",
            max_tokens=max_words * 4,
        )

    def redact_pii(self, text: str) -> str:
        """Redact common PII patterns (emails, phones, SSNs, credit cards) from text.

        Uses regex-based detection. Returns text with PII replaced by placeholders.
        """
        patterns = {
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b': '[EMAIL]',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b': '[PHONE]',
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b': '[CREDIT_CARD]',
            r'\b(?:\d[-\s]*){16}\b': '[CREDIT_CARD]',
        }
        result = text
        for pattern, replacement in patterns.items():
            result = re.sub(pattern, replacement, result)
        return result

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text string using tiktoken if available."""
        try:
            enc = _get_token_encoding()
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Rerank documents by relevance to a query using cross-encoder.

        Uses sentence-transformers cross-encoder if available. Falls back to
        embedding-based cosine similarity.

        Args:
            query: Query string
            documents: List of document strings to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of (text, score) tuples sorted by relevance descending
        """
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)
            scored = list(zip(documents, scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
        except ImportError:
            return self._rerank_by_embedding(query, documents)
        except Exception:
            return self._rerank_by_embedding(query, documents)

        if top_k is not None:
            scored = scored[:top_k]
        return scored

    def _rerank_by_embedding(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Fallback reranking using embedding cosine similarity."""
        try:
            import numpy as np
            all_texts = [query] + documents
            embeddings = self.embed(all_texts)
            query_vec = np.array(embeddings[0], dtype=np.float32)
            doc_vecs = np.array(embeddings[1:], dtype=np.float32)
            norms = np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
            scores = np.dot(doc_vecs, query_vec) / np.maximum(norms, 1e-10)
            indices = np.argsort(scores)[::-1]
            return [(documents[i], float(scores[i])) for i in indices]
        except Exception as exc:
            logger.warning(f"Embedding rerank failed: {exc}")
            return [(doc, 0.0) for doc in documents]

    # --- Cost tracking ---

    @property
    def total_cost(self) -> float:
        """Total accumulated cost in USD across all calls."""
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls."""
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        """Total number of AI calls made."""
        return self._call_count

    def reset_cost(self):
        """Reset all cost and token counters to zero."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    def _track_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None):
        """Track token usage and estimate cost."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._call_count += 1
        model_name = model or self.model
        rates = MODEL_COST_MAP.get(model_name)
        if rates:
            self._total_cost += (input_tokens / 1000) * rates["input"]
            self._total_cost += (output_tokens / 1000) * rates["output"]

    # --- Function calling / tool use ---

    def call_with_tools(
        self,
        system: str,
        user: str,
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call the LLM with tool/function definitions (OpenAI format).

        Args:
            system: System prompt
            user: User message
            tools: List of tool definitions in OpenAI function-calling format
            tool_choice: "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
            max_tokens: Max tokens for the response

        Returns:
            Dict with either {"content": str} or {"tool_calls": [...]}
        """
        if self.provider not in ("openai", "groq", "together", "deepseek", "mistral", "azure"):
            raise ExtractionError(
                f"Function calling not supported for provider '{self.provider}'. "
                f"Use an OpenAI-compatible provider.", error_code="E033"
            )
        if self.use_local:
            raise ExtractionError("Function calling not supported for local models.", error_code="E033")

        if self.rate_limiter:
            self.rate_limiter()

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                self._track_cost(
                    input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                )
                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    calls = []
                    for tc in choice.message.tool_calls:
                        calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        })
                    return {"tool_calls": calls}
                return {"content": choice.message.content.strip()}
            except Exception as exc:
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                raise ExtractionError(f"Function call failed: {exc}", error_code="E031")

    # --- Streaming AI responses ---

    def _call_stream(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ):
        """Call the LLM and yield response chunks as they arrive.

        Args:
            system: System prompt
            user: User message
            max_tokens: Max tokens for the response

        Yields:
            String chunks of the response as they arrive from the API.
        """
        if self.use_local:
            yield self._call_local(system, user)
            return
        if self.provider not in ("openai", "groq", "together", "deepseek", "mistral", "azure"):
            yield self._call(system, user, max_tokens=max_tokens)
            return

        if self.rate_limiter:
            self.rate_limiter()

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        collected = []
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    collected.append(delta.content)
                    yield delta.content
            full = "".join(collected)
            if hasattr(chunk, "usage") and chunk.usage:
                self._track_cost(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
        except Exception as exc:
            if collected:
                yield "".join(collected)
            logger.error(f"Streaming call failed: {exc}")
            raise ExtractionError(f"AI streaming failed: {exc}", error_code="E031")

    async def call_stream_async(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
    ):
        """Async generator that yields response chunks via streaming.

        Args:
            system: System prompt
            user: User message
            max_tokens: Max tokens for the response

        Yields:
            String chunks as they arrive.
        """
        for chunk in self._call_stream(system, user, max_tokens=max_tokens):
            yield chunk

    # --- Query expansion (HyDE + multi-query) ---

    def expand_query(self, query: str, n_queries: int = 3) -> List[str]:
        """Generate expanded queries for better retrieval coverage.

        Uses HyDE (Hypothetical Document Embeddings) + multi-query expansion.
        Returns the original query plus generated variants.

        Args:
            query: Original query string
            n_queries: Number of additional queries to generate

        Returns:
            List of query strings (original + expansions)
        """
        result = self._call(
            "You are a query expansion assistant. Generate diverse rephrasings "
            "of the given query to improve document retrieval recall. "
            "Return one query per line, no numbering or bullets.",
            f"Generate {n_queries} alternative phrasings of this question that "
            f"would help find relevant documents:\n\n{query}",
        )
        expansions = [line.strip() for line in result.split("\n") if line.strip() and len(line.strip()) > 5]
        all_queries = [query] + expansions[:n_queries]
        return all_queries

    def hyde(self, query: str) -> str:
        """Generate a hypothetical document that answers the query.

        The hypothetical document can be used as an embedding target
        to improve retrieval (HyDE technique).

        Args:
            query: The user's question

        Returns:
            A hypothetical document text that would be the ideal answer
        """
        return self._call(
            "You are a document-generation assistant. Write a hypothetical "
            "document that would perfectly answer the given question. "
            "Be factual, detailed, and write at least 3 paragraphs.",
            f"Write a hypothetical document that answers:\n\n{query}",
        )

    def batch_process(
        self,
        prompts: List[Dict[str, str]],
        max_concurrency: int = 4,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Process multiple prompts concurrently with concurrency control.

        Each prompt dict must have "system" and "user" keys.
        Uses a thread pool to send requests in parallel while respecting
        the rate limiter (if configured).

        Args:
            prompts: List of dicts, each with "system" and "user" keys
            max_concurrency: Max concurrent API calls (default 4)
            max_tokens: Max tokens per response (uses instance default if None)

        Returns:
            List of response strings, same order as prompts
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Optional[str]] = [None] * len(prompts)

        def _process(idx: int, p: Dict[str, str]) -> int:
            result = self._call(
                p.get("system", ""),
                p.get("user", ""),
                max_tokens=max_tokens or self.max_tokens,
            )
            results[idx] = result
            return idx

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {pool.submit(_process, i, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                pass

        return [r for r in results if r is not None]

    def embed(self, texts: Union[str, List[str]], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for text(s) using the configured provider.
        
        Args:
            texts: Single string or list of strings to embed
            model: Embedding model name (provider-specific)
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.use_local:
            return self._embed_local(texts, model)
        if self.provider == "openai":
            return self._embed_openai(texts, model)
        elif self.provider == "ollama":
            return self._embed_ollama(texts, model)
        else:
            raise ExtractionError(
                f"Embedding not supported for provider '{self.provider}'. "
                f"Use provider='openai' or provider='ollama' for embeddings.",
                error_code="E033"
            )

    def _embed_openai(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        model = model or "text-embedding-3-small"
        try:
            resp = self.client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:
            logger.error(f"OpenAI embedding failed: {exc}")
            raise ExtractionError(f"Embedding failed: {exc}", error_code="E031")

    def _embed_ollama(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        model = model or "nomic-embed-text"
        results = []
        for text in texts:
            try:
                resp = self.client.embeddings.create(model=model, input=text)
                results.append(resp.data[0].embedding)
            except Exception as exc:
                logger.error(f"Ollama embedding failed: {exc}")
                raise ExtractionError(f"Embedding failed: {exc}", error_code="E031")
        return results

    def _embed_local(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
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
