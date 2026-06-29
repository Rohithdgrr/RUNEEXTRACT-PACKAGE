"""
AI processor for summarization, keyword extraction, entities,
Q&A, flashcards, structured output, embeddings, and multi-provider support.

Providers are registered via ``runeextract.processors.providers``.
"""

import json
import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple

from runeextract.exceptions import DependencyMissingError, ExtractionError, CircuitBreakerOpenError
from runeextract.config import get_config
from runeextract.utils.rate_limiter import RateLimiter
from runeextract.processors import providers as _providers

logger = logging.getLogger(__name__)

_DEFAULT_AI_TIMEOUT = 60
_CB_FAILURE_THRESHOLD = 5
_CB_RECOVERY_TIMEOUT = 30

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


class AIProcessor:
    """Client for LLM-powered document processing.

    Supports OpenAI (default), Anthropic, Google Gemini, and local models.
    Provider-specific logic is delegated to ``runeextract.processors.providers``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "openai",
        use_local: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        fallback_providers: Optional[List[Dict[str, Any]]] = None,
    ):
        self.provider = provider.lower()
        self.use_local = use_local
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0
        self.rate_limiter = rate_limiter
        self.fallback_providers = fallback_providers or []
        self._shared_thread_pool = None

        if use_local:
            self.api_key = None
        elif self.provider == "openai" or self.provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "gemini":
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        elif self.provider == "ollama":
            self.api_key = api_key or ""
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
        elif self.provider == "local":
            self.api_key = None
        else:
            raise ExtractionError(
                f"Unknown provider '{provider}'. Options: openai, openrouter, anthropic, gemini, ollama, "
                f"azure, bedrock, groq, together, deepseek, mistral, local",
                error_code="E033"
            )

        no_key_providers = {"ollama", "bedrock", "local"}
        if not use_local and not self.api_key and self.provider not in no_key_providers:
            _env_map = {"azure": "AZURE_OPENAI_API_KEY", "openai": "OPENAI_API_KEY",
                        "openrouter": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
                        "groq": "GROQ_API_KEY", "together": "TOGETHER_API_KEY",
                        "deepseek": "DEEPSEEK_API_KEY", "mistral": "MISTRAL_API_KEY"}
            key_env = _env_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
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
        self._cb_failures = 0
        self._cb_last_failure = 0.0
        self._cb_state = "closed"
        self._request_timeout = float(cfg.extra.get("ai_request_timeout", _DEFAULT_AI_TIMEOUT))

    # --- Circuit breaker ---

    def _check_circuit_breaker(self):
        if self._cb_state == "open":
            if time.time() - self._cb_last_failure >= _CB_RECOVERY_TIMEOUT:
                self._cb_state = "half-open"
                self._cb_failures = 0
            else:
                raise CircuitBreakerOpenError(self.provider)

    def _record_failure(self):
        self._cb_failures += 1
        self._cb_last_failure = time.time()
        if self._cb_failures >= _CB_FAILURE_THRESHOLD:
            self._cb_state = "open"
            self._cb_failures = 0

    def _record_success(self):
        if self._cb_state == "half-open":
            self._cb_state = "closed"
            self._cb_failures = 0

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        msg = str(exc)
        msg = re.sub(r'(sk-[a-zA-Z0-9]{20,})', '[API_KEY_REDACTED]', msg)
        msg = re.sub(r'(pk-[a-zA-Z0-9]{20,})', '[API_KEY_REDACTED]', msg)
        msg = re.sub(r'(AIza[0-9A-Za-z_-]{35})', '[GEMINI_KEY_REDACTED]', msg)
        msg = re.sub(r'(AKIA[0-9A-Z]{16})', '[AWS_KEY_REDACTED]', msg)
        msg = re.sub(r'(gh[pousr]_[A-Za-z0-9_]{36,})', '[GH_TOKEN_REDACTED]', msg)
        msg = re.sub(r'(xox[bpras]-[0-9A-Za-z-]{10,})', '[SLACK_TOKEN_REDACTED]', msg)
        msg = re.sub(r'(Bearer\s+)[a-zA-Z0-9\-_.]+', r'\1[TOKEN_REDACTED]', msg)
        msg = re.sub(r'(Authorization:\s*)[^\s]+', r'\1[REDACTED]', msg)
        msg = re.sub(r'(api[_-]?key[=:]\s*)[^\s&]+', r'\1[REDACTED]', msg, flags=re.IGNORECASE)
        return msg

    @staticmethod
    def _sanitize_user_input(text: str, max_chars: int = 100_000) -> str:
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return cleaned[:max_chars]

    # --- Retry + cost tracking ---

    def _call_with_retry(self, call_fn, extract_fn, provider_label="AI"):
        self._check_circuit_breaker()
        for attempt in range(3):
            try:
                resp = call_fn()
                self._track_cost(*extract_fn(resp))
                self._record_success()
                return resp
            except CircuitBreakerOpenError:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self._record_failure()
                if attempt < 2:
                    delay = 1.0 * (2.0 ** attempt) * random.uniform(0.5, 1.5)
                    time.sleep(delay)
                    continue
                safe_msg = self._sanitize_error(exc)
                logger.error(f"{provider_label} call failed after retries: {safe_msg[:200]}")
                raise ExtractionError(
                    f"{provider_label} processing failed after retries: {safe_msg[:500]}",
                    error_code="E031"
                )

    # --- Client creation (delegates to provider) ---

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if self.use_local:
            return None
        self._client = _providers.create_client(self.provider, self)
        return self._client

    # --- Core call methods (delegate to provider registry) ---

    def _get_provider_chain(self):
        """Build list of (provider_name, ai_processor) tuples for fallback."""
        chain = [(self.provider, self)]
        for fb in self.fallback_providers:
            fb_provider = fb.get("provider", "")
            if fb_provider and fb_provider != chain[-1][0]:
                fb_ai = AIProcessor(
                    provider=fb_provider,
                    api_key=fb.get("api_key"),
                    model=fb.get("model"),
                    use_local=self.use_local,
                )
                chain.append((fb_provider, fb_ai))
        return chain

    def _call(
        self,
        system: str,
        user: str,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.rate_limiter:
            self.rate_limiter()
        user = self._sanitize_user_input(user)

        chain = self._get_provider_chain()
        last_error = None
        for provider_name, ai_inst in chain:
            try:
                key = "local" if ai_inst.use_local else provider_name
                return _providers.call(key, ai_inst, system, user,
                                       response_format=response_format, max_tokens=max_tokens)
            except Exception as exc:
                last_error = exc
                if provider_name != chain[-1][0]:
                    logger.warning("Provider '%s' failed, trying fallback: %s", provider_name, exc)
                continue
        safe_msg = self._sanitize_error(last_error) if last_error else "All providers failed"
        raise ExtractionError(f"AI call failed after fallbacks: {safe_msg[:500]}", error_code="E031")

    def _call_stream(self, system: str, user: str, max_tokens: Optional[int] = None):
        if self.rate_limiter:
            self.rate_limiter()
        self._check_circuit_breaker()
        user = self._sanitize_user_input(user)

        chain = self._get_provider_chain()
        last_error = None
        for provider_name, ai_inst in chain:
            try:
                key = "local" if ai_inst.use_local else provider_name
                yield from _providers.call_stream(key, ai_inst, system, user, max_tokens=max_tokens)
                return
            except Exception as exc:
                last_error = exc
                if provider_name != chain[-1][0]:
                    logger.warning("Stream provider '%s' failed, trying fallback: %s", provider_name, exc)
                continue
        safe_msg = self._sanitize_error(last_error) if last_error else "All providers failed"
        raise ExtractionError(f"AI stream failed after fallbacks: {safe_msg[:500]}", error_code="E031")

    def _call_chunked(
        self,
        system: str,
        text: str,
        chunk_size: int = 50_000,
        response_format: Optional[Dict[str, str]] = None,
        use_token_count: bool = False,
    ) -> str:
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
            except Exception as exc:
                logger.warning("Token count estimation failed: %s", exc)
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

    async def call_stream_async(self, system: str, user: str, max_tokens: Optional[int] = None):
        import asyncio
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for chunk in self._call_stream(system, user, max_tokens=max_tokens):
                    fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                    fut.result()
            except Exception as exc:
                fut = asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                fut.result()
            finally:
                fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                fut.result()

        runner = loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await runner

    # --- Public AI methods ---

    def summarize(self, text: str, max_words: int = 200) -> str:
        return self._call_chunked(
            "You are a document summarization assistant. "
            "Summarize the following text concisely.",
            f"Summarize this in {max_words} words or fewer:\n\n{text}",
        )

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        result = self._call(
            "You are a keyword extraction assistant. Extract the most important keywords.",
            f"Extract up to {top_n} keywords from this text. "
            f"Return a comma-separated list.\n\n{text}",
        )
        return [k.strip() for k in result.replace("\n", ",").split(",") if k.strip()]

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        return json.loads(self._call(
            "You are an entity extraction assistant. Extract named entities.",
            f"Extract named entities (people, organizations, locations, dates) "
            f"as a JSON array of objects with 'type' and 'name' keys.\n\n{text}",
            response_format={"type": "json_object"},
        ))

    @staticmethod
    def extract_entities_tools() -> List[Dict[str, Any]]:
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
                                    "type": {"type": "string", "enum": ["person", "organization", "location", "date", "other"]},
                                    "name": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["entities"],
                },
            },
        }]

    def generate_questions(self, text: str, n: int = 5) -> List[str]:
        result = self._call(
            "You are a question generation assistant.",
            f"Generate {n} questions based on this text. "
            f"Return as a numbered list.\n\n{text}",
        )
        return [re.sub(r'^\d+[.)]\s*', '', line).strip()
                for line in result.split("\n") if line.strip() and re.match(r'\d+[.)]', line.strip())]

    def generate_flashcards(self, text: str, n: int = 10) -> List[Dict[str, str]]:
        return json.loads(self._call(
            "You are a flashcard generation assistant.",
            f"Generate {n} flashcards (front/back pairs) from this text. "
            f"Return as JSON with a 'flashcards' array.\n\n{text}",
            response_format={"type": "json_object"},
        ))["flashcards"]

    @staticmethod
    def generate_flashcards_tools() -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "generate_flashcards",
                "description": "Generate flashcards from text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flashcards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "front": {"type": "string"},
                                    "back": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["flashcards"],
                },
            },
        }]

    def answer_question(self, question: str, context: str, max_words: int = 500) -> str:
        return self._call(
            "You are a helpful assistant. Answer the user's question based solely on the provided context. "
            "If the context doesn't contain the answer, say so.",
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            f"Answer in {max_words} words or fewer.",
            max_tokens=max_words * 4,
        )

    def redact_pii(self, text: str, use_dp: bool = False, epsilon: float = 1.0) -> str:
        if not text:
            return text
        if len(text) > 500_000:
            logger.warning("PII redaction input too large (%d bytes) — truncating to 500KB", len(text))
            text = text[:500_000]
        patterns = [
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]'),
            (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]'),
            (r'\b(?:\d[-\s]*){16}\b', '[CREDIT_CARD]'),
            (r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b', '[IP_ADDRESS]'),
            (r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', '[IPV6_ADDRESS]'),
            (r'\b(?:[0-9a-fA-F]{1,4}:){1,6}::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}?\b', '[IPV6_ADDRESS]'),
            (r'\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}?\b', '[IPV6_ADDRESS]'),
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', '[DATE]'),
            (r'\b[A-Z]{2}\d{6,9}\b', '[PASSPORT]'),
            (r'\b(?:sk-[a-zA-Z0-9]{20,}|pk-[a-zA-Z0-9]{20,})\b', '[API_KEY]'),
            (r'https?://(?:[^\s/$.?#]+\.)*(?:bit\.ly|tinyurl\.com|short\.link|t\.co)/\S+', '[SHORT_URL]'),
        ]
        result = text
        for pat, replacement in patterns:
            try:
                result = re.sub(pat, replacement, result)
            except re.error as exc:
                logger.warning("PII redaction regex error: %s — skipping pattern", exc)
        if use_dp:
            from runeextract.utils.privacy import DifferentialPrivacyEngine
            dp = DifferentialPrivacyEngine(epsilon=epsilon)
            result = dp.apply(result)
        return result

    def scan_secrets(self, text: str, auto_redact: bool = False) -> list:
        from runeextract.utils.secrets import scan_secrets as _scan, redact_secrets
        findings = _scan(text)
        if auto_redact:
            return redact_secrets(text, findings), findings
        return findings

    def _estimate_token_count(self, text: str) -> int:
        from runeextract.models.types import _get_token_encoding
        enc = _get_token_encoding("cl100k_base")
        if enc:
            return len(enc.encode(text))
        return max(1, len(text) // 4)

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        if not documents:
            return []
        if len(documents) <= 3:
            return [(doc, 1.0) for doc in documents]
        return self._rerank_by_embedding(query, documents, top_k=top_k)

    def _rerank_by_embedding(self, query: str, documents: List[str],
                              top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        import numpy as np
        query_vec = np.array(self.embed(query)[0], dtype=np.float32)
        doc_vecs = np.array(self.embed(documents), dtype=np.float32)
        norms = np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        scores = np.dot(doc_vecs, query_vec) / np.maximum(norms, 1e-10)
        pairs = [(documents[i], float(scores[i])) for i in range(len(documents))]
        pairs.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            pairs = pairs[:top_k]
        return pairs

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_cost(self):
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _track_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None):
        model = model or self.model
        cost_info = MODEL_COST_MAP.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1000) * cost_info["input"] + (output_tokens / 1000) * cost_info["output"]
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost
        self._call_count += 1

    def call_with_tools(self, system: str, user: str, tools: List[Dict[str, Any]],
                        tool_choice: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        if self.rate_limiter:
            self.rate_limiter()
        user = self._sanitize_user_input(user)
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "timeout": self._request_timeout,
        }
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        resp = self._call_with_retry(
            lambda: self.client.chat.completions.create(**kwargs),
            lambda r: (r.usage.prompt_tokens if r.usage else 0,
                       r.usage.completion_tokens if r.usage else 0),
            provider_label=self.provider.title(),
        )
        choice = resp.choices[0]
        if choice.message.tool_calls:
            return json.dumps([{
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            } for tc in choice.message.tool_calls])
        return choice.message.content.strip()

    # --- Query expansion ---

    def expand_query(self, query: str, n_queries: int = 3) -> List[str]:
        queries = [query]
        try:
            prompt = (f"Generate {n_queries} alternative versions of the following search query "
                      f"to improve retrieval coverage. Return one per line, numbered.\n\nQuery: {query}")
            result = self._call(
                "You are a query expansion assistant.",
                prompt,
                max_tokens=300,
            )
            for line in result.split("\n"):
                cleaned = re.sub(r'^\d+[.)]\s*', '', line).strip()
                if cleaned and cleaned.lower() != query.lower():
                    queries.append(cleaned)
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
        return queries[:n_queries + 1]

    def hyde(self, query: str) -> str:
        try:
            return self._call(
                "You are a hypothetical document generator. "
                "Generate a short hypothetical document that would be the perfect answer to the query.",
                f"Generate a hypothetical document for: {query}",
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return query

    def _get_thread_pool(self, max_workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor
        if self._shared_thread_pool is None:
            self._shared_thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        return self._shared_thread_pool

    def batch_process(self, prompts: List[Dict[str, str]], max_concurrency: int = 4) -> List[str]:
        results = [None] * len(prompts)

        def _process(idx: int, p: Dict[str, str]) -> int:
            system = p.get("system", "")
            user = p.get("user", "")
            results[idx] = self._call(system, user)
            return idx

        from concurrent.futures import as_completed
        pool = self._get_thread_pool(max_workers=max(1, min(max_concurrency, len(prompts))))
        futures = [pool.submit(_process, i, p) for i, p in enumerate(prompts)]
        for f in as_completed(futures):
            f.result()
        return [r for r in results if r is not None]

    def close(self):
        self._client = None
        self._local_pipeline = None
        if self._shared_thread_pool:
            self._shared_thread_pool.shutdown(wait=False)
            self._shared_thread_pool = None

    # --- Vision / Multi-modal ---

    @property
    def has_vision(self) -> bool:
        """Auto-detect if the current model supports image inputs."""
        return _providers.supports_vision(self.provider, self.model)

    def describe_image(self, image_data: bytes, image_format: str = "png",
                       prompt: str = "Describe this image in detail.") -> str:
        """Describe an image using the LLM's vision capability.

        Args:
            image_data: Raw image bytes.
            image_format: Image format (png, jpeg, etc.).
            prompt: Optional prompt describing what to extract.

        Returns:
            Text description of the image.
        """
        provider = "local" if self.use_local else self.provider
        return _providers.vision_call(
            provider, self,
            system="You are a vision analyst. Describe the image in detail.",
            user=prompt,
            images=[(image_data, image_format)],
        )

    def analyze_images(self, images: list,
                       prompt: str = "Describe these images in detail.") -> List[str]:
        """Analyze multiple images in a single call (if supported).

        Args:
            images: List of (image_data: bytes, image_format: str) tuples.
            prompt: Prompt to use for all images.

        Returns:
            The model's combined response text.
        """
        if not images:
            return []
        provider = "local" if self.use_local else self.provider
        result = _providers.vision_call(
            provider, self,
            system="You are a vision analyst. Respond to the prompt about the provided images.",
            user=prompt,
            images=images,
        )
        return [result]

    # --- Embedding ---

    def embed(self, texts: Union[str, List[str]], model: Optional[str] = None) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        chain = self._get_provider_chain()
        last_error = None
        for provider_name, ai_inst in chain:
            try:
                key = "local" if ai_inst.use_local else provider_name
                return _providers.embed(key, ai_inst, texts, model=model)
            except Exception as exc:
                last_error = exc
                if provider_name != chain[-1][0]:
                    logger.warning("Embed provider '%s' failed, trying fallback: %s", provider_name, exc)
                continue
        safe_msg = self._sanitize_error(last_error) if last_error else "All providers failed"
        raise ExtractionError(f"Embedding failed after fallbacks: {safe_msg[:500]}", error_code="E033")
