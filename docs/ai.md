# AI Providers & Features

## Configuration

AI features require an API key set via environment variable or constructor argument:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from runeextract.processors.ai import AIProcessor
ai = AIProcessor(provider="openai")
```

Or pass directly:

```python
ai = AIProcessor(api_key="sk-...", provider="openai")
```

## Available Providers

### OpenAI
- **Env**: `OPENAI_API_KEY`
- **Models**: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
- **Supports**: chat, embeddings, function calling, streaming, response_format

### Anthropic (Claude)
- **Env**: `ANTHROPIC_API_KEY`
- **Models**: claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.
- **Supports**: chat

### Google Gemini
- **Env**: `GEMINI_API_KEY`
- **Models**: gemini-1.5-pro, gemini-1.5-flash, etc.
- **Supports**: chat

### Ollama (local)
- **No API key needed** (uses localhost:11434)
- **Models**: llama3.2, mistral, etc. (must be pulled locally)
- **Supports**: chat, embeddings
- Requires `openai` Python package for the OpenAI-compatible API

### Azure OpenAI
- **Env**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_VERSION`
- **Supports**: chat, function calling, streaming
- Uses `AzureOpenAI` client from the `openai` package

### AWS Bedrock
- **Env**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- **Models**: anthropic.claude-3-5-sonnet-20241022-v2:0, etc.
- **Supports**: chat (via Anthropic Claude models on Bedrock)
- Requires `boto3` package

### Groq
- **Env**: `GROQ_API_KEY`
- **Models**: llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768, etc.
- **Supports**: chat, function calling, streaming
- Requires `groq` package

### Together AI
- **Env**: `TOGETHER_API_KEY`
- **Models**: mistralai/Mixtral-8x22B-Instruct-v0.1, etc.
- **Supports**: chat, function calling, streaming
- OpenAI-compatible API

### DeepSeek
- **Env**: `DEEPSEEK_API_KEY`
- **Models**: deepseek-chat
- **Supports**: chat, function calling, streaming
- OpenAI-compatible API

### Mistral AI
- **Env**: `MISTRAL_API_KEY`
- **Models**: mistral-large-latest, mistral-medium-latest, etc.
- **Supports**: chat, function calling, streaming
- Requires `mistralai` package

## Feature Matrix

| Feature | openai | anthropic | gemini | ollama | azure | bedrock | groq | together | deepseek | mistral |
|---------|--------|-----------|--------|--------|-------|---------|------|----------|----------|---------|
| Chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Function calling | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Embeddings | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| response_format | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Cost Tracking

Pricing is estimated based on published API prices:

```python
ai = AIProcessor(provider="openai", model="gpt-4o-mini")
ai._call("You are helpful.", "Hello!")
print(f"Cost: ${ai.total_cost:.6f}")
```

Supported pricing models: gpt-4o-mini, gpt-4o, gpt-4-turbo, claude-3-5-sonnet, claude-3-haiku, gemini-1.5-pro, gemini-1.5-flash, deepseek-chat, mistral-large-latest, llama3-70b-8192.

Reset counters:
```python
ai.reset_cost()
```

## Rate Limiter

```python
from runeextract.utils.rate_limiter import RateLimiter

# 30 requests per minute, 100K tokens per minute burst
limiter = RateLimiter(requests_per_minute=30, tokens_per_minute=100000)
ai = AIProcessor(rate_limiter=limiter)
```

## Document-Level AI

The Document class provides convenience methods:

```python
doc = extract("report.pdf")

# These all create an internal AIProcessor
print(doc.summary())
print(doc.keywords(10))
print(doc.entities())
print(doc.questions(5))
print(doc.flashcards(10))

# RAG question answering
answer = doc.ask("What are the key findings?")

# PII redaction
safe_text = doc.redact_pii()
```
