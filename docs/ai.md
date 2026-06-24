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

### OpenRouter / OpenAI-Compatible Providers

For OpenRouter or any OpenAI-compatible endpoint, set the base URL:

```bash
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# or
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

Then use `provider="openrouter"`:

```python
ai = AIProcessor(provider="openrouter", model="openai/gpt-4o")
```

The `OPENAI_BASE_URL` env var is also respected by the standard `openai` provider, so you can use it with GitHub Models, Azure AI Studio, or any OpenAI-compatible API.

## Available Providers

### OpenAI
- **Env**: `OPENAI_API_KEY`
- **Models**: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
- **Supports**: chat, embeddings, function calling, streaming, response_format

### OpenRouter
- **Env**: `OPENAI_API_KEY` + `OPENAI_BASE_URL` (or `OPENROUTER_BASE_URL`)
- **Models**: openai/gpt-4o, anthropic/claude-3.5-sonnet, etc.
- **Supports**: chat, function calling, streaming
- Access 200+ models through one API, single billing

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
- OpenAI-compatible API (uses openai Python package)

### Azure OpenAI
- **Env**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_VERSION`
- **Supports**: chat, function calling, streaming

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

### Local (transformers)
- **No API key needed**
- **Models**: any HuggingFace model
- **Supports**: chat, embeddings
- Uses sentence-transformers for embeddings, transformers for text generation

## Feature Matrix

| Feature | openai | openrouter | anthropic | gemini | ollama | azure | bedrock | groq | together | deepseek | mistral | local |
|---------|--------|------------|-----------|--------|--------|-------|---------|------|----------|----------|---------|-------|
| Chat | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Function calling | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Embeddings | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| response_format | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

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

## Streaming AI Responses

```python
from runeextract.processors.ai import AIProcessor

ai = AIProcessor(provider="openai")

# Token-by-token streaming
for chunk in ai._call_stream("You are helpful.", "Tell me a story."):
    print(chunk, end="", flush=True)

# Via Document
doc = extract("report.pdf")
for chunk in doc.ask_stream("What are the key findings?"):
    print(chunk, end="", flush=True)
```

## Multi-Turn Chat

```python
from runeextract import extract

doc = extract("policy.pdf")

# Create chat session with conversation memory
chat = doc.chat(system_prompt="You are a policy expert.")

# Each question remembers previous context
answer1 = chat.ask("What is the remote work policy?")
answer2 = chat.ask("Can you elaborate on the travel reimbursement?")
answer3 = chat.ask("How does this relate to the previous answer?")

# Standalone chat (no document)
from runeextract.models.document import ChatSession
chat = ChatSession(system_prompt="You are a helpful assistant.")
answer = chat.ask("What is Python?")
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

# Streaming question answering
for chunk in doc.ask_stream("Explain in detail."):
    print(chunk, end="", flush=True)

# PII redaction
safe_text = doc.redact_pii()

# Differential privacy
safe_text = doc.redact_pii(use_dp=True, epsilon=5.0)
```
