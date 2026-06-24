from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:
    from runeextract.models.document import Document


class ChatSession:
    """Multi-turn conversation memory for document Q&A.

    Retains conversation history across ``ask()`` and ``ask_stream()`` calls
    so the LLM has context from previous exchanges.

    Args:
        document: Optional Document instance for RAG-based answers
        system_prompt: Optional system prompt prepended to the conversation
        ai_processor: Optional AIProcessor instance (creates a default if omitted)

    Example::

        doc = extract("report.pdf")
        chat = ChatSession(doc)
        print(chat.ask("What are the key findings?"))
        print(chat.ask("How do they compare to last quarter?"))
        for token in chat.ask_stream("Summarize the main trends"):
            print(token, end="")
    """

    def __init__(self, document: Optional[Document] = None,
                 system_prompt: Optional[str] = None,
                 ai_processor=None):
        self.document = document
        self.ai_processor = ai_processor
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _get_ai(self):
        if self.ai_processor is not None:
            return self.ai_processor
        from runeextract.models.document import _get_document_ai
        return _get_document_ai(None)

    def add_user_message(self, content: str):
        """Append a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Append an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})

    def _format_history(self) -> str:
        lines = []
        for msg in self.messages:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def ask(self, question: str, top_k: int = 5) -> str:
        """Ask a question, retaining conversation history.

        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve as RAG context

        Returns:
            Answer string
        """
        rag_context = ""
        if self.document:
            results = self.document.retrieve(question, top_k=top_k)
            if results:
                rag_context = "\n\n".join(chunk.text for chunk, _ in results)

        self.add_user_message(question)
        history = self._format_history()
        ai = self._get_ai()

        if rag_context:
            user = f"Conversation history:\n{history}\n\nRelevant context:\n{rag_context}\n\nAnswer the latest question based on the context and conversation history."
        else:
            user = f"Conversation history:\n{history}\n\nAnswer the latest question."
        if not any(m["role"] == "system" for m in self.messages):
            system = ("You are a helpful assistant. Answer based on the provided context. "
                      "If the context doesn't contain the answer, say so.")
        else:
            system = ""

        answer = ai._call(system, user, max_tokens=2000)
        self.add_assistant_message(answer)
        return answer

    def ask_stream(self, question: str, top_k: int = 5):
        """Ask a question and yield answer tokens as they arrive.

        Args:
            question: Natural language question
            top_k: Number of chunks to retrieve as RAG context

        Yields:
            String tokens of the answer as they arrive from the AI.
        """
        rag_context = ""
        if self.document:
            results = self.document.retrieve(question, top_k=top_k)
            if results:
                rag_context = "\n\n".join(chunk.text for chunk, _ in results)

        self.add_user_message(question)
        history = self._format_history()
        ai = self._get_ai()

        if rag_context:
            user = f"Conversation history:\n{history}\n\nRelevant context:\n{rag_context}\n\nAnswer the latest question based on the context and conversation history."
        else:
            user = f"Conversation history:\n{history}\n\nAnswer the latest question."
        if not any(m["role"] == "system" for m in self.messages):
            system = ("You are a helpful assistant. Answer based on the provided context. "
                      "If the context doesn't contain the answer, say so.")
        else:
            system = ""

        collected = []
        for token in ai._call_stream(system, user, max_tokens=2000):
            collected.append(token)
            yield token
        self.add_assistant_message("".join(collected))
