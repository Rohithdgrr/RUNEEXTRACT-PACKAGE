"""Concrete pipeline steps for extraction, chunking, AI, embedding, filtering, and storage."""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

from runeextract.transform.pipeline import PipelineContext, PipelineStep
from runeextract.models.document import Document, ChunkingStrategy

logger = logging.getLogger(__name__)


class ExtractStep(PipelineStep):
    """Extract a single file into a Document.

    Kwargs:
        file_path (str): Path to the document file.
        ocr (bool): Enable OCR (default False).
        tables (bool): Extract tables (default True).
        images (bool): Extract images (default True).
        metadata (bool): Extract metadata (default True).
        **kwargs: Additional options passed to ``runeextract.extract()``.
    """

    def run(self, ctx: PipelineContext) -> Document:
        from runeextract import extract
        file_path = self._kwargs.get("file_path", "")
        if not file_path:
            raise ValueError("ExtractStep requires file_path")
        kw = {k: v for k, v in self._kwargs.items() if k != "file_path"}
        doc = extract(file_path, **kw)
        ctx.documents.append(doc)
        return doc


class ExtractManyStep(PipelineStep):
    """Extract multiple files into Documents.

    Kwargs:
        file_paths (List[str]): Paths to document files.
        **kwargs: Additional options passed to ``runeextract.extract()``.
    """

    def run(self, ctx: PipelineContext) -> List[Document]:
        from runeextract import extract
        paths: list = self._kwargs.get("file_paths", [])
        if not paths:
            raise ValueError("ExtractManyStep requires file_paths")
        kw = {k: v for k, v in self._kwargs.items() if k != "file_paths"}
        docs = []
        for fp in paths:
            try:
                doc = extract(fp, **kw)
                docs.append(doc)
                ctx.documents.append(doc)
            except Exception as exc:
                logger.warning("ExtractManyStep skipped %s: %s", fp, exc)
        return docs


class ChunkStep(PipelineStep):
    """Chunk documents in the context.

    Kwargs:
        strategy (str or ChunkingStrategy): Chunking strategy.
        chunk_size (int): Target chunk size (default 1000).
        chunk_overlap (int): Overlap between chunks (default 100).
        target (str): Which documents to chunk — ``"all"`` (default) or
            ``"last"`` (most recently added).
    """

    def run(self, ctx: PipelineContext) -> List[Document]:
        strategy = self._kwargs.get("strategy", ChunkingStrategy.FIXED_SIZE)
        size = self._kwargs.get("chunk_size", 1000)
        overlap = self._kwargs.get("chunk_overlap", 100)
        target = self._kwargs.get("target", "all")

        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)

        docs_to_chunk: List[Document] = ctx.documents
        if target == "last" and ctx.documents:
            docs_to_chunk = [ctx.documents[-1]]

        for doc in docs_to_chunk:
            doc.chunks(strategy=strategy, size=size, overlap=overlap)

        return ctx.documents


class FilterStep(PipelineStep):
    """Filter documents based on a predicate.

    Kwargs:
        predicate (Callable[[Document], bool]): Function returning True
            to keep a document.
        min_length (int): Keep documents with at least this many characters.
        max_length (int): Keep documents with at most this many characters.
    """

    def run(self, ctx: PipelineContext) -> List[Document]:
        pred = self._kwargs.get("predicate", None)
        min_len = self._kwargs.get("min_length", 0)
        max_len = self._kwargs.get("max_length", 0)

        def _default_pred(d: Document) -> bool:
            if min_len and len(d.text) < min_len:
                return False
            if max_len and len(d.text) > max_len:
                return False
            return True

        predicate = pred or _default_pred
        before = len(ctx.documents)
        ctx.documents = [d for d in ctx.documents if predicate(d)]
        after = len(ctx.documents)
        logger.info("FilterStep removed %d documents", before - after)
        return ctx.documents


class MapStep(PipelineStep):
    """Apply a function to each document, optionally replacing it.

    Kwargs:
        fn (Callable[[Document], Optional[Document]]): Transformation
            function. Return None to drop the document.
    """

    def run(self, ctx: PipelineContext) -> List[Document]:
        fn: Callable = self._kwargs.get("fn")
        if fn is None:
            raise ValueError("MapStep requires fn")
        new_docs = []
        for doc in ctx.documents:
            result = fn(doc)
            if result is not None:
                new_docs.append(result)
        ctx.documents = new_docs
        return ctx.documents


_shared_ai = None


def _get_ai():
    global _shared_ai
    if _shared_ai is None:
        from runeextract.processors.ai import AIProcessor
        _shared_ai = AIProcessor()
    return _shared_ai


def _reset_ai():
    """Reset the shared AI singleton (used in tests)."""
    global _shared_ai
    _shared_ai = None


class AIStep(PipelineStep):
    """Apply AI processing (summarize, extract_entities, Q&A, etc.) to documents.

    Kwargs:
        action (str): One of ``"summarize"``, ``"extract_keywords"``,
            ``"extract_entities"``, ``"generate_questions"``,
            ``"generate_flashcards"``, ``"answer_question"``,
            or ``"call"`` for a custom prompt.
        prompt (str): Custom system prompt (only for action="call").
        question (str): Question for action="answer_question".
        max_words (int): Max words for summarize.
        store_in (str): Metadata key to store the result
            (default: the action name).
        ai_kwargs (dict): Additional kwargs passed to the AIProcessor method.
    """

    def run(self, ctx: PipelineContext) -> List[Dict[str, Any]]:
        ai = _get_ai()

        action = self._kwargs.get("action", "summarize")
        store_in = self._kwargs.get("store_in", action)
        ai_kw = self._kwargs.get("ai_kwargs", {})

        results = []

        for doc in ctx.documents:
            text = doc.text
            if action == "summarize":
                max_words = self._kwargs.get("max_words", 200)
                result = ai.summarize(text, max_words=max_words, **ai_kw)
            elif action == "extract_keywords":
                top_n = self._kwargs.get("top_n", 10)
                result = ai.extract_keywords(text, top_n=top_n)
            elif action == "extract_entities":
                result = ai.extract_entities(text)
            elif action == "generate_questions":
                n = self._kwargs.get("n", 5)
                result = ai.generate_questions(text, n=n, **ai_kw)
            elif action == "generate_flashcards":
                n = self._kwargs.get("n", 10)
                result = ai.generate_flashcards(text, n=n, **ai_kw)
            elif action == "answer_question":
                question = self._kwargs.get("question", "")
                result = ai.answer_question(question, text, **ai_kw)
            elif action == "call":
                prompt = self._kwargs.get("prompt", "")
                result = ai._call(prompt, text, **ai_kw)
            else:
                raise ValueError(f"Unknown AIStep action: {action}")

            doc.metadata[store_in] = result
            results.append(result)

        return results


class EmbedStep(PipelineStep):
    """Embed documents and store vectors in metadata.

    Kwargs:
        provider (str): Embedding provider (default "openai").
        store_in (str): Metadata key to store the vector (default "embedding").
        ai_kwargs (dict): Additional kwargs for AIProcessor.embed().
    """

    def run(self, ctx: PipelineContext) -> List[List[float]]:
        ai = _get_ai()

        store_in = self._kwargs.get("store_in", "embedding")
        ai_kw = self._kwargs.get("ai_kwargs", {})

        results = []
        for doc in ctx.documents:
            vector = ai.embed(doc.text, **ai_kw)
            doc.metadata[store_in] = vector
            results.append(vector)

        return results


class StoreStep(PipelineStep):
    """Store pipeline results to disk or a vector store.

    Kwargs:
        output_dir (str): Directory to write JSON results (optional).
        vector_store (str): One of ``"chromadb"`` or ``"faiss"`` (optional).
        collection_name (str): ChromaDB collection name (default "pipeline").
        persist_directory (str): Vector store persist dir (default "./chroma_db").
        format (str): Output format for JSON — ``"json"`` or ``"jsonl"``.
        include_chunks (bool): Include chunk data in JSON output (default True).
    """

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        output_dir = self._kwargs.get("output_dir", "")
        vector_store = self._kwargs.get("vector_store", "")
        store_format = self._kwargs.get("format", "json")
        include_chunks = self._kwargs.get("include_chunks", True)

        result = {"document_count": len(ctx.documents), "documents": []}

        for doc in ctx.documents:
            entry = {
                "source_path": doc.source_path,
                "source_type": doc.source_type,
                "text_length": len(doc.text),
                "metadata": doc.metadata,
                "tables_count": len(doc.tables),
                "images_count": len(doc.images),
            }
            if include_chunks and doc._chunks:
                entry["chunks"] = [
                    {"text": c.text, "chunk_id": c.chunk_id}
                    for c in doc._chunks
                ]
            result["documents"].append(entry)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if store_format == "jsonl":
                out_path = os.path.join(output_dir, "pipeline_output.jsonl")
                with open(out_path, "w") as f:
                    for entry in result["documents"]:
                        f.write(json.dumps(entry) + "\n")
            else:
                out_path = os.path.join(output_dir, "pipeline_output.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            logger.info("Stored pipeline output to %s", out_path)

        if vector_store:
            for doc in ctx.documents:
                if vector_store == "chromadb":
                    doc.to_chromadb(
                        collection_name=self._kwargs.get("collection_name", "pipeline"),
                        persist_directory=self._kwargs.get("persist_directory", "./chroma_db"),
                    )
                elif vector_store == "faiss":
                    doc.to_faiss(
                        index_path=self._kwargs.get("persist_directory", "./faiss_index"),
                    )

        return result


class LogStep(PipelineStep):
    """Log pipeline progress message.

    Kwargs:
        message (str): Message template with ``{doc_count}`` placeholder.
    """

    def run(self, ctx: PipelineContext) -> str:
        message = self._kwargs.get("message", "Pipeline progress: {doc_count} documents")
        formatted = message.format(doc_count=len(ctx.documents))
        logger.info(formatted)
        return formatted
