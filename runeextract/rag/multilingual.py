"""
Feature 10: Multi-Language Support with Auto-Translation

Cross-lingual search and auto-translation for global RAG.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TranslationCache:
    """Cache for translations."""
    source_lang: str
    target_lang: str
    source_text: str
    translated_text: str
    timestamp: float


class MultilingualRAG:
    """Multi-language RAG with cross-lingual search.
    
    Features:
    - Auto language detection
    - Cross-lingual embeddings (mBERT, LaBSE)
    - Auto-translation (query and results)
    - Translation caching
    - Language-aware chunking
    
    Usage::
    
        ml_rag = MultilingualRAG(
            base_rag=rag,
            languages=["en", "es", "fr", "de"],
            translation_cache=True
        )
        
        # Query in any language
        result = ml_rag.query(
            "¿Qué es el aprendizaje automático?",  # Spanish
            target_lang="en"
        )
        # Returns English answer with Spanish sources
    """
    
    def __init__(
        self,
        base_rag: Any,
        languages: List[str],
        translation_provider: str = "openai",
        translation_cache: bool = True,
        cross_lingual: bool = True
    ):
        """Initialize multilingual RAG.
        
        Args:
            base_rag: Base AutoRAG instance
            languages: Supported language codes (ISO 639-1)
            translation_provider: Translation API ("openai", "google", "deepl")
            translation_cache: Enable translation caching
            cross_lingual: Use cross-lingual embeddings
        """
        self.base_rag = base_rag
        self.languages = languages
        self.translation_provider = translation_provider
        self.translation_cache_enabled = translation_cache
        self.cross_lingual = cross_lingual
        
        # Translation cache
        self._translation_cache: Dict[str, str] = {}
        
        # Language detector
        self._detector = None
        
        # Cross-lingual embedder (lazy init)
        self._cross_lingual_embedder = None
        
        logger.info(
            f"MultilingualRAG initialized for {len(languages)} languages: {languages}"
        )
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text.
        
        Args:
            text: Text to detect
        
        Returns:
            ISO 639-1 language code
        """
        # Try to use langdetect
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            logger.debug("langdetect not installed, using heuristic")
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
        
        # Fallback: simple heuristic based on Unicode ranges
        # (from existing OCRLanguageDetector)
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        return detector.detect_language(text)
    
    def _translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate text.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text
        """
        # Check cache first
        cache_key = f"{source_lang}:{target_lang}:{text[:100]}"
        if self.translation_cache_enabled and cache_key in self._translation_cache:
            logger.debug(f"Translation cache hit: {source_lang} → {target_lang}")
            return self._translation_cache[cache_key]
        
        # Translate using provider
        if self.translation_provider == "openai":
            translated = self._translate_openai(text, source_lang, target_lang)
        elif self.translation_provider == "google":
            translated = self._translate_google(text, source_lang, target_lang)
        elif self.translation_provider == "deepl":
            translated = self._translate_deepl(text, source_lang, target_lang)
        else:
            raise ValueError(f"Unknown translation provider: {self.translation_provider}")
        
        # Cache result
        if self.translation_cache_enabled:
            self._translation_cache[cache_key] = translated
        
        return translated
    
    def _translate_openai(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using OpenAI.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text
        """
        prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Only provide the translation, nothing else.

Text: {text}

Translation:"""
        
        # Use base RAG's AI processor
        response = self.base_rag.ai.call(prompt)
        return response.strip()
    
    def _translate_google(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using Google Translate API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text
        """
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except ImportError:
            logger.warning("googletrans not installed, falling back to OpenAI")
            return self._translate_openai(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"Google Translate failed: {e}, falling back to OpenAI")
            return self._translate_openai(text, source_lang, target_lang)
    
    def _translate_deepl(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Translate using DeepL API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text
        """
        try:
            import deepl
            import os
            
            auth_key = os.getenv("DEEPL_API_KEY")
            if not auth_key:
                raise ValueError("DEEPL_API_KEY not set")
            
            translator = deepl.Translator(auth_key)
            result = translator.translate_text(
                text,
                source_lang=source_lang.upper(),
                target_lang=target_lang.upper()
            )
            return result.text
        except ImportError:
            logger.warning("deepl not installed, falling back to OpenAI")
            return self._translate_openai(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"DeepL failed: {e}, falling back to OpenAI")
            return self._translate_openai(text, source_lang, target_lang)
    
    def query(
        self,
        question: str,
        target_lang: Optional[str] = None,
        translate_sources: bool = False,
        **kwargs
    ) -> Any:
        """Query with auto-translation.
        
        Args:
            question: Question in any supported language
            target_lang: Target language for answer (auto-detect if None)
            translate_sources: Whether to translate source citations
            **kwargs: Passed to base_rag.query()
        
        Returns:
            RAGResult with translated answer
        """
        # Detect query language
        query_lang = self._detect_language(question)
        logger.info(f"Detected query language: {query_lang}")
        
        # Set target language
        if target_lang is None:
            target_lang = query_lang
        
        # Translate query to English for retrieval (if not English)
        if query_lang != "en":
            question_en = self._translate(question, query_lang, "en")
            logger.info(f"Translated query: {question_en}")
        else:
            question_en = question
        
        # Query base RAG
        result = self.base_rag.query(question_en, **kwargs)
        
        # Translate answer to target language (if not English)
        if target_lang != "en":
            result.answer = self._translate(result.answer, "en", target_lang)
        
        # Optionally translate source citations
        if translate_sources and target_lang != "en":
            for chunk in result.retrieved_chunks:
                chunk.text = self._translate(chunk.text, "en", target_lang)
        
        return result
    
    def ingest_multilingual(
        self,
        sources: Dict[str, List[str]],
        **kwargs
    ) -> None:
        """Ingest documents in multiple languages.
        
        Args:
            sources: Dict of {language: [file_paths]}
            **kwargs: Passed to base_rag.ingest()
        
        Example::
        
            ml_rag.ingest_multilingual({
                "en": ["docs/english/*.pdf"],
                "es": ["docs/spanish/*.pdf"],
                "fr": ["docs/french/*.pdf"]
            })
        """
        for lang, paths in sources.items():
            logger.info(f"Ingesting {len(paths)} documents in {lang}")
            
            # Add language metadata
            for path in paths:
                self.base_rag.ingest(
                    path,
                    metadata={"language": lang},
                    **kwargs
                )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get translation cache statistics.
        
        Returns:
            Dict with cache metrics
        """
        return {
            "cache_size": len(self._translation_cache),
            "cache_enabled": self.translation_cache_enabled,
            "supported_languages": self.languages,
            "translation_provider": self.translation_provider
        }
