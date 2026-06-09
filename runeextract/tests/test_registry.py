"""
Tests for extractor registry.
"""

import pytest
from runeextract.core.registry import ExtractorRegistry, register_extractor
from runeextract.core.extractor import BaseExtractor


class CustomExtractor(BaseExtractor):
    """Custom test extractor."""
    
    def extract(self, file_path: str):
        from runeextract.models.document import Document
        return Document(text="custom", source_type="custom")
    
    def supported_extensions(self):
        return [".custom"]


def test_register_extractor():
    """Test registering a custom extractor."""
    ExtractorRegistry.register(".custom")(CustomExtractor)
    
    assert ExtractorRegistry.is_registered(".custom")
    assert ExtractorRegistry.get_extractor(".custom") == CustomExtractor


def test_unregister_extractor():
    """Test unregistering an extractor."""
    ExtractorRegistry.register(".temp")(CustomExtractor)
    
    assert ExtractorRegistry.is_registered(".temp")
    
    result = ExtractorRegistry.unregister(".temp")
    assert result is True
    assert not ExtractorRegistry.is_registered(".temp")


def test_registered_extensions():
    """Test getting all registered extensions."""
    initial_count = len(ExtractorRegistry.registered_extensions())
    
    ExtractorRegistry.register(".test1")(CustomExtractor)
    ExtractorRegistry.register(".test2")(CustomExtractor)
    
    new_count = len(ExtractorRegistry.registered_extensions())
    assert new_count >= initial_count + 2
    
    # Cleanup
    ExtractorRegistry.unregister(".test1")
    ExtractorRegistry.unregister(".test2")


def test_decorator_registration():
    """Test using decorator for registration."""
    @register_extractor(".decorated")
    class DecoratedExtractor(BaseExtractor):
        def extract(self, file_path: str):
            from runeextract.models.document import Document
            return Document(text="decorated", source_type="decorated")
        
        def supported_extensions(self):
            return [".decorated"]
    
    assert ExtractorRegistry.is_registered(".decorated")
    
    # Cleanup
    ExtractorRegistry.unregister(".decorated")
