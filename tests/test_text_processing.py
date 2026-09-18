"""Tests for Phase 6.1 Text Processing Module."""

import pytest

from backend.schemas.text_processing import EmptyTextError
from backend.services.text_processing_service import TextProcessingService


@pytest.fixture
def service():
    """Provides a fresh instance of TextProcessingService."""
    return TextProcessingService()


def test_normal_sentence(service):
    """Test A: Normal sentence."""
    input_text = "Hello how are you"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello how are you"
    assert result.tokens == ["hello", "how", "are", "you"]


def test_mixed_casing(service):
    """Test B: Mixed casing."""
    input_text = "HELLO How ARE You"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello how are you"
    assert result.tokens == ["hello", "how", "are", "you"]


def test_leading_trailing_whitespace(service):
    """Test C: Leading/trailing whitespace."""
    input_text = "   Hello how are you   "
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello how are you"
    assert result.tokens == ["hello", "how", "are", "you"]


def test_repeated_internal_whitespace(service):
    """Test D: Repeated internal whitespace."""
    input_text = "Hello    how   are      you"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello how are you"
    assert result.tokens == ["hello", "how", "are", "you"]


def test_punctuation(service):
    """Test E: Punctuation."""
    input_text = "Hello, how are you?"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello, how are you?"
    assert result.tokens == ["hello", "how", "are", "you"]


def test_multiple_punctuation_marks(service):
    """Test F: Multiple punctuation marks."""
    input_text = "Wow!!! That is... amazing! Right???"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "wow!!! that is... amazing! right???"
    assert result.tokens == ["wow", "that", "is", "amazing", "right"]


def test_empty_string(service):
    """Test G: Empty string."""
    with pytest.raises(EmptyTextError):
        service.process("")


def test_whitespace_only_string(service):
    """Test H: Whitespace-only string."""
    with pytest.raises(EmptyTextError):
        service.process("     \t\n   ")


def test_single_word(service):
    """Test I: Single word."""
    input_text = "Hello"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello"
    assert result.tokens == ["hello"]


def test_repeated_words(service):
    """Test J: Repeated words."""
    input_text = "hello hello hello"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "hello hello hello"
    assert result.tokens == ["hello", "hello", "hello"]


def test_numbers(service):
    """Test K: Numbers."""
    input_text = "I have 3 apples."
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "i have 3 apples."
    assert result.tokens == ["i", "have", "3", "apples"]


def test_determinism(service):
    """Test L: Determinism - same input produces exact same output repeatedly."""
    input_text = "  HELLO,   HOW are you?  "
    
    result1 = service.process(input_text)
    result2 = service.process(input_text)
    
    assert result1.model_dump() == result2.model_dump()


def test_internal_punctuation_preserved(service):
    """Extra test: Internal punctuation (like apostrophes) should be preserved in tokens."""
    input_text = "Don't forget the co-op!"
    result = service.process(input_text)
    
    assert result.original_text == input_text
    assert result.normalized_text == "don't forget the co-op!"
    assert result.tokens == ["don't", "forget", "the", "co-op"]


def test_none_input(service):
    """Ensure None is handled correctly."""
    with pytest.raises(EmptyTextError, match="Input text cannot be None"):
        service.process(None)
