"""Tests for Phase 6.3 Sign Resolution Module."""

import pytest

from backend.schemas.text_processing import TextProcessingResult
from backend.schemas.sign_dictionary import SignEntry
from backend.services.sign_dictionary_service import SignDictionaryService
from backend.schemas.sign_resolution import ResolutionStatus
from backend.services.sign_resolution_service import SignResolutionService


@pytest.fixture
def dictionary_service():
    """Provides a deterministic sign dictionary service with known words and phrases."""
    signs = [
        SignEntry(sign_id="HELLO", canonical_name="hello"),
        SignEntry(sign_id="YES", canonical_name="yes"),
        SignEntry(sign_id="NO", canonical_name="no"),
        SignEntry(sign_id="THANK_YOU", canonical_name="thank you")
    ]
    return SignDictionaryService(signs=signs)


@pytest.fixture
def resolution_service(dictionary_service):
    """Provides a configured SignResolutionService."""
    return SignResolutionService(dictionary_service)


def create_text_result(tokens: list[str]) -> TextProcessingResult:
    """Helper to mock a TextProcessingResult."""
    return TextProcessingResult(
        original_text=" ".join(tokens),
        normalized_text=" ".join(tokens).lower(),
        tokens=tokens
    )


def test_single_known_word(resolution_service):
    """Test A: Single known word."""
    result = resolution_service.resolve(create_text_result(["hello"]))
    
    assert len(result.resolved_tokens) == 1
    assert result.resolved_tokens[0].token == "hello"
    assert result.resolved_tokens[0].status == ResolutionStatus.MATCHED
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"


def test_multiple_known_words(resolution_service):
    """Test B: Multiple known words."""
    result = resolution_service.resolve(create_text_result(["hello", "yes"]))
    
    assert len(result.resolved_tokens) == 2
    assert result.resolved_tokens[0].status == ResolutionStatus.MATCHED
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"
    assert result.resolved_tokens[1].status == ResolutionStatus.MATCHED
    assert result.resolved_tokens[1].sign.sign_id == "YES"


def test_mixed_casing(resolution_service):
    """Test C: Mixed casing."""
    # TextProcessingResult naturally lowers text if processed by Phase 6.1
    # We simulate tokens passed with mixed casing to prove resolver delegates to case-insensitive dictionary
    result = resolution_service.resolve(create_text_result(["HeLlO"]))
    
    assert len(result.resolved_tokens) == 1
    assert result.resolved_tokens[0].status == ResolutionStatus.MATCHED
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"


def test_unknown_word(resolution_service):
    """Test D: Unknown word."""
    result = resolution_service.resolve(create_text_result(["banana"]))
    
    assert len(result.resolved_tokens) == 1
    assert result.resolved_tokens[0].token == "banana"
    assert result.resolved_tokens[0].status == ResolutionStatus.UNSUPPORTED
    assert result.resolved_tokens[0].sign is None


def test_mixed_supported_unsupported(resolution_service):
    """Test E & F & O: Mixed supported/unsupported, partial resolution, and order preservation."""
    result = resolution_service.resolve(create_text_result(["hello", "banana", "yes"]))
    
    assert len(result.resolved_tokens) == 3
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"
    assert result.resolved_tokens[1].status == ResolutionStatus.UNSUPPORTED
    assert result.resolved_tokens[1].token == "banana"
    assert result.resolved_tokens[2].sign.sign_id == "YES"


def test_repeated_known_words(resolution_service):
    """Test G: Repeated known words."""
    result = resolution_service.resolve(create_text_result(["hello", "hello"]))
    
    assert len(result.resolved_tokens) == 2
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"
    assert result.resolved_tokens[1].sign.sign_id == "HELLO"


def test_empty_text_result(resolution_service):
    """Test H: Empty TextProcessingResult."""
    result = resolution_service.resolve(create_text_result([]))
    assert len(result.resolved_tokens) == 0


def test_deterministic_repeated_resolution(resolution_service):
    """Test I: Deterministic repeated resolution."""
    input_result = create_text_result(["hello", "banana", "yes"])
    res1 = resolution_service.resolve(input_result)
    res2 = resolution_service.resolve(input_result)
    
    assert res1.model_dump() == res2.model_dump()


def test_phrase_matching_precedence(resolution_service):
    """Test L: Phrase matching precedence."""
    # "thank you" should resolve as a single phrase rather than two unsupported words
    result = resolution_service.resolve(create_text_result(["thank", "you"]))
    
    assert len(result.resolved_tokens) == 1
    assert result.resolved_tokens[0].token == "thank you"
    assert result.resolved_tokens[0].status == ResolutionStatus.MATCHED
    assert result.resolved_tokens[0].sign.sign_id == "THANK_YOU"


def test_phrase_matching_with_mixed_context(resolution_service):
    """Verify phrase matching works smoothly surrounded by other words."""
    result = resolution_service.resolve(create_text_result(["hello", "thank", "you", "banana"]))
    
    assert len(result.resolved_tokens) == 3
    assert result.resolved_tokens[0].sign.sign_id == "HELLO"
    assert result.resolved_tokens[1].sign.sign_id == "THANK_YOU"
    assert result.resolved_tokens[2].status == ResolutionStatus.UNSUPPORTED


def test_no_mutation(resolution_service, dictionary_service):
    """Test N: Ensure resolution does not mutate input or dictionary state."""
    input_result = create_text_result(["hello"])
    original_tokens = list(input_result.tokens)
    
    resolution_service.resolve(input_result)
    
    # Input result tokens should remain identical
    assert input_result.tokens == original_tokens
    
    # Dictionary state should be completely unharmed
    assert dictionary_service.contains("HELLO")
