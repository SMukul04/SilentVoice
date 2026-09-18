"""Tests for Phase 6.4 Sign Sequence Module."""

import pytest

from backend.schemas.sign_dictionary import SignEntry
from backend.schemas.sign_resolution import SignResolutionResult, ResolvedToken, ResolutionStatus
from backend.services.sign_sequence_service import SignSequenceService


@pytest.fixture
def sequence_service():
    """Provides a fresh instance of SignSequenceService."""
    return SignSequenceService()


def create_resolution(tokens_data: list[tuple[str, str | None]]) -> SignResolutionResult:
    """Helper to mock a SignResolutionResult.
    
    Args:
        tokens_data: List of tuples (token_text, sign_id).
                     If sign_id is None, it's mocked as UNSUPPORTED.
    """
    resolved_tokens = []
    for token, sign_id in tokens_data:
        if sign_id:
            resolved_tokens.append(ResolvedToken(
                token=token,
                status=ResolutionStatus.MATCHED,
                sign=SignEntry(sign_id=sign_id, canonical_name=token)
            ))
        else:
            resolved_tokens.append(ResolvedToken(
                token=token,
                status=ResolutionStatus.UNSUPPORTED,
                sign=None
            ))
            
    return SignResolutionResult(resolved_tokens=resolved_tokens)


def test_single_matched_sign(sequence_service):
    """Test A: Single matched sign."""
    res = create_resolution([("hello", "HELLO")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 1
    assert result.sequence[0].sign_id == "HELLO"
    assert result.sequence[0].source_text == "hello"
    assert result.sequence[0].original_index == 0
    assert len(result.unsupported_tokens) == 0


def test_multiple_matched_signs(sequence_service):
    """Test B: Multiple matched signs."""
    res = create_resolution([("hello", "HELLO"), ("yes", "YES"), ("please", "PLEASE")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 3
    assert [s.sign_id for s in result.sequence] == ["HELLO", "YES", "PLEASE"]
    assert [s.original_index for s in result.sequence] == [0, 1, 2]
    assert len(result.unsupported_tokens) == 0


def test_ordering_preservation(sequence_service):
    """Test C: Ordering preservation."""
    res = create_resolution([("yes", "YES"), ("hello", "HELLO")])
    result = sequence_service.build_sequence(res)
    
    assert result.sequence[0].sign_id == "YES"
    assert result.sequence[1].sign_id == "HELLO"


def test_repeated_signs(sequence_service):
    """Test D: Repeated signs."""
    res = create_resolution([("yes", "YES"), ("yes", "YES"), ("yes", "YES")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 3
    assert [s.sign_id for s in result.sequence] == ["YES", "YES", "YES"]


def test_phrase_result(sequence_service):
    """Test E: Phrase result yields exactly one sequence item."""
    res = create_resolution([("thank you", "THANK_YOU")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 1
    assert result.sequence[0].sign_id == "THANK_YOU"
    assert result.sequence[0].source_text == "thank you"


def test_mixed_supported_unsupported(sequence_service):
    """Test F: Mixed supported/unsupported preserving original indices."""
    res = create_resolution([("hello", "HELLO"), ("banana", None), ("yes", "YES")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 2
    assert result.sequence[0].sign_id == "HELLO"
    assert result.sequence[0].original_index == 0
    
    assert result.sequence[1].sign_id == "YES"
    assert result.sequence[1].original_index == 2
    
    assert len(result.unsupported_tokens) == 1
    assert result.unsupported_tokens[0] == "banana"


def test_all_unsupported_input(sequence_service):
    """Test G: All unsupported input produces empty sequence and explicit unsupported information."""
    res = create_resolution([("banana", None), ("apple", None)])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 0
    assert len(result.unsupported_tokens) == 2
    assert result.unsupported_tokens == ["banana", "apple"]


def test_empty_valid_resolution_result(sequence_service):
    """Test H: Empty valid resolution result."""
    res = create_resolution([])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 0
    assert len(result.unsupported_tokens) == 0


def test_deterministic_repeated_sequence_construction(sequence_service):
    """Test I: Deterministic repeated sequence construction."""
    res = create_resolution([("hello", "HELLO"), ("banana", None)])
    
    result1 = sequence_service.build_sequence(res)
    result2 = sequence_service.build_sequence(res)
    
    assert result1.model_dump() == result2.model_dump()


def test_input_resolution_object_is_not_mutated(sequence_service):
    """Test J: Input resolution object is not mutated."""
    res = create_resolution([("hello", "HELLO")])
    original_tokens = list(res.resolved_tokens)
    
    sequence_service.build_sequence(res)
    
    assert res.resolved_tokens == original_tokens


def test_sequence_output_cannot_mutate_internal_state(sequence_service):
    """Test K: Sequence output cannot accidentally mutate internal state."""
    res = create_resolution([("hello", "HELLO")])
    result = sequence_service.build_sequence(res)
    
    # Mutating the returned array should not affect subsequent calls if we were caching (we aren't)
    # But fundamentally pydantic models return distinct lists anyway.
    result.sequence.clear()
    
    assert len(res.resolved_tokens) == 1


def test_no_fabricated_signs_for_unsupported_content(sequence_service):
    """Test N: No fabricated signs for unsupported content."""
    res = create_resolution([("banana", None)])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 0
    assert "banana" in result.unsupported_tokens
    # Ensures we don't accidentally map it to 'HELLO' or 'DEFAULT'
    assert not any(s.sign_id == "DEFAULT" for s in result.sequence)


def test_service_does_not_perform_independent_tokenization(sequence_service):
    """Test O: Ensure the service does NOT perform independent tokenization."""
    # We pass a bizarre token that contains spaces and symbols. If the service
    # blindly splits strings, it would break. By trusting the resolution object,
    # it safely passes it through.
    res = create_resolution([("b i z a r r e !! token", "BIZARRE")])
    result = sequence_service.build_sequence(res)
    
    assert len(result.sequence) == 1
    assert result.sequence[0].source_text == "b i z a r r e !! token"
    assert result.sequence[0].sign_id == "BIZARRE"
