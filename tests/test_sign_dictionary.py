"""Tests for Phase 6.2 Sign Dictionary Module."""

import pytest

from backend.schemas.sign_dictionary import (
    SignEntry,
    SignNotFoundError,
    DuplicateSignError
)
from backend.services.sign_dictionary_service import SignDictionaryService


@pytest.fixture
def empty_service():
    """Provides an empty dictionary service."""
    return SignDictionaryService(signs=[])


@pytest.fixture
def default_service():
    """Provides a default dictionary service with default vocabulary."""
    return SignDictionaryService()


@pytest.fixture
def custom_service():
    """Provides a dictionary service with custom vocabulary."""
    signs = [
        SignEntry(sign_id="TEST_1", canonical_name="test one"),
        SignEntry(sign_id="TEST_2", canonical_name="test two")
    ]
    return SignDictionaryService(signs=signs)


def test_create_valid_sign_entry():
    """Test A: Create valid sign entry."""
    entry = SignEntry(sign_id="TEST", canonical_name="test")
    assert entry.sign_id == "TEST"
    assert entry.canonical_name == "test"
    assert entry.description is None


def test_lookup_by_sign_id(custom_service):
    """Test B: Lookup by sign_id."""
    entry = custom_service.get_by_id("TEST_1")
    assert entry.sign_id == "TEST_1"
    assert entry.canonical_name == "test one"


def test_lookup_by_canonical_name(custom_service):
    """Test C: Lookup by canonical name."""
    entry = custom_service.get_by_name("test two")
    assert entry.sign_id == "TEST_2"
    assert entry.canonical_name == "test two"


def test_case_behavior(custom_service):
    """Test D: Case behavior according to the documented contract."""
    # Lookup should be case-insensitive and handle whitespace
    entry1 = custom_service.get_by_name("TEST ONE")
    entry2 = custom_service.get_by_name("  TeSt oNe  ")
    
    assert entry1.sign_id == "TEST_1"
    assert entry2.sign_id == "TEST_1"


def test_existing_sign_returns_correctly(default_service):
    """Test E: Existing sign returns correctly."""
    entry = default_service.get_by_id("HELLO")
    assert entry.canonical_name == "hello"


def test_unknown_sign_behavior(default_service):
    """Test F: Unknown sign behavior."""
    with pytest.raises(SignNotFoundError):
        default_service.get_by_id("NONEXISTENT_ID")
        
    with pytest.raises(SignNotFoundError, match="Sign name not found: 'nonexistent name'"):
        default_service.get_by_name("nonexistent name")


def test_contains_existing_sign(default_service):
    """Test G: contains() for existing sign."""
    # By exact ID
    assert default_service.contains("HELLO") is True
    # By canonical name (case-insensitive)
    assert default_service.contains(" HeLlO ") is True


def test_contains_unknown_sign(default_service):
    """Test H: contains() for unknown sign."""
    assert default_service.contains("NONEXISTENT") is False


def test_all_signs_returns_expected_entries(custom_service):
    """Test I: all_signs() returns expected entries."""
    signs = custom_service.all_signs()
    assert len(signs) == 2
    ids = {s.sign_id for s in signs}
    assert "TEST_1" in ids
    assert "TEST_2" in ids


def test_returned_collections_cannot_mutate_state(custom_service):
    """Test J: Returned collections cannot mutate internal dictionary state."""
    # Modify returned list
    signs = custom_service.all_signs()
    signs.clear()
    
    # Internal state should be preserved
    assert len(custom_service.all_signs()) == 2
    
    # Modify returned object
    entry = custom_service.get_by_id("TEST_1")
    entry.canonical_name = "modified"
    
    # Internal state should be preserved
    original_entry = custom_service.get_by_id("TEST_1")
    assert original_entry.canonical_name == "test one"


def test_duplicate_sign_id_rejected():
    """Test K: Duplicate sign_id is rejected."""
    signs = [
        SignEntry(sign_id="TEST", canonical_name="test one"),
        SignEntry(sign_id="TEST", canonical_name="test two")
    ]
    with pytest.raises(DuplicateSignError, match="Duplicate sign_id detected: TEST"):
        SignDictionaryService(signs=signs)


def test_duplicate_canonical_name_rejected():
    """Test L: Duplicate canonical_name is rejected."""
    signs = [
        SignEntry(sign_id="TEST_1", canonical_name="test"),
        SignEntry(sign_id="TEST_2", canonical_name="TEST")
    ]
    with pytest.raises(DuplicateSignError, match="Duplicate canonical_name detected: test"):
        SignDictionaryService(signs=signs)


def test_deterministic_repeated_lookup(custom_service):
    """Test M: Deterministic repeated lookup."""
    entry1 = custom_service.get_by_id("TEST_1")
    entry2 = custom_service.get_by_id("TEST_1")
    assert entry1.model_dump() == entry2.model_dump()


def test_empty_dictionary_behavior(empty_service):
    """Test N: Empty dictionary behavior."""
    assert len(empty_service.all_signs()) == 0
    assert empty_service.contains("HELLO") is False


def test_data_validation_for_malformed_entries():
    """Test O: Data validation for malformed sign entries."""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        # Missing required fields
        SignEntry()


def test_default_dictionary_vocabulary(default_service):
    """Test P: Default dictionary contains expected vocabulary but NOT recognition classes."""
    # Ensure it contains the explicitly allowed development words
    assert default_service.contains("HELLO") is True
    assert default_service.contains("THANK_YOU") is True
    assert default_service.contains("PLEASE") is True
    assert default_service.contains("YES") is True
    assert default_service.contains("NO") is True
    
    # Ensure it does NOT contain the 13 recognition classes
    recognition_classes = [
        "ALIVE", "CLEAN", "DEAD", "DEEP", "DIRTY", "HARD", 
        "HEAVY", "HIGH", "LOW", "SHALLOW", "SOFT", "STRONG", "WEAK"
    ]
    for cls in recognition_classes:
        assert default_service.contains(cls) is False
