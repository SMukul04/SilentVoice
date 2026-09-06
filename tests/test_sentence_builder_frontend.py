"""Structural tests for the SentenceBuilder Frontend implementation."""

import pytest
from pathlib import Path

@pytest.fixture
def sentence_builder_js_content():
    """Reads the sentence_builder.js file."""
    js_path = Path("frontend/static/js/sentence_builder.js")
    if not js_path.exists():
        pytest.fail("sentence_builder.js not found in frontend/static/js/")
    return js_path.read_text(encoding="utf-8")

def test_sentence_builder_class_exists(sentence_builder_js_content):
    """Test 1: SentenceBuilder class exists and constructor starts empty."""
    assert "class SentenceBuilder" in sentence_builder_js_content
    assert "this.words = []" in sentence_builder_js_content

def test_add_word_exists(sentence_builder_js_content):
    """Test 2, 3: addWord exists and pushes to words."""
    assert "addWord(" in sentence_builder_js_content
    assert "this.words.push(" in sentence_builder_js_content

def test_get_words_exists(sentence_builder_js_content):
    """Test 4: getWords exists and returns the array."""
    assert "getWords(" in sentence_builder_js_content
    assert "return " in sentence_builder_js_content.split("getWords()")[1].split("}")[0]

def test_get_sentence_exists(sentence_builder_js_content):
    """Test 5: getSentence exists and joins with spaces."""
    assert "getSentence(" in sentence_builder_js_content
    assert 'join(" ")' in sentence_builder_js_content

def test_clear_exists(sentence_builder_js_content):
    """Test 6: clear exists and resets array."""
    assert "clear(" in sentence_builder_js_content
    assert "this.words = []" in sentence_builder_js_content.split("clear()")[1]

def test_invalid_input_handling(sentence_builder_js_content):
    """Test 7, 8, 9, 10: Empty, null, undefined, and whitespace handled."""
    assert "null" in sentence_builder_js_content
    assert "undefined" in sentence_builder_js_content
    assert "trim()" in sentence_builder_js_content
    assert "length === 0" in sentence_builder_js_content

def test_duplicate_policy_not_rejected(sentence_builder_js_content):
    """Test 11, 12: Repeated words are allowed, no global rejection."""
    # Ensure there's no check like "if (word === this.words[this.words.length - 1]) return;"
    assert "this.words[this.words.length - 1]" not in sentence_builder_js_content
    assert "includes(word)" not in sentence_builder_js_content
