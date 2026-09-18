"""Service for deterministic text processing and normalization."""

import re
import string

from backend.schemas.text_processing import TextProcessingResult, EmptyTextError


class TextProcessingService:
    """A deterministic service for processing and tokenizing text.
    
    This service prepares arbitrary user text for later processing stages
    (like Text->Sign resolution) by normalizing casing and whitespace,
    and extracting tokens while cleanly handling punctuation.
    """
    
    def process(self, text: str | None) -> TextProcessingResult:
        """Process the input text into a normalized form and extract tokens.
        
        Args:
            text: The original input string.
            
        Returns:
            TextProcessingResult containing the original, normalized text and tokens.
            
        Raises:
            EmptyTextError: If the input is None, empty, or only whitespace.
        """
        if text is None:
            raise EmptyTextError("Input text cannot be None.")
            
        original_text = text
        
        # Normalize: strip leading/trailing whitespace, collapse repeated internal whitespace, and lowercase
        normalized = text.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        if not normalized:
            raise EmptyTextError("Input text is empty or contains only whitespace.")
            
        # Tokenize: split by whitespace
        raw_tokens = normalized.split(' ')
        tokens = []
        
        for t in raw_tokens:
            # Strip punctuation from the beginning and end of each word.
            # This preserves internal punctuation (e.g. "don't", "co-op").
            cleaned = t.strip(string.punctuation)
            if cleaned:
                tokens.append(cleaned)
                
        return TextProcessingResult(
            original_text=original_text,
            normalized_text=normalized,
            tokens=tokens
        )
