"""Service for resolving processed text into dictionary signs."""

from backend.schemas.text_processing import TextProcessingResult
from backend.schemas.sign_resolution import SignResolutionResult, ResolvedToken, ResolutionStatus
from backend.services.sign_dictionary_service import SignDictionaryService
from backend.schemas.sign_dictionary import SignNotFoundError


class SignResolutionService:
    """Resolves processed text tokens against the ISL sign dictionary.
    
    Provides deterministic exact matching and greedy phrase matching
    if phrase canonical names exist in the dictionary.
    """
    
    def __init__(self, dictionary_service: SignDictionaryService):
        """Initializes the resolver with a dependency-injected dictionary.
        
        Args:
            dictionary_service: An instance of SignDictionaryService.
        """
        self._dictionary = dictionary_service
        
    def resolve(self, text_result: TextProcessingResult) -> SignResolutionResult:
        """Resolves tokens against the dictionary.
        
        Implements a greedy forward-matching strategy to prioritize longest
        phrase matches over single word matches, falling back to single
        word matches if no phrase matches.
        
        Args:
            text_result: The processed text containing normalized tokens.
            
        Returns:
            SignResolutionResult containing ordered resolution statuses.
        """
        resolved_tokens = []
        tokens = text_result.tokens
        n = len(tokens)
        i = 0
        
        while i < n:
            match_found = False
            # Try to match the longest phrase starting at index i
            for j in range(n, i, -1):
                phrase = " ".join(tokens[i:j])
                try:
                    sign = self._dictionary.get_by_name(phrase)
                    resolved_tokens.append(ResolvedToken(
                        token=phrase,
                        status=ResolutionStatus.MATCHED,
                        sign=sign
                    ))
                    i = j
                    match_found = True
                    break
                except SignNotFoundError:
                    # Expected if the phrase/token isn't in the dictionary
                    continue
                    
            if not match_found:
                # No match found, mark the single token as unsupported
                resolved_tokens.append(ResolvedToken(
                    token=tokens[i],
                    status=ResolutionStatus.UNSUPPORTED,
                    sign=None
                ))
                i += 1
                
        return SignResolutionResult(resolved_tokens=resolved_tokens)
