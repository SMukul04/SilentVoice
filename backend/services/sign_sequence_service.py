"""Service for constructing ordered sign sequences."""

from backend.schemas.sign_resolution import SignResolutionResult, ResolutionStatus
from backend.schemas.sign_sequence import SignSequenceResult, SignSequenceItem


class SignSequenceService:
    """Constructs an abstract ordered sequence of signs for downstream playback.
    
    This service strictly transforms resolved tokens into a clean sequential format.
    It does NOT implement grammatical English->ISL translations, deduplication, 
    or avatar generation.
    """
    
    def build_sequence(self, resolution_result: SignResolutionResult) -> SignSequenceResult:
        """Builds the sequence strictly from the resolution result.
        
        Preserves ordering, extracts unsupported tokens, and retains
        multi-word phrase resolutions exactly as provided by Phase 6.3.
        
        Args:
            resolution_result: The structured output from Phase 6.3 SignResolutionService.
            
        Returns:
            A constructed SignSequenceResult explicitly separating supported
            sequence items from unsupported tokens.
        """
        sequence = []
        unsupported = []
        
        for idx, resolved_token in enumerate(resolution_result.resolved_tokens):
            if resolved_token.status == ResolutionStatus.MATCHED and resolved_token.sign is not None:
                sequence.append(SignSequenceItem(
                    sign_id=resolved_token.sign.sign_id,
                    source_text=resolved_token.token,
                    original_index=idx
                ))
            else:
                unsupported.append(resolved_token.token)
                
        return SignSequenceResult(
            sequence=sequence,
            unsupported_tokens=unsupported
        )
