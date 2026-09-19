"""Text-to-Sign orchestration service.

Orchestrates the Text-to-Sign pipeline:
TextProcessingService -> SignResolutionService -> SignSequenceService
"""

import logging
from backend.services.text_processing_service import TextProcessingService
from backend.services.sign_resolution_service import SignResolutionService
from backend.services.sign_sequence_service import SignSequenceService
from backend.services.text_sign_session_service import TextSignSessionService
from backend.schemas.text_sign_session import TextSignSessionState, InvalidStateTransitionError
from backend.schemas.text_sign import TextSignResponse
from backend.schemas.text_processing import TextProcessingError, EmptyTextError

logger = logging.getLogger(__name__)

class TextSignOrchestratorService:
    """Orchestrates the Text-to-Sign workflow."""

    def __init__(
        self,
        session_service: TextSignSessionService,
        text_processing_service: TextProcessingService,
        resolution_service: SignResolutionService,
        sequence_service: SignSequenceService
    ):
        self.session_service = session_service
        self.text_processing_service = text_processing_service
        self.resolution_service = resolution_service
        self.sequence_service = sequence_service

    def process_text(self, session_id: str, text: str) -> TextSignResponse:
        """Process text through the pipeline for a given session."""
        
        # 1. Transition to PROCESSING
        try:
            self.session_service.transition_state(session_id, TextSignSessionState.PROCESSING)
        except InvalidStateTransitionError as e:
            # Re-raise to be handled by the route
            raise

        try:
            # 2. Text Processing
            text_result = self.text_processing_service.process(text)
            
            # 3. Sign Resolution
            resolution_result = self.resolution_service.resolve(text_result)
            
            # 4. Sign Sequence Construction
            sequence_result = self.sequence_service.build_sequence(resolution_result)
            
            # 5. Transition to COMPLETED
            session = self.session_service.transition_state(session_id, TextSignSessionState.COMPLETED)
            
            # 6. Return structured result
            return TextSignResponse(
                session_id=session_id,
                state=session.state,
                original_text=text_result.original_text,
                normalized_text=text_result.normalized_text,
                sequence=sequence_result.sequence,
                unsupported_tokens=sequence_result.unsupported_tokens
            )
            
        except EmptyTextError as e:
            # Empty text error is a client error, usually returns 400.
            # Transition to ERROR state.
            self.session_service.transition_state(session_id, TextSignSessionState.ERROR)
            raise
        except Exception as e:
            # Unexpected error
            logger.exception("Unexpected error during text-to-sign processing")
            self.session_service.transition_state(session_id, TextSignSessionState.ERROR)
            raise
