import pytest
from backend.services.text_sign_session_service import TextSignSessionService
from backend.services.text_processing_service import TextProcessingService
from backend.services.sign_resolution_service import SignResolutionService
from backend.services.sign_sequence_service import SignSequenceService
from backend.services.sign_dictionary_service import SignDictionaryService
from backend.services.text_sign_orchestrator_service import TextSignOrchestratorService
from backend.schemas.text_sign_session import TextSignSessionState
from backend.schemas.text_processing import EmptyTextError

@pytest.fixture
def orchestrator():
    session_service = TextSignSessionService()
    text_processing_service = TextProcessingService()
    dictionary_service = SignDictionaryService()
    resolution_service = SignResolutionService(dictionary_service)
    sequence_service = SignSequenceService()
    
    return TextSignOrchestratorService(
        session_service=session_service,
        text_processing_service=text_processing_service,
        resolution_service=resolution_service,
        sequence_service=sequence_service
    )

def test_process_text_success(orchestrator):
    # Setup session
    session = orchestrator.session_service.create_session()
    session_id = session.session_id
    
    response = orchestrator.process_text(session_id, "hello thank you")
    
    assert response.session_id == session_id
    assert response.state == TextSignSessionState.COMPLETED
    assert response.original_text == "hello thank you"
    assert response.normalized_text == "hello thank you"
    
    # Depending on dictionary, "hello" and "thank you" should resolve
    assert len(response.sequence) > 0
    assert response.sequence[0].source_text == "hello"

def test_process_empty_text(orchestrator):
    session = orchestrator.session_service.create_session()
    
    with pytest.raises(EmptyTextError):
        orchestrator.process_text(session.session_id, "   ")
        
    session = orchestrator.session_service.get_session(session.session_id)
    assert session.state == TextSignSessionState.ERROR

def test_process_unsupported_tokens(orchestrator):
    session = orchestrator.session_service.create_session()
    
    response = orchestrator.process_text(session.session_id, "hello unknownword yes")
    
    assert response.state == TextSignSessionState.COMPLETED
    assert "unknownword" in response.unsupported_tokens
    
    resolved_texts = [item.source_text for item in response.sequence]
    assert "hello" in resolved_texts
    assert "yes" in resolved_texts
