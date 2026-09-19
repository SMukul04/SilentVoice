import pytest
from backend.schemas.text_sign_session import TextSignSessionState, SessionNotFoundError, InvalidStateTransitionError
from backend.services.text_sign_session_service import TextSignSessionService

@pytest.fixture
def session_service():
    return TextSignSessionService()

def test_create_session(session_service):
    session = session_service.create_session()
    assert session.session_id is not None
    assert session.state == TextSignSessionState.IDLE

def test_get_session(session_service):
    session = session_service.create_session()
    retrieved = session_service.get_session(session.session_id)
    assert retrieved.session_id == session.session_id
    assert retrieved.state == session.state

def test_get_missing_session(session_service):
    with pytest.raises(SessionNotFoundError):
        session_service.get_session("invalid_id")

def test_valid_state_transitions(session_service):
    session = session_service.create_session()
    session_id = session.session_id
    
    # IDLE -> PROCESSING
    session = session_service.transition_state(session_id, TextSignSessionState.PROCESSING)
    assert session.state == TextSignSessionState.PROCESSING
    
    # PROCESSING -> COMPLETED
    session = session_service.transition_state(session_id, TextSignSessionState.COMPLETED)
    assert session.state == TextSignSessionState.COMPLETED
    
    # COMPLETED -> IDLE
    session = session_service.transition_state(session_id, TextSignSessionState.IDLE)
    assert session.state == TextSignSessionState.IDLE
    
    # IDLE -> CLOSED
    session = session_service.transition_state(session_id, TextSignSessionState.CLOSED)
    assert session.state == TextSignSessionState.CLOSED

def test_invalid_state_transition(session_service):
    session = session_service.create_session()
    
    # IDLE -> COMPLETED is invalid
    with pytest.raises(InvalidStateTransitionError):
        session_service.transition_state(session.session_id, TextSignSessionState.COMPLETED)

def test_reset_session(session_service):
    session = session_service.create_session()
    session_service.transition_state(session.session_id, TextSignSessionState.PROCESSING)
    session_service.transition_state(session.session_id, TextSignSessionState.ERROR)
    
    reset_session = session_service.reset_session(session.session_id)
    assert reset_session.state == TextSignSessionState.IDLE

def test_reset_closed_session(session_service):
    session = session_service.create_session()
    session_service.close_session(session.session_id)
    
    with pytest.raises(InvalidStateTransitionError):
        session_service.reset_session(session.session_id)

def test_close_session(session_service):
    session = session_service.create_session()
    closed_session = session_service.close_session(session.session_id)
    assert closed_session.state == TextSignSessionState.CLOSED
    
    # Closing an already closed session should not raise
    session_service.close_session(session.session_id)
