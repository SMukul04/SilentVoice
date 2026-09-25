import pytest
import ast
from pathlib import Path
from backend.schemas.sign_sequence import SignSequenceResult, SignSequenceItem
from backend.schemas.avatar_animation import AnimationAsset, AnimationAvailability
from backend.services.avatar_animation_service import MockAnimationAssetService
from backend.services.avatar_runtime_service import MockAvatarRuntimeService, AvatarRuntimeState, AvatarRuntimeError
from backend.services.avatar_playback_service import SequentialPlaybackService, PlaybackState, AvatarPlaybackError
from backend.services.avatar_controller_service import AvatarControllerService, PlaybackAlreadyActiveError, InvalidPlaybackControlError

@pytest.fixture
def runtime_service():
    runtime = MockAvatarRuntimeService()
    runtime.initialize()
    return runtime

@pytest.fixture
def animation_service():
    return MockAnimationAssetService()

@pytest.fixture
def playback_service(runtime_service):
    return SequentialPlaybackService(runtime_service)

@pytest.fixture
def controller(animation_service, playback_service):
    return AvatarControllerService(animation_service, playback_service)

def test_avatar_validation_happy_path(controller, animation_service, runtime_service, playback_service):
    # Setup mock assets
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="ty_01"))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01"))

    # Construct sequence
    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="THANK_YOU", source_text="thank you", original_index=1),
            SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=2),
        ],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(sequence)
    
    assert controller.get_state() == PlaybackState.PLAYING
    assert runtime_service.get_current_animation().sign_id == "HELLO"

    playback_service.on_animation_complete()
    assert runtime_service.get_current_animation().sign_id == "THANK_YOU"

    playback_service.on_animation_complete()
    assert runtime_service.get_current_animation().sign_id == "PLEASE"

    playback_service.on_animation_complete()
    assert controller.get_state() == PlaybackState.COMPLETED
    
    history = runtime_service.get_history()
    assert len(history) == 3
    assert [a.sign_id for a in history] == ["HELLO", "THANK_YOU", "PLEASE"]


def test_avatar_validation_duplicate_signs(controller, animation_service, runtime_service, playback_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="ty_01"))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="HELLO", source_text="hello again", original_index=1),
            SignSequenceItem(sign_id="THANK_YOU", source_text="thank you", original_index=2),
        ],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(sequence)
    playback_service.on_animation_complete()
    playback_service.on_animation_complete()
    playback_service.on_animation_complete()

    history = runtime_service.get_history()
    assert len(history) == 3
    assert [a.sign_id for a in history] == ["HELLO", "HELLO", "THANK_YOU"]


def test_avatar_validation_unsupported_tokens(controller, animation_service, runtime_service, playback_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01"))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=2),
        ],
        unsupported_tokens=["unknown_word"]
    )

    result = controller.play_sign_sequence(sequence)
    assert result.unsupported_semantic_tokens == ["unknown_word"]
    
    playback_service.on_animation_complete()
    playback_service.on_animation_complete()

    history = runtime_service.get_history()
    assert [a.sign_id for a in history] == ["HELLO", "PLEASE"]


def test_avatar_validation_unavailable_animation(controller, animation_service, runtime_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01", availability=AnimationAvailability.AVAILABLE))
    animation_service.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="ty_01", availability=AnimationAvailability.UNAVAILABLE))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01", availability=AnimationAvailability.AVAILABLE))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="THANK_YOU", source_text="thank you", original_index=1),
            SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=2),
        ],
        unsupported_tokens=[]
    )

    with pytest.raises(AvatarPlaybackError):
        controller.play_sign_sequence(sequence)

    assert len(runtime_service.get_history()) == 0


class FailingRuntimeService(MockAvatarRuntimeService):
    def play_animation(self, asset: AnimationAsset) -> None:
        if asset.sign_id == "FAIL":
            raise AvatarRuntimeError("Intentional failure")
        super().play_animation(asset)


def test_avatar_validation_runtime_failure(animation_service):
    runtime = FailingRuntimeService()
    runtime.initialize()
    playback = SequentialPlaybackService(runtime)
    controller = AvatarControllerService(animation_service, playback)

    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="FAIL", asset_id="fail_01"))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01"))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="FAIL", source_text="fail", original_index=1),
            SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=2),
        ],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(sequence)
    assert controller.get_state() == PlaybackState.PLAYING
    assert runtime.get_current_animation().sign_id == "HELLO"

    with pytest.raises(AvatarRuntimeError):
        playback.on_animation_complete()

    assert controller.get_state() == PlaybackState.ERROR
    assert len(runtime.get_history()) == 1


def test_avatar_validation_stop(controller, animation_service, runtime_service, playback_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="ty_01"))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01"))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="THANK_YOU", source_text="thank you", original_index=1),
            SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=2),
        ],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(sequence)
    assert controller.get_state() == PlaybackState.PLAYING

    controller.stop()
    assert controller.get_state() == PlaybackState.STOPPED
    assert runtime_service.get_state() == AvatarRuntimeState.STOPPED
    
    playback_service.on_animation_complete()
    history = runtime_service.get_history()
    assert [a.sign_id for a in history] == ["HELLO"]


def test_avatar_validation_reset(controller, animation_service, runtime_service, playback_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    sequence = SignSequenceResult(
        sequence=[SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0)],
        unsupported_tokens=[]
    )

    # 1. From PLAYING
    controller.play_sign_sequence(sequence)
    controller.reset()
    assert controller.get_state() == PlaybackState.IDLE
    assert runtime_service.get_state() == AvatarRuntimeState.UNINITIALIZED

    # 2. From COMPLETED
    runtime_service.initialize()
    controller.play_sign_sequence(sequence)
    playback_service.on_animation_complete()
    assert controller.get_state() == PlaybackState.COMPLETED
    controller.reset()
    assert controller.get_state() == PlaybackState.IDLE

    # 3. From ERROR
    class ErrorRuntime(MockAvatarRuntimeService):
        def play_animation(self, asset):
            raise AvatarRuntimeError("fail")
            
    err_runtime = ErrorRuntime()
    err_runtime.initialize()
    err_playback = SequentialPlaybackService(err_runtime)
    err_controller = AvatarControllerService(animation_service, err_playback)
    
    with pytest.raises(AvatarRuntimeError):
        err_controller.play_sign_sequence(sequence)
        
    assert err_controller.get_state() == PlaybackState.ERROR
    err_controller.reset()
    assert err_controller.get_state() == PlaybackState.IDLE
    err_runtime.initialize() # prepare for next play
    # Should be able to play again
    with pytest.raises(AvatarRuntimeError):
        err_controller.play_sign_sequence(sequence)


def test_avatar_validation_repeated_play(controller, animation_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="PLEASE", asset_id="pl_01"))

    seq1 = SignSequenceResult(
        sequence=[SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0)],
        unsupported_tokens=[]
    )
    seq2 = SignSequenceResult(
        sequence=[SignSequenceItem(sign_id="PLEASE", source_text="please", original_index=0)],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(seq1)
    
    with pytest.raises(PlaybackAlreadyActiveError):
        controller.play_sign_sequence(seq2)


def test_avatar_validation_progress(controller, animation_service, playback_service):
    animation_service.register_asset(AnimationAsset(sign_id="HELLO", asset_id="hello_01"))
    animation_service.register_asset(AnimationAsset(sign_id="THANK_YOU", asset_id="ty_01"))

    sequence = SignSequenceResult(
        sequence=[
            SignSequenceItem(sign_id="HELLO", source_text="hello", original_index=0),
            SignSequenceItem(sign_id="THANK_YOU", source_text="thank you", original_index=1),
        ],
        unsupported_tokens=[]
    )

    controller.play_sign_sequence(sequence)
    prog1 = controller.get_progress()
    assert prog1.state == PlaybackState.PLAYING
    assert prog1.current_index == 0
    assert prog1.total_assets == 2
    assert prog1.current_asset.sign_id == "HELLO"

    playback_service.on_animation_complete()
    prog2 = controller.get_progress()
    assert prog2.current_index == 1
    assert prog2.current_asset.sign_id == "THANK_YOU"

    playback_service.on_animation_complete()
    prog3 = controller.get_progress()
    assert prog3.state == PlaybackState.COMPLETED


def test_avatar_validation_empty_sequence(controller, runtime_service):
    sequence = SignSequenceResult(sequence=[], unsupported_tokens=[])
    controller.play_sign_sequence(sequence)
    assert controller.get_state() == PlaybackState.COMPLETED
    assert len(runtime_service.get_history()) == 0


def test_avatar_validation_architectural_dependencies():
    # Helper to check imports in a file
    def check_file_for_imports(filepath, forbidden_modules):
        try:
            tree = ast.parse(Path(filepath).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        for forbidden in forbidden_modules:
                            if forbidden in name.name:
                                raise Exception(f"Forbidden import {name.name} found in {filepath}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for forbidden in forbidden_modules:
                            if forbidden in node.module:
                                raise Exception(f"Forbidden import {node.module} found in {filepath}")
        except FileNotFoundError:
            pass # ignore missing file

    # 1. Runtime must NOT import: AvatarControllerService, SequentialPlaybackService, SignSequenceService, TextSignOrchestratorService
    runtime_forbidden = ["controller", "playback", "sign_sequence", "text_sign_orchestrator"]
    check_file_for_imports("backend/services/avatar_runtime_service.py", runtime_forbidden)

    # 2. Playback must NOT import: TextSignOrchestratorService, frontend, Unity
    playback_forbidden = ["text_sign_orchestrator", "frontend", "Unity", "unity"]
    check_file_for_imports("backend/services/avatar_playback_service.py", playback_forbidden)

    # 3. Animation service must NOT import: Unity, runtime, playback, frontend
    animation_forbidden = ["unity", "Unity", "runtime", "playback", "frontend"]
    check_file_for_imports("backend/services/avatar_animation_service.py", animation_forbidden)

    # 4. Controller must NOT import: Unity, WebGL, frontend, MediaPipe, speech
    controller_forbidden = ["unity", "Unity", "webgl", "frontend", "mediapipe", "speech"]
    check_file_for_imports("backend/services/avatar_controller_service.py", controller_forbidden)


def test_avatar_validation_no_hardcoded_semantic_mappings(controller, animation_service, runtime_service, playback_service):
    # Verify completely arbitrary signs pass through untouched
    animation_service.register_asset(AnimationAsset(sign_id="RANDOM_XYZ", asset_id="xyz_01"))
    sequence = SignSequenceResult(
        sequence=[SignSequenceItem(sign_id="RANDOM_XYZ", source_text="xyz", original_index=0)],
        unsupported_tokens=[]
    )
    
    controller.play_sign_sequence(sequence)
    assert runtime_service.get_current_animation().sign_id == "RANDOM_XYZ"
