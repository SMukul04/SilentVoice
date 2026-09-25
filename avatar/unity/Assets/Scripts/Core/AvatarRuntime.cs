using UnityEngine;
using Avatar.Animation;

namespace Avatar.Core
{
    public enum AvatarRuntimeState
    {
        UNINITIALIZED,
        READY,
        PLAYING,
        STOPPED,
        ERROR
    }

    public class AvatarRuntime : MonoBehaviour
    {
        private AvatarRuntimeState _state = AvatarRuntimeState.UNINITIALIZED;
        private AvatarAnimationPlayer _animationPlayer;
        private string _currentAnimation;

        private void Awake()
        {
            _animationPlayer = GetComponentInChildren<AvatarAnimationPlayer>();
            if (_animationPlayer == null)
            {
                Debug.LogError("AvatarAnimationPlayer not found in children.");
                _state = AvatarRuntimeState.ERROR;
            }
            else
            {
                // Code-level wiring for the completion boundary
                _animationPlayer.OnPlaybackComplete.AddListener(OnAnimationComplete);
            }
        }

        public void Initialize()
        {
            if (_state == AvatarRuntimeState.ERROR) return;
            _state = AvatarRuntimeState.READY;
            _currentAnimation = null;
        }

        public void PlayAnimation(string assetId)
        {
            if (_state == AvatarRuntimeState.UNINITIALIZED || _state == AvatarRuntimeState.ERROR)
            {
                Debug.LogWarning($"Cannot play animation. Current state: {_state}");
                return;
            }

            // Verify playback actually succeeds before committing state
            bool success = _animationPlayer.Play(assetId);
            
            if (success)
            {
                _state = AvatarRuntimeState.PLAYING;
                _currentAnimation = assetId;
            }
            else
            {
                Debug.LogError($"[AvatarRuntime] Failed to start playback for {assetId}. Transitioning to ERROR state.");
                _state = AvatarRuntimeState.ERROR;
                _currentAnimation = null;
            }
        }

        public void Stop()
        {
            if (_state != AvatarRuntimeState.UNINITIALIZED)
            {
                _state = AvatarRuntimeState.STOPPED;
                _currentAnimation = null;
                _animationPlayer.Stop();
            }
        }

        public void ResetRuntime()
        {
            _state = AvatarRuntimeState.UNINITIALIZED;
            _currentAnimation = null;
            _animationPlayer.Stop();
        }

        public AvatarRuntimeState GetState()
        {
            return _state;
        }

        public string GetCurrentAnimation()
        {
            return _currentAnimation;
        }

        public void OnAnimationComplete()
        {
            if (_state == AvatarRuntimeState.PLAYING)
            {
                // To be wired to SequentialPlaybackService logic in controller
                Debug.Log($"Animation {_currentAnimation} complete.");
            }
        }
    }
}
