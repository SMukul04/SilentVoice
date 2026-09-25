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

            _state = AvatarRuntimeState.PLAYING;
            _currentAnimation = assetId;
            _animationPlayer.Play(assetId);
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
