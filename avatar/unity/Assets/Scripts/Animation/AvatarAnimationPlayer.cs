using UnityEngine;
using UnityEngine.Events;

namespace Avatar.Animation 
{
    public class AvatarAnimationPlayer : MonoBehaviour 
    {
        [SerializeField] private AnimationAssetResolver _resolver;
        [SerializeField] private Animator _animator;
        
        public UnityEvent OnPlaybackComplete;
        
        private AnimatorOverrideController _overrideController;
        private bool _isLooping = false;
        private string _currentAssetId;

        private void Awake()
        {
            if (OnPlaybackComplete == null) 
            {
                OnPlaybackComplete = new UnityEvent();
            }

            if (_animator != null && _animator.runtimeAnimatorController != null)
            {
                // Prepare a data-driven override controller
                _overrideController = new AnimatorOverrideController(_animator.runtimeAnimatorController);
                _animator.runtimeAnimatorController = _overrideController;
            }
        }

        public bool Play(string assetId) 
        {
            Debug.Log($"[AvatarAnimationPlayer] Playing asset: {assetId}");
            
            if (_resolver == null)
            {
                Debug.LogError("[AvatarAnimationPlayer] Resolver is missing.");
                return false;
            }

            var resolvedAsset = _resolver.Resolve(assetId);
            if (!resolvedAsset.HasValue)
            {
                Debug.LogError($"[AvatarAnimationPlayer] Failed to resolve clip for asset: {assetId}. Playback aborted.");
                return false;
            }

            AnimationClip clip = resolvedAsset.Value.clip;
            AnimationRegistryEntry metadata = resolvedAsset.Value.metadata;
            
            _isLooping = metadata.loop;
            _currentAssetId = assetId;

            if (_animator != null && _overrideController != null)
            {
                // Override the generic "ISL_Placeholder" clip with the resolved specific ISL clip
                _overrideController["ISL_Placeholder"] = clip;
                
                // Apply speed metadata
                _animator.speed = metadata.speed > 0f ? metadata.speed : 1.0f;
                
                // Trigger the state that uses this clip
                _animator.Play("ISL_Playback", 0, 0f);
                
                return true;
            }
            else
            {
                Debug.LogWarning("[AvatarAnimationPlayer] Animator or OverrideController is missing. Cannot play actual animation.");
                return false;
            }
        }

        public void Stop() 
        {
            Debug.Log($"[AvatarAnimationPlayer] Stopping playback.");
            if (_animator != null)
            {
                _animator.speed = 0f;
                _animator.Rebind(); // Reset to default pose instead of hardcoded Idle state
            }
            _currentAssetId = null;
        }
        
        // This method represents the real completion boundary.
        // It must be called by a Unity Animation Event on the last frame of the ISL clip,
        // or by a StateMachineBehaviour. It is not driven by a fake timer.
        public void OnAnimationComplete()
        {
            if (!_isLooping)
            {
                Debug.Log($"[AvatarAnimationPlayer] Animation clip reached completion boundary for {_currentAssetId}.");
                OnPlaybackComplete?.Invoke();
            }
        }
    }
}
