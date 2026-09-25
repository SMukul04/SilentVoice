using UnityEngine;

namespace Avatar.Animation 
{
    public class AvatarAnimationPlayer : MonoBehaviour 
    {
        [SerializeField] private AnimationAssetResolver _resolver;
        [SerializeField] private Animator _animator;
        
        private AnimatorOverrideController _overrideController;

        private void Awake()
        {
            if (_animator != null && _animator.runtimeAnimatorController != null)
            {
                // Prepare a data-driven override controller
                _overrideController = new AnimatorOverrideController(_animator.runtimeAnimatorController);
                _animator.runtimeAnimatorController = _overrideController;
            }
        }

        public void Play(string assetId) 
        {
            Debug.Log($"[AvatarAnimationPlayer] Playing abstract asset: {assetId}");
            
            if (_resolver == null)
            {
                Debug.LogError("[AvatarAnimationPlayer] Resolver is missing.");
                return;
            }

            AnimationClip clip = _resolver.Resolve(assetId);
            if (clip == null)
            {
                Debug.LogError($"[AvatarAnimationPlayer] Failed to resolve clip for asset: {assetId}. Playback aborted.");
                return;
            }

            if (_animator != null && _overrideController != null)
            {
                // Override the generic "ISL_Placeholder" clip with the resolved specific ISL clip
                _overrideController["ISL_Placeholder"] = clip;
                
                // Trigger the state that uses this clip
                _animator.Play("ISL_Playback", 0, 0f);
            }
            else
            {
                Debug.LogWarning("[AvatarAnimationPlayer] Animator or OverrideController is missing. Cannot play actual animation.");
            }
            
            // Completion must be invoked explicitly by Animation events later.
        }

        public void Stop() 
        {
            Debug.Log($"[AvatarAnimationPlayer] Stopping playback.");
            if (_animator != null)
            {
                _animator.Play("Idle"); // Return to neutral state
            }
        }
    }
}
