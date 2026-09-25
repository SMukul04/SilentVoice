using UnityEngine;

namespace Avatar.Animation 
{
    public class AvatarAnimationPlayer : MonoBehaviour 
    {
        public void Play(string assetId) 
        {
            // Abstraction boundary for Unity animation playback
            // (e.g. Animator.Play, Timeline execution, etc.)
            Debug.Log($"[AvatarAnimationPlayer] Playing abstract asset: {assetId}");
            
            // Do NOT simulate timer. Completion must be invoked explicitly 
            // by Animation events in real implementation.
        }

        public void Stop() 
        {
            Debug.Log($"[AvatarAnimationPlayer] Stopping playback.");
        }
    }
}
