using UnityEngine;

namespace Avatar.Animation
{
    public struct ResolvedAnimationAsset
    {
        public AnimationClip clip;
        public AnimationRegistryEntry metadata;
    }

    public class AnimationAssetResolver : MonoBehaviour
    {
        [SerializeField] private AnimationRegistry _registry;
        
        // Resolves an asset_id to an AnimationClip and its metadata
        public ResolvedAnimationAsset? Resolve(string assetId)
        {
            if (_registry == null)
            {
                Debug.LogError("[AnimationAssetResolver] Registry is not assigned.");
                return null;
            }

            AnimationRegistryEntry entry = _registry.GetEntry(assetId);
            if (entry == null)
            {
                Debug.LogError($"[AnimationAssetResolver] Resolution failure: asset_id '{assetId}' not found in registry.");
                return null;
            }

            // Attempt to load the clip using the configured clip_name
            // We assume clips are placed in a Resources folder for runtime dynamic loading
            string resourcePath = $"Animations/ISL/{entry.clip_name}";
            AnimationClip clip = Resources.Load<AnimationClip>(resourcePath);

            if (clip == null)
            {
                Debug.LogError($"[AnimationAssetResolver] Resolution failure: AnimationClip '{entry.clip_name}' missing at '{resourcePath}'.");
                return null;
            }

            return new ResolvedAnimationAsset { clip = clip, metadata = entry };
        }
    }
}
