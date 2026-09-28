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
        
#if UNITY_EDITOR
        [Header("Editor Test Resolution")]
        [SerializeField] private bool _useTestResolution = false;
        [SerializeField] private AnimationRegistry _testRegistry;
        [SerializeField] private string _testAssetFolder = "Assets/Animations/ISL_Test";
#endif
        
        // Resolves an asset_id to an AnimationClip and its metadata
        public ResolvedAnimationAsset? Resolve(string assetId)
        {
            AnimationRegistry targetRegistry = _registry;

#if UNITY_EDITOR
            if (_useTestResolution && _testRegistry != null && _testRegistry.IsRegistered(assetId))
            {
                targetRegistry = _testRegistry;
            }
#endif

            if (targetRegistry == null)
            {
                Debug.LogError("[AnimationAssetResolver] Registry is not assigned.");
                return null;
            }

            AnimationRegistryEntry entry = targetRegistry.GetEntry(assetId);
            if (entry == null)
            {
                Debug.LogError($"[AnimationAssetResolver] Resolution failure: asset_id '{assetId}' not found in registry.");
                return null;
            }

            AnimationClip clip = null;

#if UNITY_EDITOR
            if (_useTestResolution && targetRegistry == _testRegistry)
            {
                string testPath = $"{_testAssetFolder}/{entry.clip_name}.anim";
                clip = UnityEditor.AssetDatabase.LoadAssetAtPath<AnimationClip>(testPath);
            }
            else
#endif
            {
                // Attempt to load the clip using the configured clip_name
                // We assume clips are placed in a Resources folder for runtime dynamic loading
                string resourcePath = $"Animations/ISL/{entry.clip_name}";
                clip = Resources.Load<AnimationClip>(resourcePath);
            }

            if (clip == null)
            {
                Debug.LogError($"[AnimationAssetResolver] Resolution failure: AnimationClip '{entry.clip_name}' missing.");
                return null;
            }

            return new ResolvedAnimationAsset { clip = clip, metadata = entry };
        }
    }
}
