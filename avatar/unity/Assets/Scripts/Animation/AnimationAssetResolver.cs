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

        private bool _testRegistryInitialized = false;
        private bool _testRegistryLoadSuccess = false;

        private void Awake()
        {
            if (_useTestResolution)
            {
                EnsureTestRegistryInitialized();
            }
        }

        public void EnsureTestRegistryInitialized()
        {
            if (_testRegistryInitialized) return;
            _testRegistryInitialized = true;
            _testRegistryLoadSuccess = false;

            if (string.IsNullOrWhiteSpace(_testAssetFolder))
            {
                Debug.LogError("[AnimationAssetResolver] Test asset folder is empty or not specified.");
                return;
            }

            string registryAssetPath = $"{_testAssetFolder}/test_animation_registry.json";
            TextAsset registryAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>(registryAssetPath);
            if (registryAsset == null)
            {
                Debug.LogError($"[AnimationAssetResolver] Failed to load test animation registry asset at '{registryAssetPath}'. Verify that the file exists and is imported.");
                return;
            }

            string jsonContent = registryAsset.text;
            if (string.IsNullOrWhiteSpace(jsonContent))
            {
                Debug.LogError($"[AnimationAssetResolver] Test animation registry at '{registryAssetPath}' contains empty content.");
                return;
            }

            if (_testRegistry == null)
            {
                AnimationRegistry[] registries = GetComponents<AnimationRegistry>();
                foreach (var reg in registries)
                {
                    if (reg != _registry)
                    {
                        _testRegistry = reg;
                        break;
                    }
                }

                if (_testRegistry == null)
                {
                    _testRegistry = gameObject.AddComponent<AnimationRegistry>();
                    _testRegistry.hideFlags = HideFlags.DontSaveInEditor | HideFlags.DontSaveInBuild;
                }
            }

            bool loaded = _testRegistry.LoadRegistry(jsonContent);
            if (!loaded)
            {
                Debug.LogError($"[AnimationAssetResolver] Failed to parse test animation registry JSON at '{registryAssetPath}'.");
                return;
            }

            _testRegistryLoadSuccess = true;
            Debug.Log($"[AnimationAssetResolver] Successfully initialized test registry from '{registryAssetPath}'.");
        }
#endif
        
        // Resolves an asset_id to an AnimationClip and its metadata
        public ResolvedAnimationAsset? Resolve(string assetId)
        {
#if UNITY_EDITOR
            if (_useTestResolution)
            {
                EnsureTestRegistryInitialized();

                // Fail closed: do NOT silently fall back to production registry
                if (!_testRegistryLoadSuccess || _testRegistry == null)
                {
                    Debug.LogError("[AnimationAssetResolver] Test resolution is enabled, but the test registry is not available. Playback aborted (failing closed).");
                    return null;
                }

                if (!_testRegistry.IsRegistered(assetId))
                {
                    Debug.LogError($"[AnimationAssetResolver] Resolution failure: asset_id '{assetId}' not found in test registry.");
                    return null;
                }

                AnimationRegistryEntry testEntry = _testRegistry.GetEntry(assetId);
                if (testEntry == null)
                {
                    Debug.LogError($"[AnimationAssetResolver] Resolution failure: could not resolve entry for '{assetId}' in test registry.");
                    return null;
                }

                string testPath = $"{_testAssetFolder}/{testEntry.clip_name}.anim";
                AnimationClip testClip = UnityEditor.AssetDatabase.LoadAssetAtPath<AnimationClip>(testPath);
                if (testClip == null)
                {
                    Debug.LogError($"[AnimationAssetResolver] Resolution failure: AnimationClip '{testEntry.clip_name}' missing at '{testPath}'.");
                    return null;
                }

                return new ResolvedAnimationAsset { clip = testClip, metadata = testEntry };
            }
#endif

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
                Debug.LogError($"[AnimationAssetResolver] Resolution failure: AnimationClip '{entry.clip_name}' missing.");
                return null;
            }

            return new ResolvedAnimationAsset { clip = clip, metadata = entry };
        }
    }
}
