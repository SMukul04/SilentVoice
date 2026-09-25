using System.Collections.Generic;
using UnityEngine;

namespace Avatar.Animation
{
    [System.Serializable]
    public class AnimationRegistryEntry
    {
        public string sign_id;
        public string asset_id;
        public string variant;
        public string clip_name;
        public bool loop;
        public float speed;
    }

    [System.Serializable]
    public class AnimationRegistryData
    {
        public List<AnimationRegistryEntry> animations = new List<AnimationRegistryEntry>();
    }

    public class AnimationRegistry : MonoBehaviour
    {
        private Dictionary<string, AnimationRegistryEntry> _registryMap = new Dictionary<string, AnimationRegistryEntry>();

        public void LoadRegistry(string jsonContent)
        {
            _registryMap.Clear();
            if (string.IsNullOrEmpty(jsonContent)) return;

            AnimationRegistryData data = JsonUtility.FromJson<AnimationRegistryData>(jsonContent);
            if (data != null && data.animations != null)
            {
                foreach (var entry in data.animations)
                {
                    _registryMap[entry.asset_id] = entry;
                }
            }
        }

        public AnimationRegistryEntry GetEntry(string assetId)
        {
            if (_registryMap.TryGetValue(assetId, out var entry))
            {
                return entry;
            }
            return null;
        }

        public bool IsRegistered(string assetId)
        {
            return _registryMap.ContainsKey(assetId);
        }
    }
}
