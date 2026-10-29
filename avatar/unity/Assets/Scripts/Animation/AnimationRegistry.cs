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
        private readonly Dictionary<string, AnimationRegistryEntry> _assetIdMap = new Dictionary<string, AnimationRegistryEntry>(System.StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, List<AnimationRegistryEntry>> _signIdMap = new Dictionary<string, List<AnimationRegistryEntry>>(System.StringComparer.OrdinalIgnoreCase);

        public bool LoadRegistry(string jsonContent)
        {
            _assetIdMap.Clear();
            _signIdMap.Clear();
            if (string.IsNullOrWhiteSpace(jsonContent))
            {
                Debug.LogWarning("[AnimationRegistry] Registry JSON content is empty or null.");
                return false;
            }

            AnimationRegistryData data;
            try
            {
                data = JsonUtility.FromJson<AnimationRegistryData>(jsonContent);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[AnimationRegistry] Failed to parse JSON: {ex.Message}");
                return false;
            }

            if (data == null || data.animations == null)
            {
                Debug.LogError("[AnimationRegistry] Parsed animation registry data is null.");
                return false;
            }

            foreach (var entry in data.animations)
            {
                if (entry == null) continue;

                if (!string.IsNullOrEmpty(entry.asset_id))
                {
                    if (_assetIdMap.ContainsKey(entry.asset_id))
                    {
                        Debug.LogWarning($"[AnimationRegistry] Duplicate asset_id '{entry.asset_id}' encountered. Overwriting previous entry.");
                    }
                    _assetIdMap[entry.asset_id] = entry;
                }

                if (!string.IsNullOrEmpty(entry.sign_id))
                {
                    if (!_signIdMap.TryGetValue(entry.sign_id, out var list))
                    {
                        list = new List<AnimationRegistryEntry>();
                        _signIdMap[entry.sign_id] = list;
                    }
                    list.Add(entry);
                }
            }

            return true;
        }

        public AnimationRegistryEntry GetEntry(string identifier)
        {
            if (string.IsNullOrEmpty(identifier)) return null;

            bool hasAssetMatch = _assetIdMap.TryGetValue(identifier, out var assetEntry);
            bool hasSignMatch = _signIdMap.TryGetValue(identifier, out var signList);

            if (hasAssetMatch && hasSignMatch && signList != null && signList.Count > 0)
            {
                if (signList.Count > 1 || signList[0] != assetEntry)
                {
                    Debug.LogError($"[AnimationRegistry] Conflicting identifier '{identifier}': matches concrete asset_id of '{assetEntry.asset_id}' and semantic sign_id of '{signList[0].asset_id}'.");
                    return null;
                }
            }

            if (hasAssetMatch)
            {
                return assetEntry;
            }

            if (hasSignMatch && signList != null)
            {
                if (signList.Count == 1)
                {
                    return signList[0];
                }

                Debug.LogError($"[AnimationRegistry] Ambiguous identifier '{identifier}': matches {signList.Count} entries with variant differences. Specify a concrete asset_id instead.");
                return null;
            }

            return null;
        }

        public bool IsRegistered(string identifier)
        {
            if (string.IsNullOrEmpty(identifier)) return false;

            return GetEntry(identifier) != null;
        }
    }
}
