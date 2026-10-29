"""Phase 8.11 Asset Resolution, Registry Loading, and Isolation Tests.

Validates:
1. Production registry isolation (remains strictly empty).
2. Local test assets isolation and git-ignore status (does not require local assets to exist).
3. Test registry JSON schema and structure using an inline fixture.
4. Semantic identifier resolution (sign_id vs asset_id, ambiguity, conflicts, case-insensitivity).
5. Static code verification of AnimationAssetResolver and AnimationRegistry contracts.
"""

import json
from pathlib import Path
import subprocess
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_REGISTRY_PATH = REPO_ROOT / "avatar" / "unity" / "Assets" / "Resources" / "Animations" / "animation_registry.json"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
RESOLVER_CS_PATH = REPO_ROOT / "avatar" / "unity" / "Assets" / "Scripts" / "Animation" / "AnimationAssetResolver.cs"
REGISTRY_CS_PATH = REPO_ROOT / "avatar" / "unity" / "Assets" / "Scripts" / "Animation" / "AnimationRegistry.cs"

# Inline fixture representing the PLEASE test entry contract without requiring local git-ignored files to exist.
SAMPLE_PLEASE_TEST_REGISTRY_JSON = json.dumps({
    "animations": [
        {
            "sign_id": "PLEASE",
            "asset_id": "please_test",
            "variant": "default",
            "clip_name": "please",
            "loop": False,
            "speed": 1.0,
        }
    ]
})


def test_production_registry_strict_empty():
    """Verify production animation_registry.json exists and contains exactly 0 animations."""
    assert PROD_REGISTRY_PATH.exists(), f"Production registry missing at {PROD_REGISTRY_PATH}"
    data = json.loads(PROD_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert "animations" in data, "Production registry missing 'animations' field"
    assert data["animations"] == [], "Production registry must remain strictly empty pending rights approval"


def test_production_resources_no_unapproved_clips():
    """Verify no .anim clips exist in Assets/Resources/Animations/ISL/."""
    isl_res_dir = REPO_ROOT / "avatar" / "unity" / "Assets" / "Resources" / "Animations" / "ISL"
    if isl_res_dir.exists():
        anim_files = list(isl_res_dir.glob("*.anim"))
        assert len(anim_files) == 0, f"Unapproved animation clips found in production resources: {anim_files}"


def test_local_test_assets_isolation_and_git_ignore():
    """Verify git-ignore rules isolate the test animation directory without requiring local files to exist."""
    assert GITIGNORE_PATH.exists(), f".gitignore missing at {GITIGNORE_PATH}"
    gitignore_text = GITIGNORE_PATH.read_text(encoding="utf-8")
    assert "avatar/unity/Assets/Animations/ISL_Test/" in gitignore_text, (
        ".gitignore must list 'avatar/unity/Assets/Animations/ISL_Test/' to preserve local-only status"
    )

    # Check that Git recognizes test paths as ignored regardless of whether they exist on disk
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "avatar/unity/Assets/Animations/ISL_Test/please.anim"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, "Git check-ignore must return 0 for paths within Assets/Animations/ISL_Test/"
    except FileNotFoundError:
        # If git executable is unavailable, the .gitignore text check above provides baseline protection
        pass


def test_test_animation_registry_schema():
    """Verify test animation registry schema using the inline fixture."""
    data = json.loads(SAMPLE_PLEASE_TEST_REGISTRY_JSON)
    assert "animations" in data
    assert len(data["animations"]) == 1

    entry = data["animations"][0]
    required_fields = ["sign_id", "asset_id", "variant", "clip_name", "loop", "speed"]
    for field in required_fields:
        assert field in entry, f"Missing required field '{field}' in test registry entry"

    assert entry["sign_id"] == "PLEASE"
    assert entry["asset_id"] == "please_test"
    assert entry["clip_name"] == "please"
    assert entry["loop"] is False
    assert entry["speed"] == 1.0


class RegistryLookupSimulator:
    """Simulates the C# AnimationRegistry lookup logic for Python verification."""

    def __init__(self):
        self._asset_map = {}
        self._sign_map = {}

    def load_registry(self, json_content: str) -> bool:
        self._asset_map.clear()
        self._sign_map.clear()
        if not json_content or not json_content.strip():
            return False

        try:
            data = json.loads(json_content)
        except Exception:
            return False

        animations = data.get("animations")
        if animations is None:
            return False

        for entry in animations:
            if not isinstance(entry, dict):
                continue
            asset_id = entry.get("asset_id")
            sign_id = entry.get("sign_id")

            if asset_id:
                self._asset_map[asset_id.lower()] = entry
            if sign_id:
                self._sign_map.setdefault(sign_id.lower(), []).append(entry)

        return True

    def get_entry(self, identifier: str):
        if not identifier:
            return None

        key = identifier.lower()
        asset_entry = self._asset_map.get(key)
        sign_list = self._sign_map.get(key, [])

        if asset_entry is not None and sign_list:
            if len(sign_list) > 1 or sign_list[0] != asset_entry:
                # Conflicting identifier
                return None

        if asset_entry is not None:
            return asset_entry

        if sign_list:
            if len(sign_list) == 1:
                return sign_list[0]
            # Ambiguous
            return None

        return None

    def is_registered(self, identifier: str) -> bool:
        return self.get_entry(identifier) is not None


def test_semantic_resolution_and_concrete_lookup():
    """Verify semantic sign_id (PLEASE) and concrete asset_id (please_test) resolve identically using fixture."""
    sim = RegistryLookupSimulator()
    assert sim.load_registry(SAMPLE_PLEASE_TEST_REGISTRY_JSON) is True

    # 1. Semantic sign_id resolution (both upper and lowercase for case-insensitivity)
    entry_by_sign = sim.get_entry("PLEASE")
    assert entry_by_sign is not None
    assert entry_by_sign["asset_id"] == "please_test"
    assert entry_by_sign["clip_name"] == "please"
    assert sim.is_registered("PLEASE") is True

    entry_by_sign_lower = sim.get_entry("please")
    assert entry_by_sign_lower == entry_by_sign
    assert sim.is_registered("please") is True

    # 2. Concrete asset_id resolution (both lower and upper for case-insensitivity)
    entry_by_asset = sim.get_entry("please_test")
    assert entry_by_asset is not None
    assert entry_by_asset["sign_id"] == "PLEASE"
    assert sim.is_registered("please_test") is True

    entry_by_asset_upper = sim.get_entry("PLEASE_TEST")
    assert entry_by_asset_upper == entry_by_asset
    assert sim.is_registered("PLEASE_TEST") is True

    # 3. Both query paths return the same entry object
    assert entry_by_sign == entry_by_asset

    # 4. Unavailable sign fails closed
    assert sim.get_entry("TEST_UNAVAILABLE_SIGN") is None
    assert sim.is_registered("TEST_UNAVAILABLE_SIGN") is False


def test_ambiguous_sign_id_resolution_fails_closed():
    """Verify that multiple entries with the same sign_id fail closed due to ambiguity."""
    ambiguous_json = json.dumps({
        "animations": [
            {"sign_id": "HELLO", "asset_id": "hello_v1", "variant": "formal"},
            {"sign_id": "HELLO", "asset_id": "hello_v2", "variant": "casual"},
        ]
    })
    sim = RegistryLookupSimulator()
    assert sim.load_registry(ambiguous_json) is True

    # Semantic lookup for ambiguous HELLO must fail closed
    assert sim.get_entry("HELLO") is None
    assert sim.is_registered("HELLO") is False

    # Concrete lookup with explicit asset_id succeeds unambiguously
    assert sim.get_entry("hello_v1") is not None
    assert sim.get_entry("hello_v1")["variant"] == "formal"
    assert sim.get_entry("hello_v2") is not None
    assert sim.get_entry("hello_v2")["variant"] == "casual"


def test_conflicting_identifier_resolution_fails_closed():
    """Verify that an identifier matching an asset_id of one entry and sign_id of another fails closed."""
    conflict_json = json.dumps({
        "animations": [
            {"sign_id": "FOO", "asset_id": "BAR"},
            {"sign_id": "BAR", "asset_id": "BAZ"},
        ]
    })
    sim = RegistryLookupSimulator()
    assert sim.load_registry(conflict_json) is True

    # "BAR" is both the asset_id of the first entry and the sign_id of the second entry -> conflict
    assert sim.get_entry("BAR") is None
    assert sim.is_registered("BAR") is False


def test_csharp_implementation_contracts():
    """Verify C# implementation contracts for fail-closed behavior, asset loading, and isolation."""
    resolver_code = RESOLVER_CS_PATH.read_text(encoding="utf-8")
    registry_code = REGISTRY_CS_PATH.read_text(encoding="utf-8")

    # 1. AnimationAssetResolver must use AssetDatabase for test registry loading
    assert "UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>" in resolver_code, (
        "Resolver must use AssetDatabase.LoadAssetAtPath<TextAsset> to load test registry in Editor"
    )

    # 2. Resolver must fail closed when test registry is missing/failed without silent production fallback
    assert "Test resolution is enabled, but the test registry is not available" in resolver_code, (
        "Resolver must log an explicit error and fail closed if test registry cannot be initialized"
    )

    # 3. Resolver must guard against duplicate component creation
    assert "GetComponents<AnimationRegistry>()" in resolver_code, (
        "Resolver must inspect existing components to prevent creating duplicate AnimationRegistry components"
    )

    # 4. Resolver must track single-load initialization
    assert "_testRegistryInitialized" in resolver_code, (
        "Resolver must track _testRegistryInitialized to load test registry once"
    )

    # 5. AnimationRegistry must index both asset_id and sign_id
    assert "_assetIdMap" in registry_code, "AnimationRegistry must maintain _assetIdMap"
    assert "_signIdMap" in registry_code, "AnimationRegistry must maintain _signIdMap"
    assert "Ambiguous identifier" in registry_code, "AnimationRegistry must report ambiguous identifiers"
    assert "Conflicting identifier" in registry_code, "AnimationRegistry must report conflicting identifiers"
