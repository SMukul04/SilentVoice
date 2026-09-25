# 3D Avatar & ISL Animation Requirements

## 1. Avatar Requirements

### Humanoid Structure
At a minimum, the avatar must support a standard Unity Humanoid rig hierarchy:
```
Root
 ├── Hips
 ├── Spine
 ├── Chest
 ├── Neck
 ├── Head
 ├── Left Arm
 │    ├── Upper Arm
 │    ├── Lower Arm
 │    └── Hand
 │         └── Fingers
 └── Right Arm
      ├── Upper Arm
      ├── Lower Arm
      └── Hand
           └── Fingers
```
The exact hierarchy may differ between rigs, but the avatar must provide proper humanoid bone mapping.

### Critical ISL Requirements
For Indian Sign Language (ISL) accuracy, the avatar must have:
* Independently articulated fingers
* Wrist rotation
* Hand rotation
* Forearm rotation
* Elbow articulation
* Shoulder articulation
* Sufficient facial/head movement support if required by future ISL assets

Hand and finger articulation is a hard requirement. A visually attractive avatar with poor finger rigging is not acceptable for SilentVoice.

## 2. Avatar Selection Criteria

### Technical
* Unity Humanoid compatibility
* Valid humanoid rig
* Sufficient hand/finger bones
* Animation retargeting compatibility
* Compatible file format (FBX, GLB/GLTF)
* Reasonable polygon count
* Reasonable texture size
* WebGL suitability
* No unnecessary runtime dependencies

### Visual
* Hands clearly visible
* Fingers clearly distinguishable
* Upper body visible
* Neutral/default pose
* Suitable camera framing
* Suitable lighting
* No visual obstruction of hand movements

### Licensing
The avatar must have a license that permits the intended SilentVoice use. We must record:
* Asset source
* Creator/provider
* License type
* Commercial/non-commercial restrictions
* Redistribution restrictions
* Attribution requirements

Do not assume an asset is free merely because it is downloadable.

## 3. ISL Accuracy Requirement
Do NOT treat generic hand gestures as Indian Sign Language animations. SilentVoice specifically targets Indian Sign Language (ISL). Therefore:
`generic gesture ≠ ISL sign`

We must not claim that an animation represents an ISL sign unless its provenance or intended semantic mapping supports that claim. If suitable ISL animation assets are not yet available, we will document that honestly rather than fabricating animation mappings.

## 4. Animation Asset Contract
The provider-neutral metadata contract conceptually corresponds to the backend `AnimationAsset`.
Unity-side representation:
```json
{
  "sign_id": "HELLO",
  "asset_id": "hello_default",
  "variant": "default",
  "clip_name": "HELLO_Default",
  "loop": false,
  "speed": 1.0
}
```
The Unity metadata identifies the local animation resource and avoids unnecessarily duplicating backend business logic.

## 5. SIGN_ID vs ASSET_ID
* **sign_id**: Semantic identity (e.g., `HELLO`, `THANK_YOU`, `PLEASE`)
* **asset_id**: Concrete animation resource (e.g., `hello_default`, `thank_you_default`, `please_default`)

Flow:
`sign_id -> AnimationAssetService -> asset_id -> Unity animation resource`

Unity primarily executes the `asset_id`. It does NOT perform semantic English → ISL resolution.

## 6. Animation Asset Naming Standard
* Standard: `<sign_id>_<variant>`
* Examples: `HELLO_default`, `THANK_YOU_default`, `PLEASE_default`
* Unity Clip Names: `HELLO_Default`, `THANK_YOU_Default`, `PLEASE_Default`

Naming is an implementation convention, not the semantic definition of the sign.

## 7. Animation Requirements
Each clip should ideally:
* Start from a known compatible pose
* Use the same humanoid rig as the avatar
* Have clear beginning/end boundaries
* Contain no unintended looping (`loop = false` by default)
* Preserve finger articulation and wrist/hand motion
* Avoid abrupt snapping where possible
* Provide a deterministic completion point

## 8. Transition Requirements
Expected transition flow:
```
HELLO
  ↓ completion
neutral/transition
  ↓
THANK_YOU
  ↓ completion
neutral/transition
  ↓
PLEASE
```
The runtime must not assume a fixed animation duration. Completion will come from the actual Unity animation system.

## 9. Placeholder Animation Policy
If placeholders are needed to verify architecture, they must be clearly named (e.g., `placeholder`, `placeholder_default`). They must NOT be represented as valid ISL semantic signs (e.g., `HELLO`) unless they are actually ISL animations.

## 10. Future Unity Import Contract
Actual avatar/animation imports will later require:
```
3D avatar -> Humanoid rig configuration -> Avatar definition -> Animation clips -> Animation metadata -> Animation registry
```
Before importing, we must verify rig compatibility, humanoid retargeting, root motion requirements, clip boundaries, frame rate, hand/finger bones, and animation loop settings.
