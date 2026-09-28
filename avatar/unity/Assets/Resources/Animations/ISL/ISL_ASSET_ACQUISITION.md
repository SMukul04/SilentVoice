# Production ISL Animation Asset Acquisition

## 1. Purpose
SilentVoice requires genuine ISL animation assets for the text-to-sign and avatar pipeline. To ensure legal compliance and technical reliability, all animation assets must have documented provenance and sufficient permission before they can enter the production repository. This document defines the approved workflow for acquiring and importing these assets.

## 2. Approval Gate
An animation source is production-approved only when **all** required conditions are satisfied:
- Source identity is known
- Source URL is recorded
- Exact asset/file is identifiable
- License or explicit permission is documented
- Redistribution permission is known
- Modification permission is known
- Attribution requirements are known
- Technical compatibility with the SilentVoice avatar has been verified
- The animation represents the intended ISL sign
- Provenance is recorded in `ISL_ASSET_PROVENANCE.md`
- Corresponding runtime registry entry is reviewed before production use

If any required condition is unknown, the status is **NOT_APPROVED**. The asset may remain local-only for evaluation but must not enter production.

## 3. Source Classification
Animation sources are categorized as follows:
- **A.** Explicitly licensed animation assets
- **B.** Explicit written permission from asset owner/creator
- **C.** Public research/demo assets with unclear asset licensing
- **D.** Generated/created internally by SilentVoice
- **E.** Assets derived from another licensed source
- **F.** Restricted/non-redistributable assets

Default treatment by category:
- **A** -> Potentially approved after verification
- **B** -> Potentially approved after retaining permission evidence
- **C** -> Local evaluation only
- **D** -> Potentially approved, provided SilentVoice owns/has rights to the result
- **E** -> Requires verification of upstream and derivative rights
- **F** -> Not production-approved

## 4. License Verification
The following conditions are **NOT** sufficient by themselves to assume redistribution rights:
- GitHub repository is public
- Repository description says "open source"
- README says "free"
- Software code is MIT licensed
- Asset can technically be downloaded
- Asset appears in a public demo

The exact animation/data/model asset itself must have explicitly granted rights.

## 5. Required Evidence
For each approved asset record, the following must be documented:
- `sign_id`
- `asset_id`
- `variant`
- `source`
- `source_url`
- `asset_identifier`
- `license`
- `permission_status`
- `redistribution_allowed`
- `modification_allowed`
- `attribution_required`
- `local_asset_path`
- `evidence_reference`
- `notes`

`evidence_reference` should point to the exact license file, terms page, permission statement, or written authorization used to make the determination.

**Privacy Rule:** Do not store private email addresses, personal contact details, credentials, or secrets in the repository. If permission was granted privately, document only a non-sensitive reference such as:
> "Written permission obtained from asset owner on YYYY-MM-DD; private correspondence retained outside repository."

Do not invent such evidence for existing assets.

## 6. Technical Validation Gate
Once licensing is approved, the animation must also pass technical validation:
- Imports into Unity
- Compatible with the SilentVoice humanoid avatar
- Hand/finger movement is preserved
- Intended sign motion is actually present
- Animation starts correctly
- Animation completes correctly
- No unintended looping
- Speed metadata works
- Animation can be resolved by `asset_id`
- Production playback path works
- Completion callback works
- No unrelated Animator states are affected

Explicitly distinguish **TECHNICAL_COMPATIBILITY** from **LEGAL/PROVENANCE_APPROVAL**. A technically compatible asset is NOT automatically production-approved.

## 7. ISL Semantic Validation
An animation must also be verified as representing the intended ISL sign. Technical motion compatibility alone is insufficient.

For example:
`PLEASE` -> `please_v1`
must mean the intended ISL sign PLEASE, not merely be an animation named "please".

If semantic/sign-language correctness is uncertain, do not mark the asset VERIFIED.

## 8. Approval States
The acquisition process uses the following states:
- **CANDIDATE**
- **TECHNICALLY_VALIDATED**
- **PROVENANCE_PENDING**
- **APPROVED**
- **REJECTED**
- **LOCAL_ONLY**

Example transitions:
```text
CANDIDATE
  |
  v
TECHNICALLY_VALIDATED
  |
  v
APPROVED
```
OR
```text
CANDIDATE
  |
  v
PROVENANCE_PENDING
  |
  v
LOCAL_ONLY / REJECTED
```

## 9. Production Import Procedure
Only after reaching the **APPROVED** state:
1. Place the animation in `Assets/Resources/Animations/ISL/`.
2. Record its provenance.
3. Add its runtime metadata to `Assets/Resources/Animations/animation_registry.json`.
4. Verify `sign_id` and `asset_id`.
5. Run avatar playback validation.
6. Review Git diff.
7. Ensure no unrelated binaries or temporary files are included.
8. Commit the approved asset and metadata as an atomic change.

Until approval, the animation stays completely outside the production directories.

## 10. Current Candidates

| Candidate | Current status | Reason |
|---|---|---|
| VirtualISL / IIIT-B animations | PROVENANCE_PENDING | Publicly described animation resource, but exact redistribution rights for individual animation assets have not been established |
| Local `please.anim` test asset | LOCAL_ONLY | Technical compatibility demonstrated, but production redistribution permission has not been established |
| Internally authored SilentVoice animation | CANDIDATE | Requires authorship/rights and technical/ISL validation |
| Future explicitly licensed asset | CANDIDATE | Requires verification |

*No candidates are currently approved for production.*

## 11. Current Production State
- `animation_registry.json` is intentionally empty.
- No production ISL animation binaries are currently approved.
- The existing `ISL_Test` directory is local-only.
- The Y Bot model is local-only.

## 12. Decision Checklist
- [ ] exact source identified
- [ ] exact asset identified
- [ ] source URL recorded
- [ ] license identified
- [ ] redistribution rights verified
- [ ] modification rights verified
- [ ] attribution requirements verified
- [ ] evidence recorded
- [ ] ISL semantic correctness verified
- [ ] Unity import verified
- [ ] humanoid/avatar compatibility verified
- [ ] playback verified
- [ ] completion verified
- [ ] provenance document updated
- [ ] registry entry reviewed
- [ ] production asset approved
