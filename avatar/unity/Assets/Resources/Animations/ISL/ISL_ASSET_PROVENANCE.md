# Production ISL Animation Asset Provenance

This document defines the strict provenance and licensing requirements for any Indian Sign Language (ISL) animation asset introduced into the SilentVoice production repository.

## Provenance Rules

* A production animation MUST NOT be placed in `Assets/Resources/Animations/ISL/` until provenance and redistribution permission have been verified.
* Public availability on GitHub (or other platforms) does NOT by itself establish redistribution permission.
* "Open source" must not be assumed without an identifiable license or explicit written permission.
* If license or permission is unknown, the asset remains local-only and must not enter the production registry.
* Every production registry entry must correspond to a provenance record.
* `sign_id` and `asset_id` must remain separate concepts.
* The runtime registry (`animation_registry.json`) should contain only playback metadata; licensing and provenance information belongs in the provenance documentation.
* Do not place secrets, private contact information, or credentials in the provenance file.

## Required Provenance Fields

For each future production ISL animation asset, document the following fields:

1. **sign_id**: SilentVoice canonical sign identifier (e.g., PLEASE).
2. **asset_id**: Stable SilentVoice animation asset identifier (e.g., please_v1).
3. **variant**: Animation variant identifier (e.g., default).
4. **source**: Name of the original asset/source/provider.
5. **source_url**: URL where the asset originated. Do not invent URLs.
6. **license**: Exact known license name or "UNKNOWN / NOT VERIFIED".
7. **permission_status**: Use an explicit controlled vocabulary: `VERIFIED`, `PENDING`, `NOT_VERIFIED`, `RESTRICTED`, or `DENIED`.
8. **redistribution_allowed**: `true` / `false` / `unknown`.
9. **modification_allowed**: `true` / `false` / `unknown`.
10. **attribution_required**: `true` / `false` / `unknown`.
11. **local_asset_path**: Expected path inside the SilentVoice Unity project once an approved asset exists.
12. **notes**: Additional provenance/licensing/technical notes.

## TEMPLATE / NON-PRODUCTION EXAMPLE

The following is a strictly fictional template demonstrating what a verified future production record could look like. **This is not a real production asset.**

```yaml
sign_id: EXAMPLE_SIGN
asset_id: example_sign_v1
variant: default
source: Example Licensed Provider
source_url: https://example.com/verified-source-url
license: Example License
permission_status: VERIFIED
redistribution_allowed: true
modification_allowed: true
attribution_required: true
local_asset_path: Assets/Resources/Animations/ISL/example_sign.anim
notes: Example only; not a real production asset.
```
