# ISL Animation Source Verification

## 1. Purpose
This document records research into possible sources of production ISL animation assets. The public availability of a repository does not establish redistribution permission for its assets. This document tracks the current licensing and provenance status of candidate sources to ensure compliance before any asset enters the SilentVoice production repository.

## 2. Verification Standard
For every candidate source, the verification process must distinguish between:
- Repository/project license
- Individual animation/model asset license
- Dataset/data license
- Third-party dependencies/assets
- Explicit redistribution permission
- Modification permission
- Attribution requirements

These elements must never be collapsed into a single "license" field. Each must be evaluated independently.

## 3. Candidate Evaluation

| Source | ISL relevance | Animation availability | Repository license | Asset-level license | Redistribution evidence | Current status |
|---|---|---|---|---|---|---|
| IIIT-B Virtual ISL | ~1300 signs with fewer non-manual features + ~300 additional naturally animated signs | Limited direct, rest on request | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | PROVENANCE_PENDING |
| SanketBani | Text/image/gesture to 3D ISL | 285+ prebuilt gestures | MIT | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | CANDIDATE |
| Avatar-based ISL Toolkit | Speech/text to animated gestures | Avatar capabilities present | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | CANDIDATE |
| Sijosaju Project | 3D avatar/ISL animation | Academic/educational use | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | PROVENANCE_PENDING |
| ResQSign / SignWave | Avatar interaction/emergency | Animation assets present | None specified | UNKNOWN / NOT ESTABLISHED | UNKNOWN / NOT ESTABLISHED | NOT_APPROVED |

## 4. IIIT-B Virtual ISL Interpreter
**Official Project:** [https://cognitive.iiitb.ac.in/isli/](https://cognitive.iiitb.ac.in/isli/)
**Facts:** The project details an animation vocabulary of approximately 1300 signs (fewer non-manual features) and about 300 additional naturally animated signs (with non-manual features). Limited animations are directly available, while others are available on request. The project aimed to make the solution and generated data open source.

**Important Distinction:** The project's open-source goal does not equate to verified redistribution permission for specific animation files.
**Current status:** PROVENANCE_PENDING

## 5. SanketBani
**Repository:** [https://github.com/Phinix-BI/SanketBani](https://github.com/Phinix-BI/SanketBani)
**Facts:** The project converts text/images/gestures into 3D ISL animations. According to its README, it includes 285+ prebuilt ISL gestures and claims MIT licensing for the repository.

**Important Distinction:** The repository-level MIT statement must NOT automatically be treated as proof that every bundled 3D model, animation, texture, dataset, or third-party asset is MIT licensed. Asset-level licensing still needs verification.
**Current status:** CANDIDATE

## 6. Avatar-based ISL Toolkit
**Repository:** [https://github.com/igennova/Avatar-based-ISL-Toolkit](https://github.com/igennova/Avatar-based-ISL-Toolkit)
**Facts:** A web toolkit converting speech/text into animated ISL gestures using 3D avatars. Relevant ISL avatar technology exists, but sufficiently clear asset-level redistribution rights and licensing remain uncertain.
**Current status:** CANDIDATE

## 7. Sijosaju Project
**Repository:** [https://github.com/Sijosaju/Speech-to-Indian-Sign-Language-using-3D-Avatar-Animations](https://github.com/Sijosaju/Speech-to-Indian-Sign-Language-using-3D-Avatar-Animations)
**Facts:** Contains 3D avatar animation assets and describes the project as academic, educational, and portfolio-oriented. This wording cannot be interpreted as redistribution permission for SilentVoice.
**Current status:** PROVENANCE_PENDING

## 8. ResQSign / SignWave
**Repository:** [https://github.com/SanjithaBolisetti/ResQSign-An-Intelligent-Sign-Language-System-for-Interaction-and-Emergency-Assistance](https://github.com/SanjithaBolisetti/ResQSign-An-Intelligent-Sign-Language-System-for-Interaction-and-Emergency-Assistance)
**Facts:** The repository contains animation assets for a 3D avatar but explicitly states no repository license.
**Current status:** NOT_APPROVED

## 9. Evidence Rules
For an asset to become APPROVED, the following are required:
- Exact asset identified
- Exact source identified
- Exact license identified
- Asset-level license established
  **OR**
- Explicit written permission establishing redistribution rights
- Modification rights established
- Attribution requirements established
- Technical compatibility established
- ISL semantic correctness established

If any critical item remains unknown, do not approve the asset.

## 10. Acquisition Priority
Sources are classified by acquisition state (not a quality ranking):
- **Ready for rights verification:** SanketBani
- **Requires explicit permission/request:** IIIT-B Virtual ISL Interpreter
- **Insufficient licensing evidence:** Avatar-based ISL Toolkit, Sijosaju Project
- **Not approved:** ResQSign / SignWave

## 11. Recommended Verification Actions
**A. For IIIT-B:**
- Contact/request the specific required animation assets.
- Request explicit permission to use and redistribute them inside SilentVoice.
- Request applicable license/terms.
- Retain non-sensitive evidence reference.
- Update provenance only after evidence is received.

**B. For SanketBani:**
- Inspect the exact animation/model files.
- Determine whether their license is stated separately.
- Identify any third-party assets.
- Verify whether the MIT license applies to those specific files.
- Do not copy assets until verified.

**C. For the remaining repositories:**
- Inspect asset-specific licensing.
- If unclear, do not redistribute.
- Request permission where appropriate.

## 12. Current SilentVoice Decision
- No external animation source is currently production-approved.
- Production `animation_registry.json` remains strictly empty.
- `please.anim` remains LOCAL_ONLY.
- `ISL_Test` remains local-only.
- `Y_Bot` remains local-only.
- The architecture is ready to accept approved assets once verified in the future.

## 13. Verification Checklist
- [ ] source URL verified
- [ ] repository license identified
- [ ] exact asset identified
- [ ] asset-level license identified
- [ ] third-party asset provenance checked
- [ ] redistribution rights verified
- [ ] modification rights verified
- [ ] attribution requirements verified
- [ ] evidence recorded
- [ ] ISL semantic correctness verified
- [ ] Unity compatibility verified
- [ ] playback verified
- [ ] completion verified
- [ ] provenance updated
- [ ] production registry reviewed
- [ ] asset approved
