# Roman Field asset provenance

The canonical machine-readable inventory is `asset-manifest.json`. Every shipped image is
content-addressed there and must be referenced by the product. `npm run assets:audit` verifies those
properties; `npm run release:audit` additionally requires an approved creator, license, and review
status before a public Field package may include the files.

On 2026-09-06 the product owner attested that all six current Field images were generated for this
repository with ChatGPT/Codex built-in image generation, then exported (background removed / resized
as needed) into the shipped PNG/WebP files. They ship under the same Apache-2.0 license as
Cameo/Knossos. No third-party stock library is claimed.

| Asset | Product use | Creator/source | License | Review |
| --- | --- | --- | --- | --- |
| `assets/mediterranean-relief-v1.png` | Roman Field and Atlas map relief | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |
| `assets/living-rome/capital-tier-3.webp` | Evidence-tier III/IV capital settlement | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |
| `assets/living-rome/modules/archive.webp` | Procedural archive/observatory | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |
| `assets/living-rome/modules/forum.webp` | Procedural civic forum | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |
| `assets/living-rome/modules/gate.webp` | Procedural fortified gate | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |
| `assets/living-rome/modules/works.webp` | Procedural engineering works | Korbin Sadlowski; ChatGPT/Codex built-in image generation | Apache-2.0 | **APPROVED** |

If a later asset is added from an unknown source, it must enter the manifest as `BLOCKED` until the
same creator/license/review fields are filled. Do not treat filename or git history as rights
evidence.
