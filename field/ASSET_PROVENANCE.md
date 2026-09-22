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

## Fonts

Field runs offline inside a Tauri shell, so the type is self-hosted rather than loaded from
`fonts.googleapis.com` (those requests fail and silently drop the whole app to `system-ui`). The
files live in `web/public/fonts/`, are declared with `@font-face` and `font-display: swap` in
`web/src/styles/app.css`, and are **not** covered by `asset-manifest.json` — the asset audit walks
`web/public/assets` only. They are third-party, so they are tracked here instead.

Source: the official IBM `@ibm/plex-*` npm packages (`npm pack`), `fonts/split/woff2/` — the
Latin-1 subsets (U+0000–U+00FF plus the punctuation, currency and `fi/fl` ligature ranges the UI
uses). No runtime dependency was added; only the `.woff2` files were copied. Total 168 KB.

| Asset | Product use | Creator/source | License | Review |
| --- | --- | --- | --- | --- |
| `fonts/IBMPlexSans-Regular-Latin1.woff2` | Body text (400) | IBM Corp. via `@ibm/plex-sans` 1.1.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexSans-Medium-Latin1.woff2` | Emphasis and controls (500) | IBM Corp. via `@ibm/plex-sans` 1.1.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexSans-SemiBold-Latin1.woff2` | Headings h1/h2 (600) | IBM Corp. via `@ibm/plex-sans` 1.1.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexSansCondensed-SemiBold-Latin1.woff2` | Board display face (600) | IBM Corp. via `@ibm/plex-sans-condensed` 2.0.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexSansCondensed-Bold-Latin1.woff2` | Board display face (700) | IBM Corp. via `@ibm/plex-sans-condensed` 2.0.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexMono-Regular-Latin1.woff2` | Data, labels, code (400) | IBM Corp. via `@ibm/plex-mono` 2.5.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexMono-Medium-Latin1.woff2` | Labels and emphasis (500) | IBM Corp. via `@ibm/plex-mono` 2.5.0 | SIL OFL 1.1 | **APPROVED** |
| `fonts/IBMPlexSerif-SemiBold-Latin1.woff2` | Rome display face (600) | IBM Corp. via `@ibm/plex-serif` 2.0.0 | SIL OFL 1.1 | **APPROVED** |

The full licence text, as shipped in those packages, is copied verbatim to
`web/public/fonts/LICENSE.txt` and is served with the client. IBM Plex is
"Copyright © 2017 IBM Corp. with Reserved Font Name 'Plex'", licensed under the SIL Open Font
License 1.1: redistribution (including bundling with this app) is permitted, the files must stay
under the OFL, and the reserved name must not be used for a modified version. Field ships the files
unmodified; a *user-supplied* custom typeface, chosen in Settings → Typeface, is stored only in that
operator's browser and is never redistributed.
