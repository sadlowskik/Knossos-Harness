# Field completion lane

Date: 2026-09-05

This lane records the Field work selected from the production audit. Existing
security, lifecycle, budget, routine, Markdown, ignore-policy, event-integrity,
release-package, and asset-gate changes are treated as baseline evidence; this
lane does not repeat them.

## Gap inventory and gates

| Priority | Roadmap area | Evidence in this tree | Remaining gate | Dependency |
| --- | --- | --- | --- | --- |
| P1 | C5 event-log query bounds | `server/src/api.js` previously accepted unvalidated `from`/`limit`; `GET /api/trace` and campaign trace returned unbounded arrays; `server/src/store/db.js` subject reads had no cursor | `parseEventPage` validates every query, `pageResult` returns bounded `events`, `nextFrom`, and `head`, and `EventLog.bySubject` applies the same cursor semantics on SQLite/JSONL | Existing EventLog read primitives |
| P1 | F5 accessibility | `web/src/App.jsx`, `web/src/ui/useModalFocus.js`, and mode components have partial ARIA/focus coverage, but no complete WCAG 2.2 AA workflow audit | Keyboard, zoom, contrast, reduced-motion, screen-reader, and automated scan evidence for release-critical workflows | Browser/OS qualification |
| P1 | G4 performance | `web/vite.config.js` and `scripts/audit-assets.mjs` cover bundle checks; no retained real-browser LCP/CLS/interaction or 30-agent stream evidence | CI or release profile records agreed throttled-browser metrics and stream stability | Supported browser baseline |
| P1 | I1 release qualification | `.github/workflows/ci.yml` and `package.json` define local/CI hooks; observed matrix/live-provider evidence is incomplete | Linux/macOS/Windows jobs plus at least one live cloud and local provider acceptance run | Protected CI credentials/runners |
| P0 | G1 rights | `asset-manifest.json`, `ASSET_PROVENANCE.md`, and `scripts/audit-assets.mjs --release` still report six `BLOCKED`/`UNVERIFIED` artwork records | Supply source/license/attribution evidence or approve replacement assets | User/legal provenance; cannot be inferred |

## Selected implementation

Implement the C5 query-boundary slice first. A caller can request a finite
window from `/api/events`, `/api/trace`, and `/api/campaigns/trace`; invalid
pagination fails with a uniform domain error, and responses expose `head` and
`nextFrom`. Subject reads remain ordered and campaign membership semantics stay
unchanged. The server-side limit protects memory and response size while the
cursor lets the UI or an operator retrieve later windows.

Acceptance gates for this slice:

1. Missing parameters use safe defaults; limits above the configured maximum,
   negative cursors, fractions, NaN, and non-numeric values return a client
   error without a stack trace.
2. Each endpoint returns at most the requested bounded number of events, a
   monotonic `head`, and `nextFrom` only when more matching events remain.
3. SQLite and JSONL backends use the same ordering and cursor semantics.
4. Regression tests cover ordinary paging, empty pages, invalid inputs, and
   campaign/subject filtering.

## Dependencies and deferred work

Campaign trace currently derives membership from campaign projection IDs and
embedded campaign IDs. Replacing its full-log scan with a dedicated indexed
campaign projection is a separate schema/performance change and remains
deferred; this slice still bounds the returned result and makes the scan's
operational contract explicit. Accessibility, real-browser performance, CI
matrix/live-provider qualification, and artwork provenance require external
evidence and are not fabricated by local tests.
