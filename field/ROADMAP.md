# Field — implementation plan

Measured against [PRODUCT.md](PRODUCT.md). Items are ordered by what an
operator notices first, with the cost honestly stated. A phase is done when
every item in it is landed, tested and looked at in the running desktop app,
not when the code compiles.

Status key: **todo**, **doing**, **done**, **cut**.

---

## Phase 1 — It looks made (the edges)

The complaint is "bland and basic", and the cause is measurable: the border
token is `#262B31` on a `#15181B` surface, a 1.35:1 luminance step that is
technically present and visually absent, and in the whole stylesheet there is
not one inset highlight, gradient or grain. Flat rectangles with invisible
outlines is the shape that reads as generated.

1. **Edge and depth system.** `todo` — 2 h. Raise the hairline so it is a
   hairline; give every card, sheet and modal a one-pixel inset highlight on
   the top edge and a darker bottom edge so surfaces read as lit objects; give
   the board a base gradient and the header real elevation. Define three border
   tokens (subtle, default, strong) and three shadow stacks (resting, hovered,
   raised) and use nothing else. Fix nested radii: an inner radius is the outer
   minus the padding.
2. **Control rhythm.** `todo` — 2 h. One height scale for buttons, inputs and
   the segmented control, one internal padding rhythm, hover and focus states
   that change more than a border colour. Scrollbars styled.
3. **Un-crush the art.** `todo` — 30 min. The settlement art currently renders
   at roughly 44% of its authored brightness through a stack of filters, a dark
   overlay, a vignette and scanlines. Lift all four. Largest single visible win
   on Rome, and several later items depend on being able to see the map.
4. **Type floor.** `todo` — 1 h. Type as small as five pixels ships in several
   places. Nothing below the eleven-pixel step survives; the small step is the
   uppercase mono label. Open the ramp at the top so the page has a focal
   point.

## Phase 2 — It plays as a loop

Per PRODUCT.md, the first second may show three things: where the work is,
what needs you, how much is running.

5. **Density pass.** `doing` — the counters collapse to one badge, the
   conversation meta row becomes one plain line plus a progress indication,
   district counts leave the map, and every fact that leaves the first screen
   lands exactly one tap deeper.
6. **The two decision moments.** `doing` — approving a permission and accepting
   a change each get one uncluttered card, the context that matters, one or two
   buttons, and a quick beat on completion.
7. **Attention from the world.** `todo` — 3 h. A district that wants you says
   so through its own appearance and a badge, not through text in a sheet you
   have to open. Same for a conversation in Atlas: state first, words second.
8. **Accepting a change is the reward.** `todo` — 3 h. The diff lands, the
   district registers it, the number of pending changes drops, and the
   settlement's evidence moves. This is the beat the loop pays out on; it
   should feel like the point of the session.

## Phase 3 — The world means something

The rule from PRODUCT.md: the game layer may never invent signal. These items
make real signal visible, which is the only kind of growth we ship.

9. **Visible growth.** `todo` — needs art, 1 day of authoring plus 2 h of
    wiring. Settlement tiers already exist and are driven by real maturity, but
    only the top tier has art, so a project never visibly grows. Author the
    lower tiers at the same camera and wire them to the existing score.
10. **District silhouettes.** `todo` — 5 h. Every folder in the world is the
    same small trapezoid, so nothing is legible by shape. Derive a kind from
    the folder (source, tests, docs, assets, scripts) and draw five or six
    distinct cutouts at the size the art bible already specifies.
11. **Real parchment.** `todo` — 3 h. The map ground is a dark relief
    photograph, not paper: no grain, no deckle, no edge, no drawn sea. Warm
    base, grain layer, inset edge shadow and a double rule inset from the
    canvas. Public-domain and CC0 sources exist for all of it.
12. **One accent, one job.** `todo` — 1 h. Orange means "this needs you",
    everywhere, and nothing else. The map's warm ink becomes a material rather
    than a signal, and work in progress reads cool, so a live agent is the only
    warm point on a cold field.

## Phase 4 — It is an application

13. **Own the window.** `todo` — 4 h. Custom title bar merged with the existing
    top bar, so the app stops looking like a website in a box and loses chrome
    rather than gaining it. Window effects where the platform allows.
14. **Tray presence.** `todo` — 2 h. The library is already in the lock file. A
    monochrome glyph that turns to the attention colour when an agent needs a
    decision, with a short menu. The strongest single signal that this is an
    application, for a tool that runs unattended.
15. **App icon.** `todo` — 1 h. A drawn mark rather than the scaffold icons.

## Phase 5 — The harness catches up to the map

16. **Agents are configurations.** `todo` — 1 day, server and client. Today an
    agent record is a character with a class. It becomes a model, a tool
    allowlist, a permission posture, capabilities, instructions, a delegation
    limit, a budget and a definition of done. Roles become editable presets
    over those. The fictional classes and character names go.
17. **Per-project verification contract.** `todo` — half a day. What "done"
    means for this project, editable, confirmed at first run: the commands that
    must pass before an agent may claim a change is complete.
18. **First run.** `todo` — half a day. Mount a project, choose a model, accept
    the verification contract, start the first agent. No documentation.

## Phase 6 — Ship it

19. **Signed installers.** `todo` — 1 day plus account setup. Build the three
    installers in CI, sign and notarize on macOS with the existing developer
    account, sign on Windows once a certificate exists.
20. **In-app updates.** `todo` — half a day. Update manifest published with each
    release, verified against a bundled public key, one click to install and
    restart.
21. **Field behind the Cameo origin.** `todo` — 2 h. The daemon already proxies
    `/field/`; Field needs to honour the forwarded headers so the browser flow
    works through it.

---

## Not doing, and why

- **A canvas or WebGL map.** The current layout is fine at the scale we have,
  and rebuilding it costs focus rings, hit testing and a month.
- **A component library migration.** The token system is coherent; a migration
  buys nothing visible.
- **Per-agent portrait art.** The colour and initials derived from identity are
  a good, cheap device already.
- **Swarms.** Many peers negotiating is a research demo: cost multiplies,
  coordination is unreliable, and the result cannot be reviewed.
- **Any progress that time alone fills.** See the rule in PRODUCT.md.
