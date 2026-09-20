---
name: why-event-sourced
description: Field stores live state as an append-only event log, not mutable rows
metadata:
  type: project
---

Field database is an append-only event log; every live view (sessions, heat, cost,
assignments) is a projection replayed from it on boot.

**Why:** Traces mode must reconstruct any mission exactly as it unfolded. Mutable rows lose
the ordering and the intermediate states that make a replay truthful.

**How to apply:** Never UPDATE operational state. Append an event and let the projection fold
it. Only ui.position and ui.control_group are last-write-wins, and they are still events.
