---
id: trace-vram
name: Trace the VRAM control plane
workspace: cameo
target: core/placement/
roles: [scout, architect]
thinking: high
definition_of_done:
  - "a written map from placement request to device assignment, with file:line per hop"
  - "every branch that can silently fall back to CPU is named"
---
# Trace the VRAM control plane

Produce a map of how a placement request becomes a device assignment in core/placement/,
and identify where the Knossos VRAM control plane can silently degrade to CPU.

## Orders

1. **Scout** - trace core/placement/src/command.rs then model.rs then lib.rs and record
   every decision point with file:line.
2. **Architect** - write the map to docs/inference-tuning.md. Name the silent-fallback
   branches explicitly; those are the ones that cost a support ticket.
