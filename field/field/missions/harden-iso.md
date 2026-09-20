---
id: harden-iso
name: Harden the ISO boot path
workspace: cameo
target: archiso/
roles: [scout, builder, verifier]
thinking: high
definition_of_done:
  - "shellcheck passes on every script under archiso/airootfs/usr/local/bin"
  - "cameo-install exits non-zero on every failure path it can reach"
  - "a verifier session has attached passing output"
---
# Harden the ISO boot path

The install and first-boot scripts under archiso/ run as root on hardware we cannot reach.
Every failure must be loud and every exit code must be honest.

## Orders

1. **Scout** - inventory every script under archiso/airootfs/usr/local/bin and list each
   command whose failure is currently swallowed.
2. **Builder** - add strict mode where missing, and make each failure path exit non-zero
   with a message naming the step that failed.
3. **Verifier** - run shellcheck across the tree and attach the real output.

## Constraints

- Do not change what the scripts do. This mission is about failure behavior only.
- cameo-console-init runs before the network exists. It may not call out to anything.
