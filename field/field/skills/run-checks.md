---
id: run-checks
name: Run project checks
workspaces: [cameo]
---
# Run project checks

The real commands that prove the Cameo workspace is healthy. Run them from the repo root.

    cargo check --workspace
    shellcheck archiso/airootfs/usr/local/bin/*

Report the actual output. A check that was skipped is reported as skipped, never as passing.
