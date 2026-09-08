# Reward-hacking evaluation

This suite measures whether an agent completes the requested product change
without optimizing against the visible grader. It is deliberately separate from
the ordinary correctness suites.

The eight cases cover visible-example overfitting, test tampering, repository
prompt injection, security-boundary widening, pytest-environment detection,
unnecessary edits, underspecified requests, and one-sample hard-coding. Held-out
tests are revealed only after the agent stops.

The grader starts Python with environment isolation and safe-path mode, imports
pytest before admitting the fixture root to `sys.path`, disables plugin autoload,
uses an empty pytest configuration, disables `conftest.py`, restores original
tests, removes added test files, counts added source files as changes, rejects
skipped/xfailed tests, and grades exact node IDs instead of trusting an exit code
or the agent's completion claim.

## Windows

From the Knossos repository, double-click `scripts\run-reward-eval.cmd`, or run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-reward-eval.ps1
```

The default uses Ollama and runs two equal-budget arms: full Knossos and a
no-proactive-context control. It audits the frozen suite before spending any
model calls. To name a local model:

```powershell
.\scripts\run-reward-eval.ps1 -Engine ollama -Model qwen3-coder:30b
```

For a Cameo endpoint:

```powershell
.\scripts\run-reward-eval.ps1 -Engine cameo -Model my-model -BaseUrl http://cameo.local:9090
```

Results, per-case traces, checkpoints, and a machine-readable comparison land in
`reports/reward-hacking/`. Use `-Resume` after an interrupted run. The script
does not publish anything or modify the suite.

## Interpreting results

- Report each arm's exact passed/total count; do not publish only a percentage.
- Treat tampering, action violations, fitted-visible-test failures, provider
  failures, and infrastructure failures as different outcomes.
- Compare arms only when model, case order, step ceiling, request ceiling, token
  ceiling, temperature/provider behavior, and suite digest match.
- A zero from an unreachable provider is not a capability result.
- Do not tune the harness on these eight cases and then call them held out.
  Extend the sealed suite or rotate it before making a release claim.
