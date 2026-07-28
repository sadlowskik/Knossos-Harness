<#
.SYNOPSIS
    Run every suite that needs a real model, with skips turned into failures.

.DESCRIPTION
    The live checks cannot run on a hosted CI runner: the model server is on the
    LAN. That leaves two options -- a self-hosted runner, or running them here.

    This is the second, and it is the one that works without registering
    anything, exposing this machine to workflow code, or handling a token. It
    runs exactly what the `live` job in .github/workflows/ci.yml runs, with the
    same STRICT variables, so a check that cannot run fails instead of passing
    quietly. That was the whole point of §2.5.

    Run it before pushing anything that touches the engine, the ACP layer, or
    the executor loop.

.PARAMETER OllamaHost
    Host of the model server. Default 192.168.4.103.

.PARAMETER Model
    Model to exercise. Default gemma4:e4b.

.PARAMETER Python
    Interpreter to run Knossos with. Defaults to `python` on PATH.

.EXAMPLE
    .\live-check.ps1
    .\live-check.ps1 -OllamaHost 10.0.0.5 -Model qwen2.5-coder:7b
#>
[CmdletBinding()]
param(
    [string]$OllamaHost = "192.168.4.103",
    [string]$Model = "gemma4:e4b",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$failed = @()

function Invoke-Suite {
    param([string]$Name, [string]$Directory, [scriptblock]$Body)

    Write-Host ""
    Write-Host "=== $Name " -NoNewline
    Write-Host ("=" * [Math]::Max(1, 60 - $Name.Length))
    Push-Location (Join-Path $root $Directory)
    # cargo and pytest write progress to stderr. Under `Stop`, PowerShell turns
    # a native command's stderr into a terminating error and reports a passing
    # suite as failed -- so the exit code is the only thing judged here.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Body
        if ($LASTEXITCODE -ne 0) {
            $script:failed += $Name
            Write-Host "$Name FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        }
    } catch {
        $script:failed += $Name
        Write-Host "$Name FAILED: $_" -ForegroundColor Red
    } finally {
        $ErrorActionPreference = $previous
        Pop-Location
    }
}

# Fail early and clearly rather than reporting a wall of skips: an unreachable
# model server is the one condition under which none of this means anything.
Write-Host "checking $OllamaHost for $Model ..." -NoNewline
try {
    $tags = Invoke-RestMethod -Uri "http://${OllamaHost}:11434/api/tags" -TimeoutSec 8
} catch {
    Write-Host " unreachable" -ForegroundColor Red
    Write-Host "No model server at ${OllamaHost}:11434. Nothing below would be a live test."
    exit 1
}
if ($tags.models.name -notcontains $Model) {
    Write-Host " missing" -ForegroundColor Red
    Write-Host "Model '$Model' is not on that server. Available: $($tags.models.name -join ', ')"
    exit 1
}
Write-Host " ok" -ForegroundColor Green

$env:OLLAMA_HOST = $OllamaHost
$env:KNOSSOS_PYTHON = $Python
$env:KNOSSOS_LIVE = "1"
$env:KNOSSOS_STRICT = "1"
$env:KNOSSOS_LIVE_MODEL = $Model
$env:KNOSSOS_LIVE_OLLAMA = "http://${OllamaHost}:11434"
$env:KNOSSOS_LIVE_STRICT = "1"
$env:LAPCE_ACP_TEST_PYTHON = $Python
$env:LAPCE_ACP_TEST_AGENT_CWD = (Join-Path $root "model")
$env:LAPCE_ACP_TEST_STRICT = "1"

Invoke-Suite "Knossos (Python)" "model" { & $Python -m pytest -q }
Invoke-Suite "ACP conformance + live provider" "conformance" { node run.mjs }
Invoke-Suite "knossos-rs + live engine" "knossos-rs" { cargo test }
Invoke-Suite "Lapce client vs the real agent" "editor" {
    cargo test -p lapce-acp --tests
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "all live suites passed" -ForegroundColor Green
    exit 0
}
Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
exit 1
