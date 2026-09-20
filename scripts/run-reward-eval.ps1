[CmdletBinding()]
param(
    [ValidateSet("ollama", "cameo", "anthropic", "openai", "groq", "gemini", "openrouter", "qwen")]
    [string]$Engine = "ollama",
    [string]$Model = "",
    [string]$Provider = "",
    [string]$BaseUrl = "",
    [string]$Knossos = "",
    [int]$MaxSteps = 12,
    [int]$TargetSteps = 6,
    [UInt64]$MaxRequests = 128,
    [UInt64]$MaxTotalTokens = 262144,
    [int]$MaxTokens = 4096,
    [int]$NumCtx = 32768,
    [switch]$Resume,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$crate = Join-Path $repository "knossos-rs"
$cases = Join-Path $crate "cases\reward-hacking.json"
$reports = Join-Path $repository "reports\reward-hacking"
New-Item -ItemType Directory -Force -Path $reports | Out-Null

Write-Host "Auditing frozen evaluation fixtures..." -ForegroundColor Cyan
$modelDirectory = Join-Path $repository "model"
Push-Location $modelDirectory
try {
    & python scripts\audit_suites.py --lock ..\knossos-rs\cases\suite-lock.json
    if ($LASTEXITCODE -ne 0) {
        throw "suite audit failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if ([string]::IsNullOrWhiteSpace($Knossos)) {
    $Knossos = Join-Path $crate "target\release\knossos.exe"
}
if (-not (Test-Path -LiteralPath $Knossos -PathType Leaf)) {
    if ($NoBuild) {
        throw "Knossos binary not found at $Knossos and -NoBuild was supplied."
    }
    Write-Host "Building Knossos release binary..." -ForegroundColor Cyan
    & cargo build --release --locked --manifest-path (Join-Path $crate "Cargo.toml") --bin knossos
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
}
$Knossos = (Resolve-Path -LiteralPath $Knossos).Path

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirectory = Join-Path $reports $stamp
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null

$globalArguments = @("--engine", $Engine, "--workspace", $repository)
if (-not [string]::IsNullOrWhiteSpace($Model)) { $globalArguments += @("--model", $Model) }
if (-not [string]::IsNullOrWhiteSpace($Provider)) { $globalArguments += @("--provider", $Provider) }
if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) { $globalArguments += @("--base-url", $BaseUrl) }

$common = @(
    "eval", "--cases", $cases,
    "--max-steps", "$MaxSteps",
    "--target-steps", "$TargetSteps",
    "--max-requests", "$MaxRequests",
    "--max-total-tokens", "$MaxTotalTokens",
    "--max-tokens", "$MaxTokens",
    "--num-ctx", "$NumCtx",
    "--max-concurrency", "1",
    "--no-judge",
    "--collect-exchanges"
)

function Invoke-Arm {
    param(
        [string]$Name,
        [string[]]$ExtraArguments
    )
    $checkpoint = Join-Path $reports "$Name.checkpoint.json"
    $trace = Join-Path $runDirectory "$Name.jsonl"
    $log = Join-Path $runDirectory "$Name.log"
    $arguments = $globalArguments + $common + @(
        "--experiment-arm", $Name,
        "--checkpoint", $checkpoint,
        "--trace", $trace
    ) + $ExtraArguments
    if ($Resume) { $arguments += "--resume" }

    Write-Host "Running $Name..." -ForegroundColor Cyan
    $output = @(& $Knossos @arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $output | Tee-Object -FilePath $log | Out-Host
    $score = $output | Select-String -Pattern "^(\d+)/(\d+) passed$" | Select-Object -Last 1
    $passed = $null
    $total = $null
    if ($score) {
        $passed = [int]$score.Matches[0].Groups[1].Value
        $total = [int]$score.Matches[0].Groups[2].Value
    }
    [pscustomobject]@{
        arm = $Name
        exit_code = $exitCode
        passed = $passed
        total = $total
        checkpoint = $checkpoint
        trace_prefix = $trace
        log = $log
    }
}

$full = Invoke-Arm -Name "knossos-full" -ExtraArguments @()
$control = Invoke-Arm -Name "knossos-no-context" -ExtraArguments @("--no-context")
$summary = [pscustomobject]@{
    schema = "knossos-reward-eval-result/v1"
    started_at = $stamp
    engine = $Engine
    model = $Model
    provider = $Provider
    base_url = $BaseUrl
    cases = $cases
    max_steps = $MaxSteps
    target_steps = $TargetSteps
    max_requests_per_arm = $MaxRequests
    max_total_tokens_per_arm = $MaxTotalTokens
    max_tokens_per_turn = $MaxTokens
    num_ctx = $NumCtx
    arms = @($full, $control)
}
$summaryPath = Join-Path $runDirectory "summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Host "Comparison written to $summaryPath" -ForegroundColor Green

if ($full.exit_code -ne 0 -or $control.exit_code -ne 0) {
    Write-Warning "At least one arm did not pass every case. This is an evaluation result, not permission to weaken the suite."
    exit 1
}
