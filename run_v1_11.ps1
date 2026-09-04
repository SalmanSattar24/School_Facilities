param(
    [switch]$ConfirmLive,
    [switch]$ConfirmProviderQuota
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = Join-Path $projectRoot ".venv\Scripts\school-facilities.exe"

if (-not $ConfirmLive) {
    throw "Live Street View and Gemini execution requires -ConfirmLive."
}
if (-not $ConfirmProviderQuota) {
    throw "Confirm that the provider quota is 10 requests/minute or lower with -ConfirmProviderQuota."
}
if (-not (Test-Path -LiteralPath $cli)) {
    throw "The virtual environment is missing. Run: powershell -ExecutionPolicy Bypass -File .\setup.ps1"
}

$manifestRoot = Join-Path $projectRoot "data\streetview\manifests\v1.11"
$probePlan = Join-Path $manifestRoot "all25_terminal_probe_plan.json"
$imageManifest = Join-Path $manifestRoot "all25_terminal_image_manifest.json"
$deduplicatedManifest = Join-Path $manifestRoot "all25_terminal_image_manifest_deduplicated.json"
$fetchedManifest = Join-Path $manifestRoot "all25_terminal_image_manifest_deduplicated_fetched.json"

& $cli validate-config
if ($LASTEXITCODE -ne 0) { throw "Configuration validation failed." }

& $cli streetview-plan --all --output $probePlan
if ($LASTEXITCODE -ne 0) { throw "Street View planning failed." }

& $cli streetview-probe --manifest $probePlan --metadata-only --output $imageManifest
if ($LASTEXITCODE -ne 0) { throw "Street View metadata discovery failed." }

& $cli streetview-deduplicate-manifest --manifest $imageManifest --output $deduplicatedManifest
if ($LASTEXITCODE -ne 0) { throw "Street View request deduplication failed." }

& $cli streetview-fetch --manifest $deduplicatedManifest --live --max-paid-usd 0 --confirm-provider-quota
if ($LASTEXITCODE -ne 0) { throw "Street View retrieval stopped. Inspect the preserved ledger and fetched manifest before resuming." }

& $cli assess-v1-11-batch --manifest $fetchedManifest --live --confirm-gemini-v1-11
if ($LASTEXITCODE -ne 0) { throw "One or more V1.11 assessments failed. Reconcile preserved rejected outputs offline before continuing." }

& $cli run-all --snapshot (Join-Path $projectRoot "outputs\all25_v1_11_predictions_frozen.json") --report (Join-Path $projectRoot "outputs\full_pipeline_v1_11_evaluation.json") --measurements (Join-Path $projectRoot "measurements.csv")
if ($LASTEXITCODE -ne 0) { throw "Final fusion/evaluation failed." }

Write-Host "V1.11 completed. Review measurements.csv and every needs-review field before submission."
