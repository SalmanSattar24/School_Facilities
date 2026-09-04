param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or newer is required'; print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 or newer was not found. Install Python from https://www.python.org/downloads/ and rerun this script."
    }
    & py -3 -m venv .venv
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or newer is required'; print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 or newer is required."
    }
    & python -m venv .venv
}
else {
    throw "Python was not found. Install Python 3.11 or newer and rerun this script."
}

if ($LASTEXITCODE -ne 0) {
    throw "Virtual-environment creation failed."
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

& .\.venv\Scripts\python.exe -m pip install -e ".[dev,pilot]"
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

& .\.venv\Scripts\school-facilities.exe validate-config
if ($LASTEXITCODE -ne 0) { throw "Frozen configuration validation failed." }

& .\.venv\Scripts\school-facilities.exe prepare
if ($LASTEXITCODE -ne 0) { throw "Input/template preparation failed." }

if (-not $SkipTests) {
    & .\.venv\Scripts\python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Automated tests failed." }
}

Write-Host ""
Write-Host "Setup completed successfully."
Write-Host "Next: .\.venv\Scripts\school-facilities.exe save-gemini-key"
Write-Host "Then: .\.venv\Scripts\school-facilities.exe save-streetview-key"
Write-Host "Check: .\.venv\Scripts\school-facilities.exe doctor --require-key"
Write-Host "Operator guide: docs\USER_MANUAL.md"
