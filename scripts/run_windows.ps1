<#
================================================================
  run_windows.ps1  -  Windows runner for mlops-forecast
================================================================
  Windows equivalent of the Makefile (which targets Mac/Linux and
  does not work on Windows). It runs, in order:

    1. create the virtual env            (.venv)
    2. install the project               (pip install -e ".[dev]")
    3. download the original OPSD dataset into data/01_raw/
    4. run the full Kedro pipeline end-to-end (correct order)
    5. publish features into the Feast feature store
    6. run the test suite (pytest)
    7. run the code-quality checks (ruff / mypy / kedro registry)
    8. (optional) bring up the serving stack via docker compose

  HOW TO USE - from PowerShell, inside the project folder:

      cd "C:\path\to\mlops-forecast"
      powershell -ExecutionPolicy Bypass -File .\run_windows.ps1

  Optional flags:
      -SkipTests      do not run pytest
      -SkipFeast      do not publish to / smoke-test Feast
      -SkipChecks     do not run ruff / mypy / kedro registry list
      -OnlyData       download the dataset only, do not run anything else
      -Recreate       rebuild the venv from scratch
      -Serve          after everything, start the Docker serving stack
                      (MLflow + FastAPI + Streamlit). Requires Docker
                      Desktop running. Off by default.

  Note: the download is ~124 MB and happens only once. If the file
  already exists it is skipped.
================================================================
#>

param(
    [switch]$SkipTests,
    [switch]$SkipFeast,
    [switch]$SkipChecks,
    [switch]$OnlyData,
    [switch]$Recreate,
    [switch]$Serve
)

# We do NOT use "Stop": pip/kedro frequently write warnings to stderr and with
# "Stop" PowerShell would abort the script on the first message. Instead we
# check the outcome of every command via $LASTEXITCODE.
$ErrorActionPreference = "Continue"

# --- Go to the repo root (this script lives in scripts\, root is its parent) ---
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Disable Kedro telemetry. We do NOT set PYTHONWARNINGS here: that filter
# polluted venv creation ("Invalid -W option"). On Python 3.11 the Kedro
# version warning does not appear anyway.
$env:KEDRO_DISABLE_TELEMETRY = "1"

# OPSD dataset URL PINNED to a dated release (reproducible). The original code
# uses ".../latest/..." which is a moving target: here we use the 2020-10-06
# snapshot (the last OPSD release, covers 2015-2020).
$DataUrl  = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
$DataDest = "data\01_raw\opsd_germany_hourly.csv"

# Track non-fatal problems found by the quality checks so we can summarise
# them at the end without aborting the run.
$script:CheckWarnings = @()

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# Run a quality check that should report problems but NOT stop the script.
function Invoke-Check($label, $scriptblock) {
    Write-Host "`n--- $label ---" -ForegroundColor Yellow
    & $scriptblock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN  $label reported issues (exit $LASTEXITCODE) - continuing." -ForegroundColor DarkYellow
        $script:CheckWarnings += $label
    } else {
        Write-Host "OK    $label passed." -ForegroundColor Green
    }
}

# --- 0. Find Python ------------------------------------------------------
Write-Step "Checking Python"
$pythonCmd = $null
foreach ($c in @("python", "py -3.11", "py -3")) {
    try {
        $v = & cmd /c "$c --version" 2>&1
        if ($LASTEXITCODE -eq 0) { $pythonCmd = $c; Write-Host "Found: $c -> $v"; break }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found. Install Python 3.11 from python.org and retry." -ForegroundColor Red
    exit 1
}

# --- 1. Virtual env -------------------------------------------------------
Write-Step "Virtual env (.venv)"
$VPy    = ".\.venv\Scripts\python.exe"
$VPipEx = ".\.venv\Scripts\pip.exe"

if ($Recreate -and (Test-Path ".venv")) {
    Write-Host "Removing existing venv (-Recreate)..."
    Remove-Item -Recurse -Force ".venv"
}

# A venv interrupted halfway (e.g. Ctrl+C during creation) is left without pip.
# We detect this by the absence of pip.exe: in that case we throw the folder
# away and recreate it. We do NOT run the broken python to test it (that would
# write to stderr and confuse the output).
if ((Test-Path ".venv") -and -not (Test-Path $VPipEx)) {
    Write-Host "Existing venv is incomplete (pip missing): recreating it."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path $VPy)) {
    Write-Host "Creating the virtual env (may take ~1 min, do not interrupt)..."
    & cmd /c "$pythonCmd -m venv .venv"
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR creating venv" -ForegroundColor Red; exit 1 }
}

# If pip.exe is still missing, force its installation via ensurepip.
if (-not (Test-Path $VPipEx)) {
    Write-Host "Installing pip into the venv (ensurepip)..."
    & $VPy -m ensurepip --upgrade
}

# Final check via file (not by running python): pip.exe must exist.
if (-not (Test-Path $VPipEx)) {
    Write-Host "ERROR: pip not available in the venv. Re-run with -Recreate." -ForegroundColor Red
    exit 1
}

# Direct paths to the venv executables (no Activate.ps1 -> no ExecutionPolicy
# problems).
$VKedro  = ".\.venv\Scripts\kedro.exe"
$VPytest = ".\.venv\Scripts\pytest.exe"
$VRuff   = ".\.venv\Scripts\ruff.exe"
$VMypy   = ".\.venv\Scripts\mypy.exe"

# --- 2. Installation ------------------------------------------------------
# We always use "python -m pip" (not pip.exe): it works even right after
# ensurepip, when pip.exe may not have been generated yet.
Write-Step "Installing dependencies (pip install -e .[dev])"
& $VPy -m pip install --upgrade pip setuptools wheel
& $VPy -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR during installation" -ForegroundColor Red; exit 1 }

# --- 3. Download dataset --------------------------------------------------
Write-Step "Original OPSD dataset"
New-Item -ItemType Directory -Force -Path "data\01_raw" | Out-Null
if (Test-Path $DataDest) {
    $mb = [math]::Round((Get-Item $DataDest).Length / 1MB, 1)
    Write-Host "Already present: $DataDest ($mb MB). Delete it to re-download."
} else {
    Write-Host "Downloading ~124 MB from:`n  $DataUrl"
    $ProgressPreference = "SilentlyContinue"   # much faster download
    Invoke-WebRequest -Uri $DataUrl -OutFile $DataDest
    $mb = [math]::Round((Get-Item $DataDest).Length / 1MB, 1)
    Write-Host "Downloaded $mb MB -> $DataDest"
}

if ($OnlyData) { Write-Host "`n-OnlyData: stopping here." -ForegroundColor Green; exit 0 }

# --- 4. End-to-end pipeline ----------------------------------------------
# We do NOT use a single "kedro run": the prediction node loads the model from
# MLflow via a string (models:/ElectricityForecast/Production) and does NOT
# declare it as a Kedro dependency. So in the DAG the prediction can be
# scheduled BEFORE training -> "Registered Model ... not found".
# We run the pipelines in the correct order, using the groups already defined
# in pipeline_registry.py: ingest -> prepare -> train_and_select -> inference.
Write-Step "Kedro pipelines (correct order)"
$stages = @("ingest", "prepare", "train_and_select", "inference")
foreach ($s in $stages) {
    Write-Host "`n--- kedro run --pipeline $s ---" -ForegroundColor Yellow
    & $VKedro run --pipeline $s
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR in pipeline '$s'" -ForegroundColor Red
        exit 1
    }
}

# --- 5. Feast feature store ----------------------------------------------
# Publish the engineered features into Feast (feast apply + materialize) and
# run the smoke test. The publish step is a core deliverable, so a failure
# aborts; the demo is just a retrieval smoke test, so it is non-fatal.
if (-not $SkipFeast) {
    Write-Step "Feast feature store (publish)"
    & $VPy scripts\publish_to_feast.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR while publishing to Feast" -ForegroundColor Red
        exit 1
    }

    Write-Step "Feast feature store (smoke test)"
    & $VPy scripts\feast_demo.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN  Feast smoke test reported issues - continuing." -ForegroundColor DarkYellow
        $script:CheckWarnings += "feast_demo"
    }
}

# --- 6. Tests -------------------------------------------------------------
if (-not $SkipTests) {
    Write-Step "Tests (pytest)"
    & $VPytest tests/ -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: the test suite failed" -ForegroundColor Red
        exit 1
    }
}

# --- 7. Code-quality checks (non-blocking) -------------------------------
# These mirror the CI gate (ruff + mypy + kedro registry list). They REPORT
# problems but do not abort the run, so you see everything in one pass.
if (-not $SkipChecks) {
    Write-Step "Code-quality checks (ruff / mypy / kedro registry)"
    Invoke-Check "ruff check"        { & $VRuff check src/ tests/ api/ streamlit_app/ }
    Invoke-Check "ruff format check" { & $VRuff format --check src/ tests/ api/ streamlit_app/ }
    Invoke-Check "mypy"              { & $VMypy src/mlops_forecast/ }
    Invoke-Check "kedro registry"    { & $VKedro registry list }
}

# --- 8. Serving stack (opt-in) -------------------------------------------
if ($Serve) {
    Write-Step "Serving stack (docker compose)"
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-Host "WARN  Docker not found. Install Docker Desktop to use -Serve. Skipping." -ForegroundColor DarkYellow
    } else {
        docker compose up --build -d
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARN  docker compose failed (is Docker Desktop running?)." -ForegroundColor DarkYellow
        } else {
            Write-Host "MLflow UI:    http://localhost:5000"  -ForegroundColor Green
            Write-Host "FastAPI docs: http://localhost:8000/docs" -ForegroundColor Green
            Write-Host "Streamlit:    http://localhost:8501"  -ForegroundColor Green
        }
    }
}

# --- Done -----------------------------------------------------------------
Write-Step "DONE"
Write-Host "Pipeline complete. Output in data\07_model_output and data\08_reporting." -ForegroundColor Green
if ($script:CheckWarnings.Count -gt 0) {
    Write-Host ("Non-blocking warnings from: " + ($script:CheckWarnings -join ", ")) -ForegroundColor DarkYellow
}
if (-not $Serve) {
    Write-Host "To start the serving stack:  powershell -ExecutionPolicy Bypass -File .\run_windows.ps1 -Serve"
}
Write-Host "For the MLflow UI:  .\.venv\Scripts\mlflow.exe ui --backend-store-uri ./mlruns"
