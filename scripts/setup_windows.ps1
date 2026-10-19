<#
.SYNOPSIS
One-click environment setup for MMSegmentation + EfficientViT on Windows.

.DESCRIPTION
Creates an isolated conda environment (or a venv when conda is unavailable),
installs a known-good PyTorch 2.1.0 / MMCV 2.1.0 / MMEngine 0.10.7 stack,
installs this repository in editable mode and verifies the EfficientViT model.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

.EXAMPLE
# CPU-only machine
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Cpu

.EXAMPLE
# Recreate the environment from scratch
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Force

.NOTES
Requires conda (recommended) or Python 3.10/3.11 on PATH.
#>
[CmdletBinding()]
param(
    [ValidateSet('cu118', 'cpu')]
    [string]$Device = 'cu118',
    [switch]$Cpu,
    [string]$EnvDir = '.venv-mmseg',
    [switch]$Force,
    [switch]$WithTests,
    [switch]$SkipVerify,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONNOUSERSITE = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

if ($Cpu) {
    $Device = 'cpu'
}

$absoluteEnv = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $EnvDir))
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'setup.py'))) {
    throw "setup.py not found under $RepoRoot; run this script from the " +
        'mmsegmentation repository.'
}
if ($absoluteEnv -eq [System.IO.Path]::GetFullPath($RepoRoot)) {
    throw 'The environment directory must not be the repository root.'
}
$isInsideRepo = $absoluteEnv.StartsWith(
    $RepoRoot + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase)

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "==> $Description" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host ("    `"$Python`" " + ($Arguments -join ' '))
        return
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)"
    }
}

Write-Host "Repository : $RepoRoot"
Write-Host "Environment: $absoluteEnv"
Write-Host "Device     : $Device"
Write-Host ""

# ---------------------------------------------------------------- environment
$conda = Get-Command conda -ErrorAction SilentlyContinue
$usedConda = $null -ne $conda

if ($Force -and (Test-Path -LiteralPath $absoluteEnv)) {
    if (-not $isInsideRepo) {
        throw "Refusing to delete an environment outside the repository: " +
            $absoluteEnv
    }
    Write-Host '==> Remove existing environment' -ForegroundColor Cyan
    if ($usedConda) {
        if ($DryRun) {
            Write-Host "    conda env remove -y -p `"$absoluteEnv`""
        }
        else {
            & conda env remove -y -p $absoluteEnv 2>$null
        }
    }
    if (-not $DryRun -and (Test-Path -LiteralPath $absoluteEnv)) {
        Remove-Item -LiteralPath $absoluteEnv -Recurse -Force
    }
}

if ($DryRun) {
    if ($usedConda) {
        Write-Host "    conda create -y -p `"$absoluteEnv`" python=3.10"
        $Python = Join-Path $absoluteEnv 'python.exe'
    }
    else {
        Write-Host "    py -3.10 -m venv `"$absoluteEnv`""
        $Python = Join-Path $absoluteEnv 'Scripts\python.exe'
    }
}
elseif ($usedConda) {
    if (-not (Test-Path -LiteralPath (Join-Path $absoluteEnv 'python.exe'))) {
        Write-Host '==> Create conda environment (python 3.10)' `
            -ForegroundColor Cyan
        & conda create -y -p $absoluteEnv python=3.10
        if ($LASTEXITCODE -ne 0) {
            throw 'conda create failed. Accept the Anaconda Terms of Service ' +
                "or create the environment manually at $absoluteEnv."
        }
    }
    $Python = Join-Path $absoluteEnv 'python.exe'
}
else {
    $Python = Join-Path $absoluteEnv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        Write-Host '==> Create venv (conda not found)' -ForegroundColor Yellow
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -ne $pyLauncher) {
            & py -3.10 -m venv $absoluteEnv
        }
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $Python)) {
            & python -m venv $absoluteEnv
        }
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $Python)) {
            throw 'Could not create a Python environment. Install Miniconda ' +
                '(recommended) or Python 3.10 and retry.'
        }
    }
}

if (-not $DryRun) {
    $pyVersion = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($pyVersion -notin @('3.10', '3.11')) {
        throw "Python $pyVersion found, but 3.10 or 3.11 is required for " +
            'the pinned MMCV wheels. Use conda or install Python 3.10.'
    }
}

# ------------------------------------------------------------------ packages
Invoke-Python 'Upgrade pip / setuptools / wheel' @(
    '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools<81')

Invoke-Python 'Install numpy / opencv (pinned)' @(
    '-m', 'pip', 'install', 'numpy==1.26.4',
    'opencv-python==4.8.1.78', '--only-binary=opencv-python')

if ($Device -eq 'cpu') {
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
    $mmcvIndex = 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html'
}
else {
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
    $mmcvIndex = 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html'
}

Invoke-Python "Install PyTorch 2.1.0 ($Device)" @(
    '-m', 'pip', 'install', 'torch==2.1.0', 'torchvision==0.16.0',
    '--index-url', $torchIndex)

Invoke-Python 'Install MMEngine 0.10.7' @(
    '-m', 'pip', 'install', 'mmengine==0.10.7')

Invoke-Python 'Install MMCV 2.1.0 (prebuilt wheel)' @(
    '-m', 'pip', 'install', 'mmcv==2.1.0', '--only-binary=mmcv',
    '-f', $mmcvIndex)

Invoke-Python 'Install MMDetection / MMPretrain (ConvNeXt, Mask2Former)' @(
    '-m', 'pip', 'install', 'mmdet==3.3.0', 'mmpretrain==1.2.0')

Invoke-Python 'Install MMSEG runtime requirements' @(
    '-m', 'pip', 'install', '-r', 'requirements/runtime.txt')

if ($WithTests) {
    Invoke-Python 'Install test requirements' @(
        '-m', 'pip', 'install', '-r', 'requirements/tests.txt')
}

Invoke-Python 'Install this repository (editable, no dependency changes)' @(
    '-m', 'pip', 'install', '-e', '.', '--no-deps', '--no-build-isolation')

# ------------------------------------------------------------------- verify
if (-not $SkipVerify) {
    if ($DryRun) {
        Write-Host "==> Verify environment"
        Write-Host "    `"$Python`" tools/verify_env.py"
    }
    else {
        Write-Host '==> Verify environment' -ForegroundColor Cyan
        & $Python tools/verify_env.py
        if ($LASTEXITCODE -ne 0) {
            throw 'Environment verification failed; see the output above.'
        }
    }
}

Write-Host ''
Write-Host "Environment ready: $absoluteEnv" -ForegroundColor Green
if ($usedConda) {
    Write-Host "Activate with: conda activate `"$absoluteEnv`""
}
else {
    Write-Host "Activate with: & `"$absoluteEnv\Scripts\Activate.ps1`""
}
Write-Host 'Train:'
Write-Host '  python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py'
