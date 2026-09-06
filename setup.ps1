<#
.SYNOPSIS
    One-command setup for CallCribe: creates .venv and installs dependencies.

.DESCRIPTION
    Run this once after cloning:

        powershell -ExecutionPolicy Bypass -File setup.ps1

    By default it looks for an NVIDIA GPU and installs the CUDA wheels only
    if it finds one. Override with -Gpu or -Cpu when the guess is wrong, for
    example on a machine whose GPU you deliberately want left alone.

    Nothing here is destructive: an existing .venv is reused, not replaced.

    Kept ASCII on purpose. Windows PowerShell 5.1 reads a .ps1 file using the
    system ANSI codepage unless it starts with a UTF-8 BOM, so a stray em-dash
    or a curly quote turns into a parser error on someone else's machine.

.PARAMETER Gpu
    Install the CUDA 12 wheels regardless of what was detected (~700 MB).

.PARAMETER Cpu
    Skip the CUDA wheels regardless of what was detected.

.PARAMETER Recreate
    Delete an existing .venv and build it from scratch.
#>

[CmdletBinding()]
param(
    [switch]$Gpu,
    [switch]$Cpu,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

if ($Gpu -and $Cpu) {
    throw "-Gpu and -Cpu contradict each other; pass at most one."
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host ""
Write-Host "CallCribe setup" -ForegroundColor Cyan
Write-Host "  folder: $root"

# --- platform -------------------------------------------------------------
# The capture side is WASAPI loopback through PyAudioWPatch, which exists
# only on Windows. Better to say so here than to fail at the first import.
if (-not ($env:OS -eq "Windows_NT")) {
    throw "CallCribe records system audio through WASAPI loopback and runs on Windows only."
}

# --- find an interpreter --------------------------------------------------
# The py launcher first: on Windows it is the one thing that reliably knows
# about every installed version, while `python` may well be the Store stub
# that does nothing but open the Store.
function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("-3.12", "-3.11", "-3.10", "-3")) {
            $candidates += ,@("py", $v)
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += ,@("python")
    }

    foreach ($c in $candidates) {
        $exe = $c[0]
        $prefix = @()
        if ($c.Count -gt 1) { $prefix = @($c[1]) }
        try {
            $out = & $exe @prefix -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null
        } catch {
            continue
        }
        if ($LASTEXITCODE -ne 0 -or -not $out) { continue }

        $parts = $out[0].Split(".")
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        if ($major -eq 3 -and $minor -ge 10) {
            return [pscustomobject]@{
                Exe     = $exe
                Prefix  = $prefix
                Version = $out[0]
                Path    = $out[1]
            }
        }
    }
    return $null
}

if ($Recreate -and (Test-Path $venv)) {
    Write-Host "  removing the existing .venv (-Recreate)" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $python)) {
    $found = Find-Python
    if ($null -eq $found) {
        throw @"
No Python 3.10 or newer found.

Install it from https://www.python.org/downloads/windows/ (tick
"Add python.exe to PATH" in the installer), then run this script again.
3.11 or 3.12 are the safe picks: the newest release often has no
PyAudioWPatch wheel yet, and building that one from source is not fun.
"@
    }
    Write-Host "  python: $($found.Version) - $($found.Path)"
    Write-Host "  creating .venv ..."
    & $found.Exe @($found.Prefix) -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
} else {
    Write-Host "  .venv already exists, reusing it (pass -Recreate to rebuild)"
}

if (-not (Test-Path $python)) {
    throw "Expected $python to exist after creating the environment, but it does not."
}

# --- GPU or CPU -----------------------------------------------------------
$useGpu = $false
if ($Gpu) {
    $useGpu = $true
    Write-Host "  CUDA wheels: yes (-Gpu)"
} elseif ($Cpu) {
    Write-Host "  CUDA wheels: no (-Cpu)"
} else {
    # nvidia-smi ships with the driver, so its presence is a decent proxy
    # for "there is an NVIDIA GPU with a working driver here".
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $useGpu = $true
        Write-Host "  CUDA wheels: yes (NVIDIA driver detected)"
    } else {
        Write-Host "  CUDA wheels: no (no NVIDIA driver detected; pass -Gpu to force)"
    }
}

$reqs = "requirements.txt"
if ($useGpu) { $reqs = "requirements-gpu.txt" }

Write-Host ""
Write-Host "Installing $reqs ..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $python -m pip install -r (Join-Path $root $reqs)
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See the output above." }

# --- report ---------------------------------------------------------------
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Next:"
Write-Host "  .\run.cmd                            start it (console doubles as the log)"
Write-Host "  .\run.cmd --ui-lang ru               start with a Russian interface"
Write-Host "  .\install-shortcut.ps1               desktop shortcut, no console window"
Write-Host "  .venv\Scripts\python selftest.py     check the install without a call"
Write-Host ""
Write-Host "The first start downloads the speech model (large-v3-turbo on the CPU," -ForegroundColor Yellow
Write-Host "large-v3 on a GPU) from Hugging Face. That one download needs internet;" -ForegroundColor Yellow
Write-Host "everything after it is local. Use run.cmd for it so you can watch progress." -ForegroundColor Yellow
Write-Host ""
