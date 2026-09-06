<#
.SYNOPSIS
    Puts a CallCribe shortcut on the desktop.

.DESCRIPTION
        powershell -ExecutionPolicy Bypass -File install-shortcut.ps1

    The shortcut launches pythonw.exe, so there is no console window. That is
    deliberate: a console sitting next to the transcript window is easy to
    close by accident, and closing it kills the process hard (CTranslate2
    carries an Intel runtime with its own console-close handler). With the
    shortcut there is exactly one window, and closing it shuts down cleanly.

    The cost is that diagnostics go nowhere. Run run.cmd when you need them.

    Kept ASCII on purpose: Windows PowerShell 5.1 reads a .ps1 file using the
    system ANSI codepage unless it starts with a UTF-8 BOM, so a stray dash of
    the wrong kind becomes a parser error on someone else's machine.
#>

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $pythonw)) {
    throw @"
$pythonw not found.

Set the environment up first:

    powershell -ExecutionPolicy Bypass -File setup.ps1
"@
}

$desktop = [Environment]::GetFolderPath("Desktop")
$linkPath = Join-Path $desktop "CallCribe.lnk"

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath = $pythonw
$link.Arguments = "-m callcribe"
$link.WorkingDirectory = $root
$link.Description = "Live call transcription, entirely on this machine"
$link.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
$link.Save()

Write-Output "Shortcut created: $linkPath"
Write-Output "  runs: $pythonw -m callcribe"
Write-Output "  working folder: $root"
