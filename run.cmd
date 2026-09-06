@echo off
rem Console launcher - the console window doubles as the log.
rem For everyday use prefer the desktop shortcut (install-shortcut.ps1),
rem which runs without a console. Use this file when something breaks
rem and you need to read the diagnostics, and for the very first start:
rem the model download prints its progress here.
rem
rem Kept ASCII on purpose: cmd.exe parses a .cmd file using the current
rem console codepage, so non-ASCII text here turns into broken commands.
rem The app's own Russian output is unaffected - Python writes to the
rem console through WriteConsoleW, independent of the codepage.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo No virtual environment found in this folder.
    echo Run setup once:
    echo.
    echo     powershell -ExecutionPolicy Bypass -File setup.ps1
    echo.
    pause
    exit /b 1
)

rem %* forwards arguments through, so `run.cmd --lang en` works.
".venv\Scripts\python.exe" -m callcribe %*
set CODE=%ERRORLEVEL%

if not "%CODE%"=="0" (
    echo.
    echo CallCribe exited with code %CODE% - see the lines above.
    pause
)
