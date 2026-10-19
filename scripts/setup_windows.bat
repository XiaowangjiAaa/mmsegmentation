@echo off
rem Convenience wrapper so setup_windows.ps1 can be double-clicked or run
rem from cmd.exe without changing the PowerShell execution policy.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
endlocal
