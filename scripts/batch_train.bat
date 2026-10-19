@echo off
rem Run the crack batch trainer with the project's .venv-mmseg environment.
setlocal
set "REPO=%~dp0.."
if exist "%REPO%\.venv-mmseg\python.exe" (
    "%REPO%\.venv-mmseg\python.exe" "%~dp0batch_train.py" %*
) else (
    python "%~dp0batch_train.py" %*
)
endlocal
