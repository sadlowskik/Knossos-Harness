@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-reward-eval.ps1" %*
exit /b %ERRORLEVEL%
