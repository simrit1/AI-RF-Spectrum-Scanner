@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" diagnose_pluto.py
) else (
  python diagnose_pluto.py
)
echo.
pause
