@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe house_rf_scanner.py
) else (
  python house_rf_scanner.py
)
pause
