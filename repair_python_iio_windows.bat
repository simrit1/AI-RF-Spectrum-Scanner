@echo off
setlocal
cd /d "%~dp0"
echo This repairs the Python bindings. It does NOT install the native LibIIO Windows DLL/package.
echo If diagnose_pluto.py still fails after this, install the native Analog Devices LibIIO Windows release first.
echo.
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
%PY% -m pip install --upgrade pip
%PY% -m pip install --upgrade --force-reinstall pylibiio pyadi-iio
%PY% diagnose_pluto.py
echo.
pause
