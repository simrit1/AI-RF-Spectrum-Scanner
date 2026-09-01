@echo off
cd /d "%~dp0"
echo Testing Wi-Fi out-of-distribution rejection gate...
python test_ood_gate.py
pause
