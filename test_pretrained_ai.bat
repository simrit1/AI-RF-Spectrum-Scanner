@echo off
setlocal
cd /d "%~dp0"
echo Testing public pretrained LWM-Spectro backend...
python test_pretrained_ai.py
pause
