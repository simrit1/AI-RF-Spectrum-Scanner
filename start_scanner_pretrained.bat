@echo off
setlocal
cd /d "%~dp0"
echo Starting ADALM-Pluto House RF Scanner with public pretrained LWM-Spectro AI...
python house_rf_scanner.py --cnn-model hf:lwm-spectro
if errorlevel 1 pause
