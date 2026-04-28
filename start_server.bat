@echo off
REM Start Flask server with hand detection enabled
REM Skip loading Florence model to avoid CUDA memory issues
cd /d e:\ai_vision
setlocal enabledelayedexpansion
set SKIP_MODEL_LOAD=1
D:\anaconda3\envs\vision\python.exe aiagent.py
