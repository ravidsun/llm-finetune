@echo off
REM Merge LoRA Adapter with Base Model - Windows Batch Script
REM Simple wrapper for merge_local.py

echo ========================================
echo Merge LoRA Adapter (Local Windows)
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Run the merge script
python scripts\merge_local.py %*

pause
