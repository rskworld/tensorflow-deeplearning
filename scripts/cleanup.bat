@echo off
REM Clean __pycache__ folders and temporary files
REM Author: RSK World - https://rskworld.in

echo ============================================================
echo Cleaning __pycache__ folders and .pyc files
echo Author: RSK World - https://rskworld.in
echo ============================================================
echo.

REM Remove __pycache__ directories
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

REM Remove .pyc, .pyo files
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f" 2>nul
for /r . %%f in (*.pyo) do @if exist "%%f" del /q "%%f" 2>nul

echo Cleanup complete!
echo ============================================================
pause
