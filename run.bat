@echo off
rem CareerCompass launcher: double-click to set up (first run only) and start the app.
title CareerCompass
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11+ from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating the virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :fail
)

if not exist ".venv\.installed" (
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 goto :fail
  echo ok> ".venv\.installed"
)

echo.
echo Starting CareerCompass at http://127.0.0.1:5000
echo Close this window (or press Ctrl+C) to stop the server.
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo Something went wrong. Check the messages above.
pause
exit /b 1
