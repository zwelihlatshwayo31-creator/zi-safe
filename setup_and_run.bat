@echo off
title Zi Safe
REM ======================
REM Zi Safe (SerpAPI-only, fixed)
REM One-shot setup & run (Windows)
REM ======================

SETLOCAL

IF NOT EXIST ".env" (
  echo.
  echo [!] .env file not found. Creating a template...
  echo SERPAPI_KEY=your_serpapi_key_here> .env
  echo Created .env. Please edit it to add your SERPAPI key.
)

python --version >NUL 2>&1
IF ERRORLEVEL 1 (
  echo.
  echo [!] Python is not installed or not on PATH.
  echo     Install Python 3.x from https://www.python.org/downloads/windows/
  echo     and check "Add Python to PATH" during installation.
  pause
  EXIT /B 1
)

python -m venv venv
IF ERRORLEVEL 1 (
  echo [!] Failed to create virtual environment.
  pause
  EXIT /B 1
)

call venv\Scripts\activate
IF ERRORLEVEL 1 (
  echo [!] Failed to activate virtual environment.
  pause
  EXIT /B 1
)

python -m pip install --upgrade pip
IF ERRORLEVEL 1 (
  echo [!] Failed to upgrade pip.
  pause
  EXIT /B 1
)

pip install streamlit requests pdfminer.six beautifulsoup4 lxml python-dotenv pydantic rapidfuzz
IF ERRORLEVEL 1 (
  echo [!] Failed to install dependencies.
  pause
  EXIT /B 1
)

echo.
echo Starting Zi Safe...
streamlit run app.py

echo.
echo Press any key to close this window.
pause >NUL
ENDLOCAL
