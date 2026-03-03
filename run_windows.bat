@echo off
SETLOCAL
TITLE FRAS - Face Recognition Attendance System

echo ======================================================
echo   Face Recognition Attendance System (Windows)
echo ======================================================

:: 1. Check if virtual environment exists
if not exist ".venv" (
    echo [INFO] Virtual environment not found. Initializing setup...
    echo [IMPORTANT] Ensure you have Visual Studio C++ Build Tools installed!
    python setup_env.py
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Setup failed. Please check your Python installation and Build Tools.
        pause
        exit /b %ERRORLEVEL%
    )
)

:: 2. Set the PYTHONPATH so 'utils' can be imported correctly
:: %CD% is the current directory in Windows CMD
set PYTHONPATH=%PYTHONPATH%;%CD%\face_attendance

:: 3. Launch the Flask App
echo [INFO] Starting the Web Dashboard...
echo [INFO] Access the app at http://localhost:5000
.venv\Scripts\python face_attendance\app.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application crashed. See above for details.
    pause
)

ENDLOCAL
