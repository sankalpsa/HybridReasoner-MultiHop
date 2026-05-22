@echo off
title Hybrid Neural-Symbolic Kinship Reasoner Launcher
color 0B

echo =======================================================================
echo     HYBRID NEURAL-SYMBOLIC KINSHIP REASONER - ONE-CLICK LAUNCHER
echo =======================================================================
echo.

:: 1. Forcefully release port 5001 if it's currently bound
echo [1/4] Checking and freeing port 5001...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5001" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo Found process running on port 5001 with PID %%a. Killing it...
        taskkill /F /PID %%a
    )
)
echo Port 5001 is clean!
echo.

:: 2. Set directory path to the active workspace
echo [2/4] Navigating to Hybrid Reasoner workspace...
cd /d "%~dp0"
if %errorlevel% neq 0 (
    echo [ERROR] Could not find the project directory: "%~dp0"
    pause
    exit /b
)

:: 3. Detect and activate functional virtual environments
echo [3/4] Detecting functional virtual environment...
set "VENV_ACTIVE=0"

if not exist ".venv\Scripts\python.exe" goto check_venv2
".venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 goto venv1_fail
echo Activating virtual environment in .venv...
call .venv\Scripts\activate.bat
set "VENV_ACTIVE=1"
goto detect_done

:venv1_fail
echo [WARNING] .venv is cloud-dehydrated or invalid. Skipping...

:check_venv2
if not exist "venv\Scripts\python.exe" goto detect_done
"venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 goto venv2_fail
echo Activating virtual environment in venv...
call venv\Scripts\activate.bat
set "VENV_ACTIVE=1"
goto detect_done

:venv2_fail
echo [WARNING] venv is cloud-dehydrated or invalid. Skipping...

:detect_done
if "%VENV_ACTIVE%"=="0" (
    echo [INFO] Using standard system Python directly - recommended for local hydration.
)
echo.

:: Check python executable availability
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in your system PATH!
    echo Please install Python and ensure it is added to your PATH environment variable.
    pause
    exit /b
)

:: 4. Start the backend server and open the browser
echo [4/4] Starting the Flask server on port 5001...
echo Launching the web browser in 3 seconds...

:: Using robust ping command to delay and open the browser (bypasses input redirection bugs in 'timeout')
start "" cmd /c "ping 127.0.0.1 -n 4 >nul && start http://127.0.0.1:5001/"

echo.
echo =======================================================================
echo   SERVER IS STARTING!
echo   Keep this command window open to keep the backend running.
echo   To stop the server, press Ctrl+C or simply close this window.
echo =======================================================================
echo.

python app.py

echo.
echo =======================================================================
echo   Server has stopped.
echo =======================================================================
echo.
pause
