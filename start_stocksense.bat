@echo off
setlocal
REM Start Stocksense backend using virtualenv if available, otherwise system python
cd /d "%~dp0"
SET "ROOT_DIR=%~dp0"
SET "LOGFILE=%ROOT_DIR%stocksense_backend.log"

IF EXIST "%ROOT_DIR%New folder\flask_backend.py" (
  SET "BACKEND_SCRIPT=%ROOT_DIR%New folder\flask_backend.py"
) ELSE (
  SET "BACKEND_SCRIPT=%ROOT_DIR%flask_backend.py"
)

IF EXIST "%ROOT_DIR%venv\Scripts\python.exe" (
  "%ROOT_DIR%venv\Scripts\python.exe" "%BACKEND_SCRIPT%" >> "%LOGFILE%" 2>&1
) ELSE IF EXIST "%ROOT_DIR%.venv\Scripts\python.exe" (
  "%ROOT_DIR%.venv\Scripts\python.exe" "%BACKEND_SCRIPT%" >> "%LOGFILE%" 2>&1
) ELSE (
  python "%BACKEND_SCRIPT%" >> "%LOGFILE%" 2>&1
)

endlocal
