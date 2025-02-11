@echo off
setlocal

:: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python and try again.
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate


:: Install dependencies
echo Installing dependencies...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt


:: Done
echo.
echo Environment successfully set up!

exit /b 0