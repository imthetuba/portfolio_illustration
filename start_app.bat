@echo off
setlocal

REM Check if python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python not found. Downloading and installing Python...
    powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe -OutFile python-installer.exe"
    start /wait python-installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del python-installer.exe
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%PATH%"
)

REM Upgrade pip
python -m ensurepip --upgrade
python -m pip install --upgrade pip

REM Install Streamlit and requirements
python -m pip install streamlit
python -m pip install -r requirements.txt

REM Run the Streamlit app
streamlit run app.py

pause