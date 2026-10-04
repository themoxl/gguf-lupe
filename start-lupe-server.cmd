@echo off
rem Start the GGUF-Lupe server (Windows).
rem Python is looked for in this order:
rem   1. environment variable LUPE_PYTHON (full path to python.exe)
rem   2. first line of the file lupe-python.txt next to this script
rem   3. conda environment "gguf-lupe" (anaconda3 or miniconda3)
rem   4. py -3, then python
rem Further options are passed on, e.g.:  start-lupe-server.cmd --device cpu
setlocal
cd /d "%~dp0"
set "PY=%LUPE_PYTHON%"
set "PYA="
if not defined PY if exist "lupe-python.txt" set /p PY=<"lupe-python.txt"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\gguf-lupe\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\gguf-lupe\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\gguf-lupe\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\gguf-lupe\python.exe"
if not defined PY (
  where py >nul 2>nul && (set "PY=py" & set "PYA=-3")
)
if not defined PY set "PY=python"

"%PY%" %PYA% -c "import numpy, gguf" 2>nul
if not errorlevel 1 goto run
rem Messages in the Windows display language, German or English (the server does the same).
set "LDE="
for /f %%L in ('powershell -NoProfile -NonInteractive -Command "[cultureinfo]::CurrentUICulture.TwoLetterISOLanguageName" 2^>nul') do if /i "%%L"=="de" set "LDE=1"
echo.
if defined LDE goto missing_de
echo Python packages are missing. Install them once with:
echo    "%PY%" %PYA% -m pip install numpy gguf
echo Optional, for a GPU or a faster CPU: PyTorch, see https://pytorch.org
goto missing_end
:missing_de
echo Es fehlen Python-Pakete. Einmal installieren mit:
echo    "%PY%" %PYA% -m pip install numpy gguf
echo Optional fuer Grafikkarte oder schnellere CPU: PyTorch, siehe https://pytorch.org
:missing_end
echo.
pause
exit /b 1

:run
"%PY%" %PYA% lupe_server.py %*
pause
