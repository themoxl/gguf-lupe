@echo off
rem Start the GGUF-Lupe server (Windows).
rem Python is looked for in this order:
rem   1. environment variable LUPE_PYTHON (full path to python.exe)
rem   2. first line of the file lupe-python.txt next to this script
rem   3. conda environment "gguf-lupe" (anaconda3 or miniconda3)
rem   4. py -3, then python
rem A path from 1. or 2. that no longer exists (e.g. a deleted conda environment) is skipped with a note.
rem Further options are passed on, e.g.:  start-lupe-server.cmd --device cpu
setlocal
cd /d "%~dp0"
set "PY=%LUPE_PYTHON%"
set "PYA="
set "PYSRC=LUPE_PYTHON"
set "BADPY="
set "LDE="
set "LDONE="
if not defined PY if exist "lupe-python.txt" set "PYSRC=lupe-python.txt"
if not defined PY if exist "lupe-python.txt" set /p PY=<"lupe-python.txt"
if not defined PY goto search
if exist "%PY%" goto search
where "%PY%" >nul 2>nul && goto search
set "BADPY=%PY%"
set "PY="
:search
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\gguf-lupe\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\gguf-lupe\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\gguf-lupe\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\gguf-lupe\python.exe"
if not defined PY (
  where py >nul 2>nul && (set "PY=py" & set "PYA=-3")
)
if not defined PY set "PY=python"

if not defined BADPY goto check
call :lang
echo.
if defined LDE goto badpy_de
echo Note: the Python from %PYSRC% no longer exists:
echo    "%BADPY%"
echo Using instead: "%PY%" %PYA%
echo Fix or delete %PYSRC%.
goto check
:badpy_de
echo Hinweis: Das Python aus %PYSRC% gibt es nicht mehr:
echo    "%BADPY%"
echo Stattdessen verwendet: "%PY%" %PYA%
echo %PYSRC% korrigieren oder loeschen.

:check
"%PY%" %PYA% -c "import sys" >nul 2>nul
if errorlevel 1 goto nopython
"%PY%" %PYA% -c "import numpy, gguf" 2>nul
if not errorlevel 1 goto run
call :lang
echo.
if defined LDE goto missing_de
echo Python packages are missing. Install them once with:
echo    "%PY%" %PYA% -m pip install numpy gguf
echo Optional, for a GPU or a faster CPU: PyTorch, see https://pytorch.org
goto fail
:missing_de
echo Es fehlen Python-Pakete. Einmal installieren mit:
echo    "%PY%" %PYA% -m pip install numpy gguf
echo Optional fuer Grafikkarte oder schnellere CPU: PyTorch, siehe https://pytorch.org
goto fail

:nopython
call :lang
echo.
if defined LDE goto nopython_de
echo Python could not be started: "%PY%" %PYA%
echo Install Python 3.9 or newer, e.g. a conda environment named gguf-lupe,
echo or put the full path to python.exe into lupe-python.txt.
goto fail
:nopython_de
echo Python laesst sich nicht starten: "%PY%" %PYA%
echo Python 3.9 oder neuer installieren, z.B. eine conda-Umgebung namens gguf-lupe,
echo oder den vollen Pfad zu python.exe in lupe-python.txt eintragen.

:fail
echo.
pause
exit /b 1

:run
"%PY%" %PYA% src\lupe_server.py %*
pause
exit /b

rem Messages in the Windows display language, German or English (the server does the same).
:lang
if defined LDONE exit /b
set "LDONE=1"
for /f %%L in ('powershell -NoProfile -NonInteractive -Command "[cultureinfo]::CurrentUICulture.TwoLetterISOLanguageName" 2^>nul') do if /i "%%L"=="de" set "LDE=1"
exit /b
