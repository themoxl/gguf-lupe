@echo off
rem Start the GGUF-Lupe server (Windows).
rem Python is looked for in this order:
rem   1. environment variable LUPE_PYTHON (full path to python.exe)
rem   2. first line of the file lupe-python.txt next to this script
rem   3. the Python that comes with the portable package (python\python.exe next to this script)
rem   4. conda environment "gguf-lupe" (anaconda3 or miniconda3)
rem   5. py -3, then python
rem A path from 1. or 2. that no longer exists (e.g. a deleted conda environment) is skipped with a note.
rem If numpy or gguf are missing, the script offers to install them (pip); without internet: the portable package.
rem Further options are passed on, e.g.:  start-lupe-server.cmd --device cpu
setlocal
cd /d "%~dp0"
set "PY=%LUPE_PYTHON%"
set "PYA="
set "PYSRC=LUPE_PYTHON"
set "BADPY="
set "LDE="
set "LDONE="
set "TRIED="
set "PORTABLE=https://github.com/themoxl/gguf-lupe/releases"
if not defined PY if exist "lupe-python.txt" set "PYSRC=lupe-python.txt"
if not defined PY if exist "lupe-python.txt" set /p PY=<"lupe-python.txt"
if not defined PY goto search
if exist "%PY%" goto search
where "%PY%" >nul 2>nul && goto search
set "BADPY=%PY%"
set "PY="
:search
if not defined PY if exist "python\python.exe" set "PY=%CD%\python\python.exe"
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
if defined TRIED goto pipfail
call :lang
echo.
if defined LDE goto missing_de
echo Python packages are missing: numpy and gguf (about 30 MB).
choice /c YN /n /m "Install them now automatically? [Y/N] "
goto missing_ask
:missing_de
echo Es fehlen Python-Pakete: numpy und gguf (etwa 30 MB).
choice /c JN /n /m "Jetzt automatisch installieren? [J/N] "
:missing_ask
if errorlevel 2 goto manual
set "TRIED=1"
"%PY%" %PYA% -m pip install --disable-pip-version-check numpy gguf
goto check

:pipfail
echo.
if defined LDE goto pipfail_de
echo The installation did not work (e.g. no internet or a company proxy).
goto manual_en
:pipfail_de
echo Die Installation hat nicht geklappt (z.B. kein Internet oder Firmen-Proxy).
goto manual_de

:manual
if defined LDE goto manual_de
:manual_en
echo By hand:  "%PY%" %PYA% -m pip install numpy gguf
echo Or without any installation: the portable package (Python included) from
echo    %PORTABLE%
echo Optional, for a GPU or a faster CPU: PyTorch, see https://pytorch.org
goto fail
:manual_de
echo Von Hand:  "%PY%" %PYA% -m pip install numpy gguf
echo Oder ganz ohne Installation: das portable Paket (Python ist dabei) von
echo    %PORTABLE%
echo Optional fuer Grafikkarte oder schnellere CPU: PyTorch, siehe https://pytorch.org
goto fail

:nopython
call :lang
echo.
if defined LDE goto nopython_de
echo Python could not be started: "%PY%" %PYA%
echo Easiest: the portable package, Python included, nothing to install:
echo    %PORTABLE%
echo Or install Python 3.9 or newer, or put the full path to python.exe into lupe-python.txt.
goto fail
:nopython_de
echo Python laesst sich nicht starten: "%PY%" %PYA%
echo Am einfachsten: das portable Paket, Python ist dabei, nichts zu installieren:
echo    %PORTABLE%
echo Oder Python 3.9 oder neuer installieren, oder den vollen Pfad zu python.exe in lupe-python.txt eintragen.

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
