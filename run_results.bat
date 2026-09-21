@echo off
setlocal

rem Always run from the project root, regardless of the caller's current directory.
cd /d "%~dp0"
if errorlevel 1 (
    echo Failed to change to the project root.
    exit /b 1
)

set "OUTDIR=<dir>"
set "RUN500=%OUTDIR%\runs_500ep"
set "RUN5000=%OUTDIR%\runs_5000ep"

echo Running the 500-epoch sweep...
python run_thermo_metrics.py --epochs 500 --outdir "%RUN500%"
if errorlevel 1 (
    echo The 500-epoch sweep failed.
    exit /b 1
)

echo Running the 5000-epoch sweep...
python run_thermo_metrics.py --epochs 5000 --seeds 1 2 3 4 5 6 7 8 9 10 --outdir "%RUN5000%"
if errorlevel 1 (
    echo The 5000-epoch sweep failed.
    exit /b 1
)

echo Generating figures, tables, and summary...
python plot_results.py --dir500 "%RUN500%" --dir5000 "%RUN5000%" --outdir "%OUTDIR%"
if errorlevel 1 (
    echo Plot generation failed.
    exit /b 1
)

echo.
echo All results were generated successfully.
exit /b 0
