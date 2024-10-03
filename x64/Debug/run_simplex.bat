@echo off

:: Check for at least one argument (dataset file)
if "%1"=="" (
    echo Usage: run_simplex.bat ^<dataset_file^> [SEQUENCE/OMP/CUDA/MPI/ALL] [isPrint Y/N]
    exit /b 1
)

set dataset=%1
set mode=%2
set isPrint=%3

:: If no mode is provided, default to ALL
if "%mode%"=="" (
    set mode=ALL
)

:: If isPrint is not provided, default to "N"
if "%isPrint%"=="" (
    set isPrint=N
)

:: Define the path to your executable (this assumes it is in the same folder as this script)
set executable_path=.\Assignment.exe

:: Check if the executable exists
if not exist "%executable_path%" (
    echo Executable not found! Ensure Assignment.exe is in the same directory as this script.
    pause
    exit /b 1
)

:: Run based on the mode
if /i "%mode%"=="SEQUENCE" (
    echo Running in SEQUENCE mode...
    "%executable_path%" "%dataset%" SEQUENCE "%isPrint%"
) else if /i "%mode%"=="OMP" (
    echo Running in OMP mode...
    "%executable_path%" "%dataset%" OMP "%isPrint%"
) else if /i "%mode%"=="CUDA" (
    echo Running in CUDA mode...
    "%executable_path%" "%dataset%" CUDA "%isPrint%"
) else if /i "%mode%"=="MPI" (
    echo Running in MPI mode with 4 processes...
    mpiexec -n 4 "%executable_path%" "%dataset%" MPI "%isPrint%"
) else if /i "%mode%"=="ALL" (
    echo Running in ALL mode...
    mpiexec -n 4 "%executable_path%" "%dataset%" ALL "%isPrint%"
) else (
    echo Invalid mode! Use SEQUENCE, OMP, CUDA, MPI, or ALL.
    exit /b 1
)

pause
