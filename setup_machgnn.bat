@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  MachinaQ GNN Setup
echo  Project: AIModel_StepAnalyze
echo ============================================================
echo.

REM ── 1. Clone AAGNet ──────────────────────────────────────────
if not exist "machgnn\" (
    echo [1/3] Setting up machgnn...
    REM machgnn already present locally
    if errorlevel 1 (
        echo [ERROR] git clone failed. Is git installed and on PATH?
        exit /b 1
    )
    echo [OK]  machgnn/ ready
) else (
    echo [OK]  machgnn/ already present -- skipping clone
)

REM ── 2. Create pyocc conda environment ────────────────────────
conda env list 2>nul | find "pyocc" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [2/3] Creating pyocc conda environment from machgnn/environment.yaml
    echo       This downloads ~3 GB of packages and takes 10-20 minutes...
    conda env create -f machgnn\environment.yaml -n pyocc
    if errorlevel 1 (
        echo [ERROR] conda env create failed.
        echo         Try: conda clean --all  then re-run this script.
        exit /b 1
    )
    echo [OK]  pyocc environment created
) else (
    echo [OK]  pyocc conda environment already exists -- skipping
)

REM ── 3. Ensure occwl is installed (sometimes missing from yaml) ─
echo.
echo [3/3] Verifying occwl in pyocc env...
conda run -n pyocc python -c "import occwl" 2>nul
if errorlevel 1 (
    echo       occwl not found -- installing...
    conda run -n pyocc pip install occwl
    if errorlevel 1 (
        echo [WARN] occwl pip install failed. You may need to build from source.
        echo        See: https://github.com/AutodeskAILab/occwl
    ) else (
        echo [OK]  occwl installed
    )
) else (
    echo [OK]  occwl already installed
)

REM ── 4. Create dataset directory ──────────────────────────────
if not exist "dataset\MFInstSeg\" (
    mkdir dataset\MFInstSeg
)

echo.
echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo.
echo  NEXT STEP: Download the MFInstSeg dataset (~8 GB)
echo.
echo  1. Open  machgnn\README.md  and find the Google Drive link
echo     for MFInstSeg  (60,000 STEP files + AAG graphs + labels)
echo.
echo  2. Download and extract into:
echo       dataset\MFInstSeg\
echo.
echo  Expected folder structure after extraction:
echo       dataset\MFInstSeg\
echo           aag\          <- DGL graph .bin files
echo           labels\       <- per-face label .json files
echo           steps\        <- original STEP files
echo           train_val_test_split.json
echo.
echo  3. Train:
echo       conda run -n pyocc python train_machgnn.py
echo.
echo  Optional flags:
echo       --epochs 100   (default: 100)
echo       --batch_size 32  (default: 32; increase if GPU VRAM > 16 GB)
echo       --nist_only    use only local NIST STEP files (tiny dataset, for testing)
echo.
endlocal
