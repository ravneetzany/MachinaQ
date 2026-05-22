@echo off
set DGLBACKEND=pytorch
set KMP_DUPLICATE_LIB_OK=TRUE
set CONDA_ENV=E:\AiTools\AIModel_StepAnalyze\conda_env
set PATH=%CONDA_ENV%;%CONDA_ENV%\Library\mingw-w64\bin;%CONDA_ENV%\Library\usr\bin;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%PATH%
cd /d E:\AiTools\AIModel_StepAnalyze
"%CONDA_ENV%\python.exe" run_train.py
