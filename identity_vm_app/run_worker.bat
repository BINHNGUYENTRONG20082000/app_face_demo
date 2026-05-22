@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0..
python "%~dp0worker\camera_worker.py" %*
