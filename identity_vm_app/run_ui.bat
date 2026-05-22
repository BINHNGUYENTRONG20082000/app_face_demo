@echo off
cd /d "%~dp0.."
set PYTHONPATH=%~dp0..
if not defined IVM_VIDEO_ANALYZE_MAX_MB set IVM_VIDEO_ANALYZE_MAX_MB=0
if not defined IVM_VIDEO_ANALYZE_MAX_DURATION_S set IVM_VIDEO_ANALYZE_MAX_DURATION_S=0
streamlit run identity_vm_app/streamlit_test.py --server.port 8510 --server.maxUploadSize 102400 --server.maxMessageSize 102400 %*
