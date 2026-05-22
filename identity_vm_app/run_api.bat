@echo off
REM Chạy từ đúng thư mục identity_vm_app
cd /d "%~dp0"
set PYTHONPATH=%~dp0..
REM Tối đa 4 video phân tích đồng thời (đổi số hoặc xóa dòng này nếu cần)
if not defined IVM_VIDEO_ANALYZE_MAX_CONCURRENT set IVM_VIDEO_ANALYZE_MAX_CONCURRENT=2
if not defined IVM_VIDEO_ANALYZE_SPLIT_PARTS set IVM_VIDEO_ANALYZE_SPLIT_PARTS=4
if not defined IVM_VIDEO_ANALYZE_MAX_MB set IVM_VIDEO_ANALYZE_MAX_MB=0
if not defined IVM_VIDEO_ANALYZE_MAX_DURATION_S set IVM_VIDEO_ANALYZE_MAX_DURATION_S=0
python "%~dp0main.py" %*
