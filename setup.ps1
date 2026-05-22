#Requires -Version 5.1
<#
.SYNOPSIS
  Cài đặt môi trường Python cho Identity VM App.

.EXAMPLE
  .\setup.ps1
  .\setup.ps1 -RecreateVenv
#>
param(
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
    foreach ($cmd in @("python", "py")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            if ($cmd -eq "py") { return @("py", "-3") }
            return @("python")
        }
    }
    throw "Không tìm thấy Python. Cài Python 3.10+ và thêm vào PATH."
}

$py = Find-Python
$venv = Join-Path $Root ".venv"

if ($RecreateVenv -and (Test-Path $venv)) {
    Write-Host "Xóa virtualenv cũ: $venv"
    Remove-Item -LiteralPath $venv -Recurse -Force
}

if (-not (Test-Path $venv)) {
    Write-Host "Tạo virtualenv: $venv"
    & @py @("-m", "venv", $venv)
}

$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "Nâng cấp pip..."
& $python -m pip install --upgrade pip

Write-Host "Cài requirements.txt..."
& $pip install -r (Join-Path $Root "requirements.txt")

$cfg = Join-Path $Root "camera_config.json"
$example = Join-Path $Root "camera_config.example.json"
if (-not (Test-Path $cfg) -and (Test-Path $example)) {
    Copy-Item $example $cfg
    Write-Host "Đã tạo camera_config.json từ example — chỉnh RTSP/webcam trước khi chạy."
}

Write-Host ""
Write-Host "Hoàn tất. Kích hoạt venv:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Chạy app:"
Write-Host "  python main.py"
Write-Host "  streamlit run identity_vm_app/streamlit_test.py --server.port 8510"
