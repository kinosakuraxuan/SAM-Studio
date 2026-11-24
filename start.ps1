param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.runtime/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw '项目环境不存在，请先按 README 安装依赖。'
}
& $python -X utf8 -m uvicorn sam_app.server:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
