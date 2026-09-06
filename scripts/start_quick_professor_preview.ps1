param(
    [string]$EnvironmentFile = ".env.quick-preview"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
    throw "找不到 $EnvironmentFile。請先由 .env.quick-preview.example 建立它，並填入新的密碼與 Fernet key。"
}

if (-not (Test-Path -LiteralPath "models")) {
    throw "找不到 models 資料夾。請先放入封版模型權重；不可放入資料集或 Blind Test 影像。"
}

docker compose -f docker-compose.quick-preview.yml up --detach --build

Write-Host "教授預覽服務已只綁定本機 127.0.0.1:8000。"
Write-Host "接下來會建立短期 HTTPS 網址；關閉此視窗即停止外部連線。"

if (Get-Command cloudflared -ErrorAction SilentlyContinue) {
    cloudflared tunnel --url http://127.0.0.1:8000
} elseif (Get-Command npx -ErrorAction SilentlyContinue) {
    Write-Host "未偵測到 cloudflared，將以 npx 下載並啟動 Cloudflare Quick Tunnel。"
    npx wrangler tunnel quick-start http://127.0.0.1:8000
} else {
    throw "請安裝 cloudflared，或先安裝 Node.js（含 npx）後重新執行。"
}
