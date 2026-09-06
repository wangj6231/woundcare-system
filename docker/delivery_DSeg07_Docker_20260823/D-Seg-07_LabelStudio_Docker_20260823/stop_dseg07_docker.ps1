$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
docker compose stop
Write-Host 'D-Seg-07 Label Studio stopped. Persistent data was retained.'
