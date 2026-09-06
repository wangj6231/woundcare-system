$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
docker compose ps
$images = Get-ChildItem -LiteralPath './annotation_package/images' -Recurse -File |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|webp)$' }
$tasks = Get-Content -LiteralPath './annotation_package/label_studio_tasks.json' -Raw | ConvertFrom-Json
[pscustomobject]@{
    images = $images.Count
    tasks = $tasks.Count
    test_images_used = 0
    blind_test_used = $false
    url = 'http://127.0.0.1:8083'
} | ConvertTo-Json
