$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$images = Get-ChildItem -LiteralPath './annotation_package/images' -Recurse -File |
    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|webp)$' }
if ($images.Count -ne 383) { throw "Expected 383 development images; found $($images.Count)" }
if ($images | Where-Object { $_.FullName -match '(?i)[\\/]test[\\/]' }) {
    throw 'Protected test image found; Docker start blocked.'
}
docker compose config --quiet
docker compose pull
docker compose up -d
$deadline = (Get-Date).AddMinutes(5)
do {
    $health = docker inspect --format '{{.State.Health.Status}}' dseg07-label-studio 2>$null
    if ($health -eq 'healthy') { break }
    Start-Sleep -Seconds 5
} while ((Get-Date) -lt $deadline)
if ($health -ne 'healthy') {
    docker compose logs --tail 200
    throw "Label Studio health check failed: $health"
}
Write-Host 'D-Seg-07 Label Studio is ready: http://127.0.0.1:8083'
Write-Host 'test_images_used=0; blind_test_used=false'
