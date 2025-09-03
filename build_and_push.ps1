
$ErrorActionPreference = "Stop"

if (-not $env:REGISTRY) {
    $env:REGISTRY = "docker-registry.ebrains.eu/workbench"
    Write-Host "Using default registry: $env:REGISTRY" -ForegroundColor Yellow
}

try {
    Write-Host "Building API image..." -ForegroundColor Cyan
    docker build -t "$env:REGISTRY/webnutil-api" -f Dockerfile.api .
    if ($LASTEXITCODE -ne 0) { throw "API build failed" }
    Write-Host "✓ API build successful" -ForegroundColor Green

    Write-Host "Building Worker image..." -ForegroundColor Cyan
    docker build -t "$env:REGISTRY/webnutil-worker" -f Dockerfile.worker .
    if ($LASTEXITCODE -ne 0) { throw "Worker build failed" }
    Write-Host "✓ Worker build successful" -ForegroundColor Green

    Write-Host "Pushing images..." -ForegroundColor Cyan
    docker push "$env:REGISTRY/webnutil-api"
    if ($LASTEXITCODE -ne 0) { throw "API push failed" }
    
    docker push "$env:REGISTRY/webnutil-worker"
    if ($LASTEXITCODE -ne 0) { throw "Worker push failed" }

    Write-Host "✓ Build and push completed!" -ForegroundColor Green
}
catch {
    Write-Host "✗ $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}