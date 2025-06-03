
if (-not $env:REGISTRY) {
    $env:REGISTRY = "docker-registry.ebrains.eu/workbench"
    Write-Host "REGISTRY environment variable not set. Using default: $env:REGISTRY"
}

Write-Host "Building API image for WebNutil"
docker build -t "$env:REGISTRY/webnutil-api" -f ./dockerfile_api .

Write-Host "Building WebNutil Worker image"
docker build -t "$env:REGISTRY/webnutil-worker" -f ./worker/Dockerfile .

Write-Host "Pushing images to harbor"
docker push "$env:REGISTRY/webnutil-api"
docker push "$env:REGISTRY/webnutil-worker"

Write-Push "Build and push completed successfully."