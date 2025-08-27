
if (-not $env:REGISTRY) {
    $env:REGISTRY = "docker-registry.ebrains.eu/workbench"
    Write-Host "REGISTRY environment variable not set. Using default: $env:REGISTRY"
}
Write-Host "Pushing images to harbor"
docker push "$env:REGISTRY/webnutil-api"
docker push "$env:REGISTRY/webnutil-worker"
Write-Host "Push completed successfully."