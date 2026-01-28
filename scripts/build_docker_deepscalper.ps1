param (
    [string]$Tag = "v10",
    [switch]$Push = $false
)

$ImageName = "bigcan/deepscalper"
$FullImageName = "$ImageName`:$Tag"

Write-Host "Building Docker Image: $FullImageName" -ForegroundColor Cyan
docker build -f Dockerfile.deepscalper -t $FullImageName .

if ($Push) {
    Write-Host "Pushing Docker Image: $FullImageName" -ForegroundColor Cyan
    docker push $FullImageName
} else {
    Write-Host "Skipping Push. Use -Push to push to registry." -ForegroundColor Yellow
}
