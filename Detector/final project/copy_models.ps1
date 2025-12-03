# PowerShell helper to copy model files into final project models folder
# Run this from PowerShell on your machine.

$src = "C:\Users\Madhav\Work\Project\traffic-violation-detector\models"
$dst = "C:\Users\Madhav\Work\final project\models"

if (-Not (Test-Path $src)) {
    Write-Error "Source models folder not found: $src"
    exit 1
}

if (-Not (Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst -Force | Out-Null
}

$files = @('yolov8n.pt','license_plate_detector.pt','numberplt.pt','helmet.pt')

foreach ($f in $files) {
    $s = Join-Path $src $f
    $d = Join-Path $dst $f
    if (Test-Path $s) {
        Copy-Item -Path $s -Destination $d -Force
        Write-Host "Copied $f to final project models folder"
    } else {
        Write-Warning "$f not found in source ($s)."
    }
}

Write-Host "Done. Verify files in $dst"