$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}

Start-Process -FilePath $python -ArgumentList '-c','from waitress import serve; import backend_app; serve(backend_app.app, host="0.0.0.0", port=5000)' -WindowStyle Hidden
Write-Host 'Stocksense backend started in the background.'
