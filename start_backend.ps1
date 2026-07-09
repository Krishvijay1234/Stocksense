$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) {
    $python = 'python'
}

Start-Process -FilePath $python -ArgumentList @('-u', 'backend_app.py') -WorkingDirectory $projectRoot -WindowStyle Hidden
Write-Host 'Backend launch command sent.'
