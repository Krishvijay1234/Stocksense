$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$frontendDir = Join-Path $projectRoot 'frontend'
if (-not (Test-Path $frontendDir)) {
    Write-Error 'Frontend folder not found.'
    exit 1
}

Set-Location $frontendDir

if (Get-Command npm -ErrorAction SilentlyContinue) {
    Start-Process -FilePath 'npm' -ArgumentList @('run','dev','--','--host','0.0.0.0') -WorkingDirectory $frontendDir -WindowStyle Hidden
    Write-Host 'Frontend launch command sent.'
} else {
    Write-Error 'npm not found in PATH.'
    exit 1
}
