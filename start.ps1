# ModelForge launcher
# Usage: powershell -ExecutionPolicy Bypass -File start.ps1
# Backend API : http://127.0.0.1:8000
# Frontend UI : http://127.0.0.1:8096/index.html
# Login       : admin / Admin@123
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI,
# so non-ASCII characters break the parser.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = "C:\Users\<you>\AppData\Local\Python\pythoncore-3.14-64\python.exe"

foreach ($p in 8000, 8096) {
  $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop } catch {}
  }
}
Start-Sleep -Seconds 1

if (-not (Test-Path $Py)) {
  Write-Host "Python interpreter not found: $Py" -ForegroundColor Red
  exit 1
}

New-Item -ItemType Directory -Force -Path "$Root\logs" | Out-Null

# NOTE: use cmd /c start instead of Start-Process.
# Start-Process inherits the environment and dies when both HTTPS_PROXY and
# https_proxy exist (duplicate key) -- that happens on many Chinese machines.
Start-Process -FilePath "cmd.exe" -ArgumentList "/c","start","/b","$Py","-m","uvicorn","main:app","--host","127.0.0.1","--port","8000","--log-level","warning","^>","..\logs\api.out","2^>^&1" `
  -WorkingDirectory "$Root\backend" -WindowStyle Hidden
Write-Host "backend  -> http://127.0.0.1:8000" -ForegroundColor Cyan

Start-Process -FilePath "cmd.exe" -ArgumentList "/c","start","/b","$Py","-m","http.server","8096","--bind","127.0.0.1","^>","..\logs\web.out","2^>^&1" `
  -WorkingDirectory "$Root\frontend" -WindowStyle Hidden
Write-Host "frontend -> http://127.0.0.1:8096/index.html" -ForegroundColor Cyan

$ok = $false
for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 1
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 3 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch {}
}
if ($ok) {
  Write-Host ""
  Write-Host "READY" -ForegroundColor Green
  Write-Host "  UI   : http://127.0.0.1:8096/index.html" -ForegroundColor Green
  Write-Host "  User : admin / Admin@123" -ForegroundColor Green
} else {
  Write-Host ""
  Write-Host "backend not ready in 40s - check logs\api.err" -ForegroundColor Yellow
}
