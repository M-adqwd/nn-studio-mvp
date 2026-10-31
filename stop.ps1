# ModelForge 停止服务
# 用法：powershell -ExecutionPolicy Bypass -File stop.ps1
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($p in 8000, 8096) {
  $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    try {
      Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop
      Write-Host "已停止端口 $p (PID $($c.OwningProcess))" -ForegroundColor Green
    } catch { Write-Host "端口 $p 停止失败: $($_.Exception.Message)" -ForegroundColor Yellow }
  }
}

# 清理上传/解析工作目录（不动 _pending_*：会话可能仍在等待用户选择）
Get-ChildItem "$Root\backend\storage\_upload_*","$Root\backend\storage\_aiparse_*" `
  -ErrorAction SilentlyContinue | ForEach-Object {
    try { Remove-Item $_.FullName -Force -Recurse -ErrorAction Stop } catch {}
  }
Write-Host "服务已停止" -ForegroundColor Cyan