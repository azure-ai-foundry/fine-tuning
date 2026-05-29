# Quick status check for the 3 detached autopilot runs.
# Usage:  pwsh -File check_runs.ps1
Set-Location $PSScriptRoot
$pidList = Get-Content .\test_agent_pids.txt
foreach ($line in $pidList) {
  $parts = $line -split " ", 2
  $id = [int]$parts[0]
  $run = $parts[1]
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  $alive = if ($p) { "alive ($(Get-Date $p.StartTime -f HH:mm))" } else { "EXITED" }
  $logPath = ".\$run\autopilot.log"
  $errPath = ".\$run\autopilot.err"
  $logSize = if (Test-Path $logPath) { (Get-Item $logPath).Length } else { 0 }
  $errSize = if (Test-Path $errPath) { (Get-Item $errPath).Length } else { 0 }
  # Latest phase line
  $phase = "(no phase yet)"
  if ($logSize -gt 0) {
    $latest = Get-Content $logPath | Select-String -Pattern "^\s*PHASE\s+\d" | Select-Object -Last 1
    if ($latest) { $phase = $latest.Line.Trim() }
  }
  $finalLine = if ($logSize -gt 0) { (Get-Content $logPath -Tail 1) } else { "" }
  Write-Host ""
  Write-Host "==== $run ===="
  Write-Host "  pid=$id $alive  log=$logSize  err=$errSize"
  Write-Host "  $phase"
  Write-Host "  > $finalLine"
  if ($errSize -gt 0) {
    Write-Host "  -- LAST ERR --"
    Get-Content $errPath -Tail 3 | ForEach-Object { Write-Host "    $_" }
  }
}
