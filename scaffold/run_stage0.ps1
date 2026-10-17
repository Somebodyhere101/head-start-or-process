# SCAFFOLD Stage 0 main run (PREREGISTRATION.md section 5 with DEVIATIONS D2-D4): adapter 1 full candidate set (done), adapters 2-6 adaptive light mode.
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\stage0\chain.log"; New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
foreach ($ad in @('ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924')) {
  if (Test-Path "results\stage0\$ad.json") { "[$(Get-Date -Format s)] skip $ad (done)" | Add-Content $log; continue }
  Cool; "[$(Get-Date -Format s)] start $ad (light)" | Add-Content $log
  & $py stage0.py --adapter $ad --n 32 --tag stage0 --light *> "results\stage0\$ad.stdout.txt"
  "[$(Get-Date -Format s)] end $ad exit $LASTEXITCODE" | Add-Content $log
}
"[$(Get-Date -Format s)] STAGE0 DONE" | Add-Content $log
