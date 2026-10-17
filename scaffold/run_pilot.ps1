# SCAFFOLD Stage P (PREREGISTRATION.md section 5): budget calibration, base and A_raw only, B seeds 1-3, both B tasks, 6 passes.
# Waits for the GPU to cool below 62 C before each cell (thermal policy); one job on the card at a time.
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\pilot\chain.log"; New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
foreach ($rule in @('bind', 'bext')) {
  foreach ($s in 1..3) {
    $cell = "${rule}_b${s}_ST2_20260919"
    if (Test-Path "results\pilot\$cell.json") { $j = Get-Content "results\pilot\$cell.json" | ConvertFrom-Json; if ($j.seconds) { "[$(Get-Date -Format s)] skip $cell (done)" | Add-Content $log; continue } }
    Cool; "[$(Get-Date -Format s)] start $cell" | Add-Content $log
    & $py h2run.py --rule $rule --bseed $s --arms base,A_raw --adapter ST2_20260919 --passes 6 --tag pilot *> "results\pilot\$cell.stdout.txt"
    "[$(Get-Date -Format s)] end $cell exit $LASTEXITCODE" | Add-Content $log
  }
}
"[$(Get-Date -Format s)] PILOT DONE" | Add-Content $log
