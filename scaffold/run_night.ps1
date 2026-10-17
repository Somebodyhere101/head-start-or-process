# SCAFFOLD night chain (PREREGISTRATION_STAGE2.md): control adapters (shuf, single) -> PC-H2 calibration on pilot seeds -> factor selection ->
# Stage-2 grid -> replicate floor. Holds the system awake for its own lifetime (SetThreadExecutionState; changes no settings). Thermal rule:
# each job starts only when the GPU is below 62 C.
Add-Type -Name Pwr -Namespace Scf -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000041") | Out-Null      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\night_chain.log"
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
"[$(Get-Date -Format s)] night start" | Add-Content $log
foreach ($kind in @('shuf', 'single')) {
  foreach ($s in @(20260919, 20260921, 20260922)) {
    if (Test-Path "results\controls\${kind}_$s.json") { continue }
    Cool; "[$(Get-Date -Format s)] start control $kind $s" | Add-Content $log
    & $py train_controls.py --kind $kind --seed $s *> "results\controls\${kind}_$s.stdout.txt"
    "[$(Get-Date -Format s)] end control $kind $s exit $LASTEXITCODE" | Add-Content $log
  }
}
$budget = @{ 'bind' = @(1, 24); 'bext' = @(3, 72) }
foreach ($rule in @('bind', 'bext')) {
  foreach ($bs in 1..3) {
    $p = $budget[$rule][0]; $e = $budget[$rule][1]
    foreach ($job in @(@('pc_calib_base', 'base', 4), @('pc_calib_f2', 'PC_H2', 2), @('pc_calib_f4', 'PC_H2', 4), @('pc_calib_f8', 'PC_H2', 8))) {
      $tag = $job[0]; $arm = $job[1]; $f = $job[2]; $cell = "${rule}_b${bs}_ST2_20260919"
      if (Test-Path "results\$tag\$cell.json") { $j = Get-Content "results\$tag\$cell.json" | ConvertFrom-Json; if ($j.arms.PSObject.Properties.Name -contains $arm) { continue } }
      New-Item -ItemType Directory -Force "results\$tag" | Out-Null; Cool; "[$(Get-Date -Format s)] start $tag $cell" | Add-Content $log
      & $py h2run.py --rule $rule --bseed $bs --passes $p --eval_every $e --arms $arm --adapter ST2_20260919 --pc_factor $f --tag $tag *> "results\$tag\$cell.stdout.txt"
      "[$(Get-Date -Format s)] end $tag $cell exit $LASTEXITCODE" | Add-Content $log
    }
  }
}
& $py pc_select.py *> "results\pc_select.stdout.txt"
"[$(Get-Date -Format s)] pc factor $(Get-Content results\pc_calib\factor.txt)" | Add-Content $log
& powershell -NoProfile -ExecutionPolicy Bypass -File run_grid.ps1
"[$(Get-Date -Format s)] NIGHT DONE" | Add-Content $log
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
