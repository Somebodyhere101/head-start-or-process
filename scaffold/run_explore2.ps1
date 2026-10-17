# SCAFFOLD exploratory E2 (DEVIATIONS.md): control adapters in the raw and prior frames. Starts after run_explore.ps1 has finished.
Add-Type -Name Pwr -Namespace Scf -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000041") | Out-Null
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\explore_chain.log"
while (-not (Select-String -Path $log -Pattern 'EXPLORE DONE' -Quiet)) { Start-Sleep -Seconds 30 }
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
$budget = @{ 'bind' = @(1, 24); 'bext' = @(3, 72) }
New-Item -ItemType Directory -Force "results\stage2" | Out-Null
foreach ($bs in @(101, 102, 103)) { foreach ($rule in @('bind', 'bext')) { foreach ($k in @('shuf', 'single')) { foreach ($s in @(20260919, 20260921, 20260922)) {
  $p = $budget[$rule][0]; $e = $budget[$rule][1]; $cell = "${rule}_b${bs}_${k}_$s"
  Cool; "[$(Get-Date -Format s)] start E2 $cell" | Add-Content $log
  & $py h2run.py --rule $rule --bseed $bs --passes $p --eval_every $e --arms A_raw,prior --adapter "results\controls\${k}_$s.pt" --tag stage2 *> "results\stage2\$cell.E2.stdout.txt"
  "[$(Get-Date -Format s)] end E2 $cell exit $LASTEXITCODE" | Add-Content $log } } } }
"[$(Get-Date -Format s)] E2 DONE" | Add-Content $log
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
