# SCAFFOLD exploratory E1 (DEVIATIONS.md): learning-rate sensitivity benchmark, replicate floor, forecast on B seed 101.
Add-Type -Name Pwr -Namespace Scf -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000041") | Out-Null
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\explore_chain.log"
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
$budget = @{ 'bind' = @(1, 24); 'bext' = @(3, 72) }
foreach ($t in @('explore_lr', 'stage2_rep', 'forecast')) { New-Item -ItemType Directory -Force "results\$t" | Out-Null }
foreach ($bs in @(101, 102, 103)) { foreach ($rule in @('bind', 'bext')) {
  $p = $budget[$rule][0]; $e = $budget[$rule][1]; $cell = "${rule}_b${bs}_ST2_20260919"
  Cool; "[$(Get-Date -Format s)] start lr $cell" | Add-Content $log
  & $py h2run.py --rule $rule --bseed $bs --passes $p --eval_every $e --arms base_lr2,base_lr4 --adapter ST2_20260919 --tag explore_lr *> "results\explore_lr\$cell.stdout.txt"
  "[$(Get-Date -Format s)] end lr $cell exit $LASTEXITCODE" | Add-Content $log } }
foreach ($rule in @('bind', 'bext')) {
  $p = $budget[$rule][0]; $e = $budget[$rule][1]; $cell = "${rule}_b101_ST2_20260919"
  Cool; "[$(Get-Date -Format s)] start rep $cell" | Add-Content $log
  & $py h2run.py --rule $rule --bseed 101 --passes $p --eval_every $e --arms base --adapter ST2_20260919 --tag stage2_rep *> "results\stage2_rep\$cell.stdout.txt"
  "[$(Get-Date -Format s)] end rep $cell exit $LASTEXITCODE" | Add-Content $log }
$ads = @('ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924')
foreach ($k in @('shuf', 'single', 'perm')) { foreach ($s in @(20260919, 20260921, 20260922)) { $ads += "results\controls\${k}_$s.pt" } }
foreach ($rule in @('bind', 'bext')) { $first = $true; foreach ($ad in $ads) {
  $arms = if ($first) { 'base,A_matched,A_raw' } else { 'A_matched,A_raw' }; $first = $false; $name = [IO.Path]::GetFileNameWithoutExtension($ad)
  Cool; "[$(Get-Date -Format s)] start forecast $rule $name" | Add-Content $log
  & $py forecast.py --rule $rule --bseed 101 --arms $arms --adapter $ad --tag forecast *> "results\forecast\${rule}_$name.stdout.txt"
  "[$(Get-Date -Format s)] end forecast $rule $name exit $LASTEXITCODE" | Add-Content $log } }
"[$(Get-Date -Format s)] EXPLORE DONE" | Add-Content $log
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
