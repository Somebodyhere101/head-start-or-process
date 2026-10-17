# SCAFFOLD Stage 3 chain (PREREGISTRATION_STAGE3.md): pilot (budgets) -> pilot_select -> 0.6B controls -> 0.6B light Stage 0 -> 0.6B grid ->
# 1.7B B_para grid -> analyze3. Holds the system awake (SetThreadExecutionState); each job starts below 62 C.
Add-Type -Name Pwr -Namespace Scf -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000041") | Out-Null
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\stage3_chain.log"
$M06 = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
function SetModel($m) { if ($m -eq 'q06') { $env:CED_MODEL_DIR = $M06 } else { Remove-Item Env:CED_MODEL_DIR -ErrorAction SilentlyContinue } }
function Run($m, $rule, $bs, $arms, $adapter, $tag, $passes, $ntrain, $every) {
  SetModel $m; New-Item -ItemType Directory -Force "results\$tag" | Out-Null
  $name = [IO.Path]::GetFileNameWithoutExtension($adapter); $cell = "${rule}_b${bs}_$name"
  if (Test-Path "results\$tag\$cell.json") { $j = Get-Content "results\$tag\$cell.json" | ConvertFrom-Json; $have = @($j.arms.PSObject.Properties.Name); if (-not ($arms.Split(',') | Where-Object { $have -notcontains $_ })) { return } }
  Cool; "[$(Get-Date -Format s)] start $m $tag $cell $arms" | Add-Content $log
  & $py h2run.py --rule $rule --bseed $bs --passes $passes --n_train $ntrain --eval_every $every --arms $arms --adapter $adapter --tag $tag *> "results\$tag\$cell.$($arms.Replace(',','_')).stdout.txt"
  "[$(Get-Date -Format s)] end $m $tag $cell exit $LASTEXITCODE" | Add-Content $log
}
"[$(Get-Date -Format s)] stage3 start" | Add-Content $log
# 1. pilot (base, A_raw; 6 passes; eval every 96)
foreach ($pc in @(@('q06', 'bind', 'ST2s_20260919'), @('q06', 'bext', 'ST2s_20260919'), @('q06', 'bpara', 'ST2s_20260919'), @('q17', 'bpara', 'ST2_20260919'))) {
  foreach ($bs in 1..3) { Run $pc[0] $pc[1] $bs 'base,A_raw' $pc[2] "stage3_pilot_$($pc[0])" 6 192 96 } }
& $py pilot_select.py *> "results\stage3_pilot_select.stdout.txt"
$B = Get-Content "results\stage3_budgets.json" | ConvertFrom-Json
"[$(Get-Date -Format s)] budgets $((Get-Content results\stage3_budgets.json) -join ' ')" | Add-Content $log
# 2. 0.6B controls
foreach ($kind in @('shuf', 'single')) { foreach ($s in @(20260919, 20260921, 20260922)) {
  if (Test-Path "results\controls\q06${kind}_$s.json") { continue }
  Remove-Item Env:CED_MODEL_DIR -ErrorAction SilentlyContinue; Cool; "[$(Get-Date -Format s)] start control q06 $kind $s" | Add-Content $log
  & $py train_controls.py --kind $kind --seed $s --model q06 *> "results\controls\q06${kind}_$s.stdout.txt"
  "[$(Get-Date -Format s)] end control q06 $kind $s exit $LASTEXITCODE" | Add-Content $log } }
# 3. 0.6B light Stage 0 (descriptive)
foreach ($ad in @('ST2s_20260919', 'ST2s_20260921', 'ST2s_20260922')) {
  if (Test-Path "results\stage0_q06\$ad.json") { continue }
  SetModel 'q06'; New-Item -ItemType Directory -Force "results\stage0_q06" | Out-Null; Cool; "[$(Get-Date -Format s)] start stage0 q06 $ad" | Add-Content $log
  & $py stage0.py --adapter $ad --n 32 --tag stage0_q06 --light *> "results\stage0_q06\$ad.stdout.txt"
  "[$(Get-Date -Format s)] end stage0 q06 $ad exit $LASTEXITCODE" | Add-Content $log }
# 4. grids
$plan = @(@('q06', @('bind', 'bpara', 'bext'), @('ST2s_20260919', 'ST2s_20260921', 'ST2s_20260922'), 'q06'), @('q17', @('bpara'), @('ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924'), ''))
foreach ($pl in $plan) {
  $m = $pl[0]; $tag = "stage3_$m"
  foreach ($bs in @(201, 202, 203)) { foreach ($rule in $pl[1]) {
    $b = $B."${m}_$rule"; $p = $b.passes; $n = $b.n_train; $e = $b.eval_every
    Run $m $rule $bs 'base,base_lr2' $pl[2][0] $tag $p $n $e
    foreach ($ad in $pl[2]) { Run $m $rule $bs 'A_raw,prior,A_matched,A_matched_fade' $ad $tag $p $n $e }
    foreach ($k in @('shuf', 'single')) { foreach ($s in @(20260919, 20260921, 20260922)) { Run $m $rule $bs 'A_raw,prior,A_matched' "results\controls\$($pl[3])${k}_$s.pt" $tag $p $n $e } }
  } }
}
Remove-Item Env:CED_MODEL_DIR -ErrorAction SilentlyContinue
& $py analyze3.py *> "results\stage3_analyze.stdout.txt"
"[$(Get-Date -Format s)] STAGE3 DONE" | Add-Content $log
[Scf.Pwr]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
