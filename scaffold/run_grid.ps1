# SCAFFOLD Stage 2 grid (PREREGISTRATION_STAGE2.md). Order: B seed -> rule -> adapter, so an interruption leaves complete comparison cells.
# Per (rule, B seed): base + PC_H2 (adapter-independent arms; run in the ST2_20260919 cell); per A adapter: A_raw, prior, A_matched, A_matched_fade;
# per control adapter: A_matched. Then the plain-vs-plain replicate floor (base again, B seed 101, both rules).
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\stage2\chain.log"; New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
$budget = @{ 'bind' = @(1, 24); 'bext' = @(3, 72) }      # passes, eval_every (8 evaluation points over the budget)
$adapters = @('ST2_20260919', 'ST2_20260921', 'ST2_20260922', 'ST2b_20260920', 'P1_20260923', 'P1_20260924')
$controls = @()
foreach ($k in @('shuf', 'single', 'perm')) { foreach ($s in @(20260919, 20260921, 20260922)) { $controls += "results\controls\${k}_$s.pt" } }
$pcf = (Get-Content "results\pc_calib\factor.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
$baseArms = if ($pcf -and $pcf -ne 'none') { 'base,PC_H2' } else { 'base' }; if (-not $pcf -or $pcf -eq 'none') { $pcf = '4' }
"[$(Get-Date -Format s)] grid start; PC factor file '$((Get-Content results\pc_calib\factor.txt -ErrorAction SilentlyContinue))'; base-cell arms $baseArms" | Add-Content $log
function Run($rule, $bs, $arms, $adapter, $tag) {
  $p = $budget[$rule][0]; $e = $budget[$rule][1]; $name = [IO.Path]::GetFileNameWithoutExtension($adapter); $cell = "${rule}_b${bs}_$name"
  if (Test-Path "results\$tag\$cell.json") { $j = Get-Content "results\$tag\$cell.json" | ConvertFrom-Json; $have = @($j.arms.PSObject.Properties.Name); $need = $arms.Split(','); if (-not ($need | Where-Object { $have -notcontains $_ })) { "[$(Get-Date -Format s)] skip $cell $arms" | Add-Content $log; return } }
  Cool; "[$(Get-Date -Format s)] start $cell $arms" | Add-Content $log
  & $py h2run.py --rule $rule --bseed $bs --passes $p --eval_every $e --arms $arms --adapter $adapter --pc_factor $pcf --tag $tag *> "results\$tag\$cell.stdout.txt"
  "[$(Get-Date -Format s)] end $cell exit $LASTEXITCODE" | Add-Content $log
}
foreach ($bs in @(101, 102, 103)) {
  foreach ($rule in @('bind', 'bext')) {
    Run $rule $bs $baseArms 'ST2_20260919' 'stage2'
    foreach ($ad in $adapters) { Run $rule $bs 'A_raw,prior,A_matched,A_matched_fade' $ad 'stage2' }
    foreach ($c in $controls) { Run $rule $bs 'A_matched' $c 'stage2' }
  }
}
foreach ($rule in @('bind', 'bext')) { Run $rule 101 'base' 'ST2_20260919' 'stage2_rep' }
"[$(Get-Date -Format s)] GRID DONE" | Add-Content $log
