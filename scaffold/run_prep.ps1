# SCAFFOLD prep chain: Stage 0 (adapters 2-6, light) -> control adapters (shuf / single / perm on ST2 seeds) -> smoke test of the fade and PC-H2 arms.
$py = "C:\Users\caabm\Documents\Codex\2026-09-06\co\work\qwen-cuda-env\Scripts\python.exe"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "results\prep_chain.log"
function Cool { while ([int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -ge 62) { Start-Sleep -Seconds 15 } }
"[$(Get-Date -Format s)] prep start" | Add-Content $log
& powershell -NoProfile -ExecutionPolicy Bypass -File run_stage0.ps1
"[$(Get-Date -Format s)] stage0 chain finished" | Add-Content $log
New-Item -ItemType Directory -Force results\controls | Out-Null
foreach ($kind in @('perm', 'shuf', 'single')) {
  foreach ($s in @(20260919, 20260921, 20260922)) {
    if (Test-Path "results\controls\${kind}_$s.json") { continue }
    Cool; "[$(Get-Date -Format s)] start control $kind $s" | Add-Content $log
    & $py train_controls.py --kind $kind --seed $s *> "results\controls\${kind}_$s.stdout.txt"
    "[$(Get-Date -Format s)] end control $kind $s exit $LASTEXITCODE" | Add-Content $log
  }
}
Cool; "[$(Get-Date -Format s)] start smoke2" | Add-Content $log
& $py h2run.py --rule bind --bseed 9 --n_train 16 --n_eval 16 --passes 3 --eval_every 8 --arms A_matched_fade,PC_H2 --tag smoke2 *> "results\smoke2.stdout.txt"
"[$(Get-Date -Format s)] end smoke2 exit $LASTEXITCODE" | Add-Content $log
"[$(Get-Date -Format s)] PREP DONE" | Add-Content $log
