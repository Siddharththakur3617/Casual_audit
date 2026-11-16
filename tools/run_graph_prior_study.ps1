# Graph-prior study, end to end: every arm, every control, every evaluation. Run from the project root:
#   .\tools\run_graph_prior_study.ps1                          # all stages, in order
#   .\tools\run_graph_prior_study.ps1 -Stages sweep,main       # selected stages
#   .\tools\run_graph_prior_study.ps1 -Stages analysis         # only the evaluations (after training)
# Every training step is skipped when its final checkpoint exists and resumed from checkpoint_last.pt after
# a crash (the RTX 5090 drops off the driver now and then, see README "Setup"). Logs go to logs\graph_prior\.
#
# Stages (approximate RTX 5090 time):
#   sweep     10-epoch runs that pick lambda_ctx (per-output arm) and lambda_jac (arm J)          ~1.5 h
#   main      seed 42, 100 epochs: standard, masked (M), jacobian (J), per-output, dense control   ~9 h
#   seeds     seeds 43, 44 for standard, per-output, dense (-AllArmSeeds adds M and J)             ~10 h
#   controls  wrong-graph controls on the per-output arm: shuffled, 2 true edges deleted, 2 added  ~5.5 h
#   anchor    per-output arm with slot anchoring on all episodes and on 5% of episodes            ~3.5 h
#   lowdata   25% of the training episodes: standard, per-output, dense                           ~1.5 h
#   ood       held-out-physics evaluation sets for cart-pole and Acrobot (data only)              ~15 min
#   acrobot   Acrobot, 100 epochs: standard, per-output, dense (22 truly absent distractor edges)  ~6 h
#   analysis  graph-recovery curves, training comparisons, seeds, slot selectivity, OOD rollouts
param([string[]]$Stages = @('sweep', 'main', 'seeds', 'controls', 'anchor', 'lowdata', 'ood', 'acrobot', 'analysis'),
      [string]$Python = '.\.venv\Scripts\python.exe',
      [string]$Data = 'datasets/cc_phys', [string]$AcData = 'datasets/ac_phys',
      [int]$Epochs = 100, [int]$SweepEpochs = 10, [int[]]$ExtraSeeds = @(43, 44),
      [switch]$AllArmSeeds, [int]$MaxRestarts = 20)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$py = (Resolve-Path $Python).Path
$null = New-Item -ItemType Directory -Force logs\graph_prior, runs\_sweep, runs\_configs
$GP = 'configs/models/graph_priors'
$PO = 'models.jepa_peroutput.train'
$STD = 'models.jepa.train'

function Run-Py([string]$Log, [string[]]$PyArgs) {
    Write-Host ("{0:HH:mm:ss} python {1}" -f (Get-Date), ($PyArgs -join ' '))
    # Out-Host keeps the console output out of the function's return value (only the exit code is returned)
    & $py -u @PyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append | Out-Host
    return $LASTEXITCODE
}

function Invoke-Train([string]$Module, [string]$D, [string]$Config, [string]$SaveDir, [int]$E,
                      [int]$Seed = 42, [string[]]$Extra = @()) {
    $final = Join-Path $SaveDir ("checkpoints\checkpoint_epoch{0:D4}.pt" -f $E)
    $last = Join-Path $SaveDir 'checkpoints\checkpoint_last.pt'
    $name = Split-Path $SaveDir -Leaf
    for ($try = 1; $try -le $MaxRestarts; $try++) {
        if (Test-Path $final) { Write-Host "[$name] finished"; return }
        $a = @('-m', $Module, '--data', $D, '--config', $Config, '--save-dir', $SaveDir,
               '--epochs', "$E", '--seed', "$Seed", '--probe-every', '5')
        if (Test-Path $last) { $a += @('--resume', $last) }
        if ($Extra.Count -gt 0) { $a += @('--set') + $Extra }
        $code = Run-Py "logs\graph_prior\$name.log" $a
        if ($code -ne 0) { Write-Host "[$name] exited with code $code, retrying in 20 s"; Start-Sleep -Seconds 20 }
    }
    Write-Host "[$name] gave up after $MaxRestarts attempts"
}

function New-Config([string]$Src, [string]$Dst, [hashtable]$Training) {
    # copy of a config with training keys replaced (keeps the committed configs untouched)
    $pairs = ($Training.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ';'
    $null = New-Item -ItemType Directory -Force (Split-Path $Dst)
    & $py -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); [c['training'].__setitem__(k, yaml.safe_load(v)) for k, v in (p.split('=', 1) for p in sys.argv[3].split(';'))]; yaml.safe_dump(c, open(sys.argv[2], 'w'), sort_keys=False)" $Src $Dst $pairs
}

function Get-Pick([string]$File, [double]$Default) {
    # returned as an invariant-culture string ("0.1", never "0,1") because it goes into --set and YAML
    $v = $Default
    if (Test-Path $File) { $v = [double]((Get-Content $File -Raw | ConvertFrom-Json).chosen) }
    else { Write-Host "[pick] $File missing: using $Default" }
    return $v.ToString([cultureinfo]::InvariantCulture)
}

function Has([string]$s) { return $Stages -contains $s }

# ---------------------------------------------------------------- sweep: pick the loss weights by constraint
if (Has 'sweep') {
    Invoke-Train $STD $Data 'configs/models/jepa_cartpole.yaml' 'runs/_sweep/standard_e10' $SweepEpochs
    foreach ($l in @('0.1', '1.0')) {
        Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput.yaml" "runs/_sweep/peroutput_ctx$l" $SweepEpochs 42 @("training.lambda_ctx=$l")
        $cfg = "runs/_configs/graphjac_jac$l.yaml"
        New-Config 'configs/models/jepa_cartpole_graphjac.yaml' $cfg @{ lambda_jac = $l; epochs = $SweepEpochs }
        Invoke-Train $STD $Data $cfg "runs/_sweep/graphjac_jac$l" $SweepEpochs
    }
    Run-Py 'logs\graph_prior\pick.log' @('-m', 'evaluate.pick_lambda', '--reference', 'runs/_sweep/standard_e10',
        '--candidate', '0.1=runs/_sweep/peroutput_ctx0.1', '--candidate', '1.0=runs/_sweep/peroutput_ctx1.0',
        '--epoch', "$SweepEpochs", '--out', 'runs/_sweep/pick_lambda_ctx.json') | Out-Null
    Run-Py 'logs\graph_prior\pick.log' @('-m', 'evaluate.pick_lambda', '--reference', 'runs/_sweep/standard_e10',
        '--candidate', '0.1=runs/_sweep/graphjac_jac0.1', '--candidate', '1.0=runs/_sweep/graphjac_jac1.0',
        '--epoch', "$SweepEpochs", '--out', 'runs/_sweep/pick_lambda_jac.json') | Out-Null
}
$LCtx = Get-Pick 'runs/_sweep/pick_lambda_ctx.json' 0.1
$LJac = Get-Pick 'runs/_sweep/pick_lambda_jac.json' 1.0
Write-Host "[weights] lambda_ctx = $LCtx, lambda_jac = $LJac"
$SetCtx = @("training.lambda_ctx=$LCtx")
New-Config 'configs/models/jepa_cartpole_graphmask.yaml' 'runs/_configs/graphmask.yaml' @{ lambda_ctx = $LCtx; epochs = $Epochs }
New-Config 'configs/models/jepa_cartpole_graphjac.yaml' 'runs/_configs/graphjac.yaml' @{ lambda_ctx = $LCtx; lambda_jac = $LJac; epochs = $Epochs }

function Train-Arm([string]$Arm, [string]$Suffix, [int]$Seed) {
    switch ($Arm) {
        'standard'   { Invoke-Train $STD $Data 'configs/models/jepa_cartpole.yaml' "runs/jepa_cc_phys_e$Epochs$Suffix" $Epochs $Seed }
        'masked'     { Invoke-Train $STD $Data 'runs/_configs/graphmask.yaml' "runs/jepa_cc_phys_gmask_e$Epochs$Suffix" $Epochs $Seed }
        'jacobian'   { Invoke-Train $STD $Data 'runs/_configs/graphjac.yaml' "runs/jepa_cc_phys_gjac_e$Epochs$Suffix" $Epochs $Seed }
        'per-output' { Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput.yaml" "runs/jepa_cc_phys_peroutput_e$Epochs$Suffix" $Epochs $Seed $SetCtx }
        'dense'      { Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput_dense.yaml" "runs/jepa_cc_phys_peroutput_dense_e$Epochs$Suffix" $Epochs $Seed $SetCtx }
    }
}

# ---------------------------------------------------------------- main comparison (seed 42)
if (Has 'main') { foreach ($arm in @('standard', 'masked', 'jacobian', 'per-output', 'dense')) { Train-Arm $arm '' 42 } }

# ---------------------------------------------------------------- extra seeds
$SeedArms = @('standard', 'per-output', 'dense')
if ($AllArmSeeds) { $SeedArms += @('masked', 'jacobian') }
if (Has 'seeds') { foreach ($s in $ExtraSeeds) { foreach ($arm in $SeedArms) { Train-Arm $arm "_s$s" $s } } }

# ---------------------------------------------------------------- wrong-graph controls (per-output arm)
if (Has 'controls') {
    Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput_shuffled.yaml" "runs/jepa_cc_phys_peroutput_shuffled_e$Epochs" $Epochs 42 $SetCtx
    Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput.yaml" "runs/jepa_cc_phys_peroutput_drop2_e$Epochs" $Epochs 42 ($SetCtx + @('model.graph_corrupt={drop: 2, seed: 0}'))
    Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput.yaml" "runs/jepa_cc_phys_peroutput_add2_e$Epochs" $Epochs 42 ($SetCtx + @('model.graph_corrupt={add: 2, seed: 0}'))
}

# ---------------------------------------------------------------- slot anchoring (uses state labels)
if (Has 'anchor') {
    Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput_anchor.yaml" "runs/jepa_cc_phys_peroutput_anchor_e$Epochs" $Epochs 42 $SetCtx
    Invoke-Train $PO $Data "$GP/jepa_cartpole_peroutput_anchor05.yaml" "runs/jepa_cc_phys_peroutput_anchor05_e$Epochs" $Epochs 42 $SetCtx
}

# ---------------------------------------------------------------- low data (25% of training episodes)
$Low = "${Data}_frac25"
if (Has 'lowdata') {
    if (-not (Test-Path "$Low/train.hdf5")) { Run-Py 'logs\graph_prior\data.log' @('-m', 'datagen.subset', '--data', $Data, '--fraction', '0.25') | Out-Null }
    Invoke-Train $STD $Low 'configs/models/jepa_cartpole.yaml' "runs/jepa_cc_phys_frac25_e$Epochs" $Epochs
    Invoke-Train $PO $Low "$GP/jepa_cartpole_peroutput.yaml" "runs/jepa_cc_phys_frac25_peroutput_e$Epochs" $Epochs 42 $SetCtx
    Invoke-Train $PO $Low "$GP/jepa_cartpole_peroutput_dense.yaml" "runs/jepa_cc_phys_frac25_peroutput_dense_e$Epochs" $Epochs 42 $SetCtx
}

# ---------------------------------------------------------------- held-out physics (evaluation data only)
if (Has 'ood') {
    foreach ($d in @($Data, $AcData)) {
        if (-not (Test-Path "${d}_ood/val.hdf5")) { Run-Py 'logs\graph_prior\data.log' @('-m', 'datagen.make_ood', '--data', $d) | Out-Null }
    }
}

# ---------------------------------------------------------------- Acrobot
if (Has 'acrobot') {
    Invoke-Train $STD $AcData 'configs/models/jepa_acrobot.yaml' "runs/jepa_ac_phys_e$Epochs" $Epochs
    Invoke-Train $PO $AcData "$GP/jepa_acrobot_peroutput.yaml" "runs/jepa_ac_phys_peroutput_e$Epochs" $Epochs 42 $SetCtx
    Invoke-Train $PO $AcData "$GP/jepa_acrobot_peroutput_dense.yaml" "runs/jepa_ac_phys_peroutput_dense_e$Epochs" $Epochs 42 $SetCtx
}

# ---------------------------------------------------------------- analysis
if (Has 'analysis') {
    $L = 'logs\graph_prior\analysis.log'
    $S = 'results/studies/graph_prior_training'
    $main = @("standard=runs/jepa_cc_phys_e$Epochs", "masked=runs/jepa_cc_phys_gmask_e$Epochs", "jacobian=runs/jepa_cc_phys_gjac_e$Epochs",
              "per-output=runs/jepa_cc_phys_peroutput_e$Epochs", "dense=runs/jepa_cc_phys_peroutput_dense_e$Epochs")
    $ctrl = @("per-output=runs/jepa_cc_phys_peroutput_e$Epochs", "shuffled=runs/jepa_cc_phys_peroutput_shuffled_e$Epochs",
              "drop2=runs/jepa_cc_phys_peroutput_drop2_e$Epochs", "add2=runs/jepa_cc_phys_peroutput_add2_e$Epochs",
              "dense=runs/jepa_cc_phys_peroutput_dense_e$Epochs")
    $anch = @("standard=runs/jepa_cc_phys_e$Epochs", "per-output=runs/jepa_cc_phys_peroutput_e$Epochs",
              "anchor05=runs/jepa_cc_phys_peroutput_anchor05_e$Epochs", "anchor=runs/jepa_cc_phys_peroutput_anchor_e$Epochs")
    $low = @("standard=runs/jepa_cc_phys_frac25_e$Epochs", "per-output=runs/jepa_cc_phys_frac25_peroutput_e$Epochs",
             "dense=runs/jepa_cc_phys_frac25_peroutput_dense_e$Epochs")
    $acro = @("standard=runs/jepa_ac_phys_e$Epochs", "per-output=runs/jepa_ac_phys_peroutput_e$Epochs",
              "dense=runs/jepa_ac_phys_peroutput_dense_e$Epochs")

    function Existing([string[]]$Pairs) { return @($Pairs | Where-Object { Test-Path (Join-Path ($_ -split '=', 2)[1] 'history.json') }) }
    function Compare-Runs([string[]]$Pairs, [string]$D, [string]$Out) {
        $p = Existing $Pairs
        if ($p.Count -lt 2) { Write-Host "[skip] ${Out}: fewer than 2 finished runs"; return }
        $a = @('-m', 'evaluate.compare_training', '--data', $D, '--out', $Out, '--runs') + @($p | ForEach-Object { ($_ -split '=', 2)[1] }) +
             @('--labels') + @($p | ForEach-Object { ($_ -split '=', 2)[0] })
        Run-Py $L $a | Out-Null
    }

    # graph-recovery curves (low-data runs are scored on the full dataset, which has the expert graph and null)
    Run-Py $L @('-m', 'discovery.null_calibration', '--data', $AcData, '--reps', '20', '--splits', '20') | Out-Null
    foreach ($pair in (Existing ($main + $ctrl + $anch + $low))) {
        Run-Py $L @('run_all.py', '--data', $Data, '--run-dir', ($pair -split '=', 2)[1], '--curve', '--splits', '20', '--prune', 'null_cal_fwer') | Out-Null
    }
    foreach ($pair in (Existing $acro)) {
        Run-Py $L @('run_all.py', '--data', $AcData, '--run-dir', ($pair -split '=', 2)[1], '--curve', '--splits', '20', '--prune', 'null_cal_fwer') | Out-Null
    }

    # training comparisons (in distribution, held-out physics, low data, Acrobot)
    Compare-Runs $main $Data "$S/main"
    Compare-Runs $main "${Data}_ood" "$S/main_ood"
    Compare-Runs $ctrl $Data "$S/controls"
    Compare-Runs $anch $Data "$S/anchor"
    Compare-Runs $low $Data "$S/lowdata"
    Compare-Runs $low "${Data}_ood" "$S/lowdata_ood"
    # Acrobot: compare_training assumes cart-pole's 4 state columns, so use compare_seeds (one run per group)
    $acroOk = Existing $acro
    if ($acroOk.Count -ge 2) {
        $g = @(); foreach ($pair in $acroOk) { $g += @('--group', $pair) }
        Run-Py $L (@('-m', 'evaluate.compare_seeds', '--data', $AcData, '--out', "$S/acrobot") + $g) | Out-Null
        if (Test-Path "${AcData}_ood/val.hdf5") {
            Run-Py $L (@('-m', 'evaluate.compare_seeds', '--data', "${AcData}_ood", '--out', "$S/acrobot_ood") + $g) | Out-Null
        }
    }

    # seeds (standard, per-output, dense; plus M and J with -AllArmSeeds)
    $groups = @()
    foreach ($pair in $main) {
        $lab, $run = $pair -split '=', 2
        $runs = @($run) + @($ExtraSeeds | ForEach-Object { "${run}_s$_" })
        $runs = @($runs | Where-Object { Test-Path (Join-Path $_ 'history.json') })
        if ($runs.Count -ge 2) { $groups += @('--group', "$lab=" + ($runs -join ',')) }
    }
    if ($groups.Count -ge 4) { Run-Py $L (@('-m', 'evaluate.compare_seeds', '--data', $Data, '--out', "$S/seeds") + $groups) | Out-Null }

    # do the slots hold their variables, and does ctx hold physics? (unseen episodes)
    $sel = Existing ($main + $ctrl[1..3] + $anch[2..3])
    if ($sel.Count -gt 0) {
        Run-Py $L (@('-m', 'evaluate.slot_selectivity', '--data', $Data, '--split', 'val', '--out', "$S/selectivity", '--runs') +
                   @($sel | ForEach-Object { ($_ -split '=', 2)[1] }) + @('--labels') + @($sel | ForEach-Object { ($_ -split '=', 2)[0] })) | Out-Null
    }
    $selA = Existing $acro
    if ($selA.Count -gt 0) {
        Run-Py $L (@('-m', 'evaluate.slot_selectivity', '--data', $AcData, '--split', 'val', '--out', "$S/selectivity_acrobot", '--runs') +
                   @($selA | ForEach-Object { ($_ -split '=', 2)[1] }) + @('--labels') + @($selA | ForEach-Object { ($_ -split '=', 2)[0] })) | Out-Null
    }
    Write-Host "[analysis] outputs in $S"
}
