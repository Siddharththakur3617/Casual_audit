# Trains one DreamerV3 run and restarts it from checkpoint_last.pt whenever it crashes,
# until checkpoint_epoch0500.pt exists. Run from the project root, one window per run:
#   .\tools\train_supervised.ps1 -Name cc -Data datasets/cc_phys -Config configs/models/dreamer_cartpole.yaml -SaveDir runs/dreamer_cc_phys
#   .\tools\train_supervised.ps1 -Name ac -Data datasets/ac_phys -Config configs/models/dreamer_acrobot.yaml -SaveDir runs/dreamer_ac_phys
param([Parameter(Mandatory)][string]$Name, [Parameter(Mandatory)][string]$Data,
      [Parameter(Mandatory)][string]$Config, [Parameter(Mandatory)][string]$SaveDir,
      [int]$FinalEpoch = 500, [int]$MaxRestarts = 50)

$env:PYTHONUTF8 = '1'
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$final = Join-Path $SaveDir ("checkpoints\checkpoint_epoch{0:D4}.pt" -f $FinalEpoch)
$last = Join-Path $SaveDir 'checkpoints\checkpoint_last.pt'
$null = New-Item -ItemType Directory -Force logs

for ($try = 1; $try -le $MaxRestarts; $try++) {
    if (Test-Path $final) { Write-Host "[$Name] finished: $final"; break }
    $args = @('-u', '-m', 'models.dreamer.train', '--data', $Data, '--config', $Config,
              '--save-dir', $SaveDir, '--device', 'cuda')
    if (Test-Path $last) { $args += @('--resume', $last) }
    $log = "logs\dreamer_${Name}_phys_run$try"
    Write-Host ("[{0}] {1:HH:mm:ss} start #{2} {3}" -f $Name, (Get-Date), $try, $(if (Test-Path $last) { '(resume)' } else { '(fresh)' }))
    $p = Start-Process $py -ArgumentList $args -PassThru -NoNewWindow -Wait `
         -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err"
    Write-Host ("[{0}] {1:HH:mm:ss} exited with code {2}" -f $Name, (Get-Date), $p.ExitCode)
    Start-Sleep -Seconds 20   # let the driver settle after a crash
}
