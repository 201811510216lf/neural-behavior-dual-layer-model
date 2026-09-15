$ErrorActionPreference = 'Stop'

$PythonExe = Join-Path $env:USERPROFILE '.conda\envs\pytorch-GPU\python.exe'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

& $PythonExe (Join-Path $Here 'run_experiments.py') --max-epochs 200 --min-epochs 60 --patience 40 --shuffle-repeats 20 --random-repeats-full 500 --random-repeats-arch 100 @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe (Join-Path $Here 'run_shape_matched_null.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe (Join-Path $Here 'build_figures.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe (Join-Path $Here 'run_paired_transition_analysis.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe (Join-Path $Here 'audit_legacy_result_consistency.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe (Join-Path $Here 'build_transition_figures.py')
exit $LASTEXITCODE
