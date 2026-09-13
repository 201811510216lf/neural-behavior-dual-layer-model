# Temporal-window mechanism experiment bundle

This folder contains the new experiment code, fold-level source data, statistical tables, and publication exports requested from `实验.md`.

## Reproduce

Use the specified environment:

```powershell
$PythonExe = Join-Path $env:USERPROFILE '.conda\envs\pytorch-GPU\python.exe'
& $PythonExe run_experiments.py --max-epochs 200 --min-epochs 60 --patience 40 --shuffle-repeats 20 --random-repeats-full 500 --random-repeats-arch 100
& $PythonExe run_shape_matched_null.py
& $PythonExe build_figures.py
```

The experiment runner reuses the frozen Full checkpoints and saved three-fold splits from the legacy 5 s / 50% overlap protocol. It retrains NoConv, NoLSTM, and NoAttention under those same splits, then runs the target-specific perturbations. Existing maximum, attention, and local-LSTM perturbation outputs are copied into normalized source-data tables rather than manually transcribed.

## Output layout

- `checkpoints/` — newly trained structural-ablation weights and training histories in the local run only; Git excludes this directory from the public release.
- `source_data/` — one tidy CSV per figure/table.
- `tables/` — publication-facing CSV tables.
- `figures/` — SVG, PDF, PNG, TIFF, and grayscale previews.
- `qa/` — data profiles, layout audits, figure checks, and run metadata.

`run_all.ps1` runs the same three steps with the settings used for this release. Existing ablation checkpoints are reused unless `--force-training` is passed.

The plotting script uses the installed `scipilot-figure-skill` helper scripts for data profiling, export, and visual QA. It resolves them from the current user's `.codex/skills` directory and does not depend on a hard-coded account name.

## Interpretation guardrails

The outputs are mechanistic sensitivity analyses of fitted models. They do not by themselves establish biological causality. In particular, sparse L3 mask coverage and overlapping-window splits must remain visible in any manuscript claim.
