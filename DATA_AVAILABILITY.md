# Data availability and public-release boundary

## Private data that are not included

This public repository intentionally excludes:

- raw hippocampal electrophysiology and spike-sorting exports;
- synchronized neural and video records;
- behavior-label files and manual labeling workbooks;
- 10 ms binned, aligned, or normalized neural tables;
- animal-level grouping spreadsheets and per-recording statistical tables;
- per-window predictions, fold membership manifests, model checkpoints, and intermediate arrays.

These exclusions are enforced both by whitelist-based repository assembly and by `.gitignore` patterns.

## What the private aligned tables look like

The model-facing data are recording-level tables sampled at 10 ms per row. A typical aligned table contains:

| Field | Meaning |
|---|---|
| `time_bin` | Monotonic 10 ms time-bin index |
| `recording` | Recording identifier |
| `N1` ... `N12` | Neural activity channels; recordings with fewer neurons are zero-padded for model input |
| `label` | Behavior code: 0 for invalid/other, 1–5 for the five target behaviors |

Behavior labels are:

| Raw label | Behavior |
|---:|---|
| 0 | Invalid or other behavior |
| 1 | Hole Exploration |
| 2 | Wrong Hole Exploration |
| 3 | Hesitating |
| 4 | Changing Direction |
| 5 | Walking |

The training pipeline removes label 0, converts labels 1–5 to classes 0–4, constructs 500-bin windows, and advances by 250 bins. It retains only windows whose label stays constant across the full 5 s interval.

## Public artifacts

Only aggregate metrics, fold-level performance summaries without animal or recording identifiers, explanatory documents, selected p-value figures, raster images, randomization-null summaries, and source code are public. The raster and group-difference figures are static derived outputs and cannot be used as drop-in replacements for the omitted numerical tables. The `results/window_mechanism/` bundle contains no checkpoints, aligned bins, labels, per-window predictions, or animal-level rows.
