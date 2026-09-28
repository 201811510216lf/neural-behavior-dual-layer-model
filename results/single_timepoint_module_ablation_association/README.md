# Single-timepoint neural-data ablation × module ablation

This directory contains the complete frozen-model analysis linking 24 behavior-aligned neural timepoint masks to NoConv, NoLSTM, and NoAttention structural ablations.

## Locked protocol

- No model was retrained and no data split was regenerated.
- Full, NoConv, NoLSTM, and NoAttention use coherent three-fold checkpoints and the same saved Run1/Run2 test indices.
- Legacy preprocessing remains 5 s windows, 10 ms/bin, 500 bins, 250-bin step, and stage-label-constant selection.
- Each point-wise intervention activates one behavior label and one time center only.
- The mask maps the behavior onset back to recording time and replaces `[center-1 s, center+1 s)`, clipped to the current 5 s window.
- Every neural channel is filled with its own maximum from the original, unmodified 5 s window.
- Every reported delta is `ablated - baseline`; signed and raw values are both retained.
- Sample-level outputs remain Run1/Run2 stage-wise and are not converted into artificial five-class rows.

## Input gate

The analysis reads `Summary_P_long` from the original t-test workbook and requires exactly 24 unique points after filtering `window_ms == 10`, `p_value < 0.05`, labels 1–5, and `time_s` within [-1, 1]. The run aborts if the exact expected list is not recovered.

## Main results

| Result | Value |
|---|---:|
| Strongest point-wise target-class ΔRecall | L5 +0.70 s → L5: -6.313 pp |
| Strongest structural target-class ΔRecall | NoConv → L4: -14.810 pp |
| Largest absolute five-behavior ΔRecall-vector Spearman | L1 -0.69 s × NoConv: 0.821 |
| Largest pooled CW damage-set Jaccard | L2 +0.06 s × NoLSTM: 0.065 |
| Largest absolute sample-level Spearman | L4 -0.56 s × NoConv, Run2, ΔPtrue: 0.154 |
| Strongest recall double-ablation interaction | L5 +0.08 s × NoAttention × L5: +4.261 pp |
| Run1 non-diagonal mask pairs with Jaccard ≥ 0.90 | 52 |

High recall-vector similarity did not translate into strong sample-level or CW-set agreement. The results therefore distinguish three claims:

1. **Correlation/similarity** describes resemblance between two fitted-model effect patterns.
2. **Double-ablation interaction** is a difference-in-differences quantity and describes non-additivity within the frozen classifier.
3. **Causal biological interpretation** is not established by either quantity and would require independent recording/animal-level validation and direct neural intervention.

Some significant centers did not intersect any retained legacy window. In particular, L3 +0.42 s and L3 +0.65 s have zero actual mask hits in both stages. Their zero effects are exposure-unavailable results, not evidence that those neural times are biologically irrelevant.

## QA summary

- The exact 24-point gate passed.
- Full clean predictions reproduce the archived sample indices, truth labels, and predicted labels exactly.
- Maximum clean probability deviation from the archived Full outputs is `4.4107437e-06`, below the recorded `2e-05` tolerance.
- Full and all three module-ablated clean metrics reproduce the coherent structural table with zero absolute metric difference.
- All architectures use identical saved fold test indices.
- The union of point-wise masks within each label equals the earlier label-level maximum-replacement mask.
- Checkpoint SHA-256 values, data paths, split files, environment, analysis-script hash, and nearby repository commit are recorded under `qa/`.
- Recording-file block bootstrap is used for sample-effect correlation confidence intervals when a single recording can be recovered. Mixed/discontinuous windows remain in point estimates but are excluded from those CIs.

## Directory map

- `RESULTS_SUMMARY_CN.md`: concise Chinese scientific interpretation.
- `paper_tables.xlsx`: formatted 11-sheet workbook containing the main results and QA tables.
- `figures/`: eight figures in PNG, SVG, PDF, and TIFF.
- `source_data/`: one CSV for every figure plus complete compressed stage-wise predictions and paired effects.
- `tables/`: condition metrics, confusion matrices, ΔFP/ΔFN, A–G association tables, and interaction tables.
- `qa/`: baseline reproduction, split alignment, mask equivalence/coverage/overlap, checkpoint hashes, provenance, and workbook previews.
- `run_analysis.py`: complete analysis and plotting implementation.
- `build_paper_tables.mjs` and `validate_paper_tables.mjs`: workbook generation and post-export validation.
- `LITERATURE_NOTES.md`: method literature used for design and interpretation only.
- `FIGURE_CONTRACT.md`: figure claims, encodings, and review risks.

The compressed sample-level files contain predictions, probabilities/logits, fold and dataset indices, pseudonymous recording filenames, window-bin coordinates, CC/CW/WC/WW transitions, actual mask-hit flags, and replaced-bin counts. They do not contain the underlying neural-feature arrays.
