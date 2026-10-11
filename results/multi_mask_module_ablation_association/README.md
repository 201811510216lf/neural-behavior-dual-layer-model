# Multi-region neural masking × module-ablation association

This bundle extends the frozen-model single-timepoint analysis to eight preregistered pair masks and two gate-triggered triple masks. It contains all source data, sample-level outputs, QA records, publication figures, and executable code used for the results.

## Locked protocol

- No model was trained, fine-tuned, or repartitioned.
- Full, NoConv, NoLSTM, and NoAttention retain the coherent three-fold checkpoints and the original saved Run1/Run2 test indices.
- Legacy preprocessing remains 5 s windows, 10 ms/bin, 500 bins, a 250-bin step, stage-label-constant selection, and the original hierarchical stage definitions.
- Every constituent region uses the historical `[center-1 s, center+1 s)` mask, clipped to the current window.
- Pair/triple masks are Boolean unions computed on the original clean window; overlap is replaced once.
- Fill values are the per-channel maxima of that original, unmodified 5 s window.
- Deltas are always `ablated - baseline`.
- Sample-level analysis stays Run1/Run2 stage-wise; no artificial five-class sample table is created.

## Condition selection

The candidate pool was fixed from the seven highest archived single-mask × NoConv Pearson results, not selected after viewing pair results. The primary shortlist contains eight pairs. Candidate geometry was audited before inference; the highly redundant L2 -0.39/L2 -0.57 geometry was excluded from the primary set. The prespecified gate—any positive pair NoConv Pearson gain over the best constituent single—passed, so two representative triples were run. No exhaustive search over 276 pairs or 2,024 triples was performed.

## Main findings

| Finding | Result |
|---|---:|
| Pair with highest NoConv Pearson | L1 -0.69 s + L2 +0.06 s: 0.859 |
| Largest pair NoConv Pearson gain | L3 -0.08 s + L2 +0.06 s: +0.064 |
| Number of pairs with positive NoConv Pearson gain | 2 / 8 |
| Best triple NoConv Pearson | L1 -0.69 s + L3 -0.08 s + L2 +0.06 s: 0.878 |
| Best triple gain over constituent singles | -0.003 |
| Largest absolute sample-level ΔPtrue Spearman, all modules | 0.106 |
| Largest pooled CW Jaccard, all modules | 0.070 |
| NoConv real-vs-null minimum empirical p (Pearson) | 0.1876 |
| NoConv Pearson-family minimum BH q | 0.9900 |

The pair result is not a monotonic single→pair→triple accumulation: only two pair Pearson gains are positive, the strongest pair's Spearman/cosine gains are 0.000/+0.003, and neither triple exceeds its best constituent single. Sample-level probability associations and damage-set overlap remain weak. Most pair effects are descriptively additive; one pair (L2 -0.57 s + L2 +0.82 s) is sub-additive/redundant or compensatory. Triple pair-adjusted residuals are numerically zero, indicating that observed triple effects are explained by lower-order terms under this descriptive formulation.

## Matched random control

Each real condition has 500 within-retained-window shifted-centre controls. The controls preserve mask order, stage, affected-window coverage, replacement rule, clipping, and approximate union size/component geometry. All affected-window counts match exactly. The maximum observed stage-specific union-bin relative error is 5.86%, and condition-level 95th percentiles are 1.04%–4.02%.

No real NoConv Pearson correlation or gain exceeds the matched null at conventional thresholds. The highest NoConv percentile is 83.4% (empirical p=0.1876; global BH q=1.0). Therefore, the result supports output-level similarity associated with perturbation extent, not region-specific Conv correspondence.

## Directory map

- `MULTI_MASK_RESULTS_SUMMARY_CN.md`: Chinese Q1–Q10 result interpretation.
- `tables/multi_mask_summary_for_group_meeting.csv`: compact group-meeting table.
- `figures/`: 12 figures in 300 dpi PNG, 600 dpi TIFF, vector PDF, and SVG.
- `source_data/`: complete figure source data, metrics, interactions, matched-null tables, and compressed stage-wise predictions.
- `qa/`: split/checkpoint/provenance audits, GPU–CPU class equivalence, output validation, montage, and grayscale previews.
- `run_multi_mask_analysis.py`: end-to-end reproducible implementation with a non-inference finalization mode.
- `LITERATURE_NOTES.md`: design/interpretation literature inherited from the audited single-mask analysis.
- `FIGURE_CONTRACT.md`: figure claims, encodings, and review risks.

The prediction files contain pseudonymous recording filenames and derived OOF predictions/probabilities, but no neural-feature arrays, raw electrophysiology, checkpoints, or animal/group metadata.
