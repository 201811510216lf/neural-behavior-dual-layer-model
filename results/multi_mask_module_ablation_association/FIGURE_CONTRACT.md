# Figure contract and review risks

All figures use the frozen-model outputs in `source_data/`; no values are manually entered into plotting code. Heatmaps use diverging color scales centered at zero when signs matter and sequential scales for non-negative overlaps. Annotated values remain the primary redundant encoding, so conclusions survive grayscale conversion.

| Figure | Intended claim | Source data | Main review risk |
|---|---|---|---|
| A1–A3 | Single/pair/triple effect-fingerprint similarity for all modules | `figA_single_pair_triple_module_correlation.csv` | Five-behavior correlations are low-dimensional and descriptive |
| B | Distribution and order-wise median across mask order | `figB_correlation_accumulation.csv` | Order groups are not independent confirmatory samples |
| C | Gain over the best constituent single | `figC_gain_over_best_single.csv` | Positive Pearson gain may not replicate in Spearman/cosine |
| D | Pair residual from additive single-mask expectation | `figD_pair_interaction.csv` | Recall boundaries make additivity descriptive, not a biological law |
| E | Same-sample stage-wise ΔPtrue association | `figE_sample_delta_ptrue_correlation.csv` | Run1 and Run2 must not be interpreted as one five-class sample set |
| F | CW damage-set overlap | `figF_CW_overlap.csv` | Small Jaccard values do not imply biological independence |
| G | Real similarity versus matched random multi-mask distribution | `figG_real_vs_random_null_summary.csv`, `figG_real_vs_random_null_distribution.csv.gz` | Null controls perturbation geometry, not all conceivable biological confounds |
| H | Union size, pairwise overlap, and effective added bins | `figH_mask_geometry_QA.csv` | A center denotes a ±1 s mask region, not a 10 ms causal node |

Interpretation must keep three layers separate: correlation/similarity, frozen-model descriptive interaction, and causal biological claims. Only the first two are measured here; causality is not established.
