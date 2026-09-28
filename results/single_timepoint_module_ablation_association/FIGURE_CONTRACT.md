# Figure contract

Core conclusion: frozen-model single-timepoint perturbations and structural ablations show quantifiable but non-identical output-effect patterns, with structure-dependent double-ablation interactions and strong mask-overlap constraints.

Figure archetype: quantitative grid.

Target/output: manuscript-ready double-column figures; Python/matplotlib-seaborn only; editable SVG/PDF plus 600-dpi TIFF and PNG preview.

Panel map:

- Fig. 1: 24 × 5 data-ablation ΔRecall (primary input-effect evidence).
- Fig. 2: 3 × 5 module-ablation ΔRecall (structural comparison).
- Fig. 3: 24 × 3 recall-vector Spearman (similarity, not causality).
- Fig. 4: 24 × 3 CW-set Jaccard (sample damage overlap).
- Fig. 5: one-vs-rest ΔFP / ΔFN (error decomposition).
- Fig. 6: double-ablation recall interaction (structure dependence).
- Fig. 7: selected ΔPtrue scatter (sample-level association).
- Fig. 8: actual point-mask overlap (critical QA/control).

Statistics: three saved folds; same OOF indices; recording-file block bootstrap for sample-effect Spearman CI; no naive overlapping-window p-values.

Source data: one dedicated CSV per figure under `source_data/`.

Reviewer risks: high ±1 s overlap among adjacent 10-ms centers; hierarchical Run1/Run2 sample definitions; optimistic legacy split; perturbation may be out-of-distribution; model intervention is not biological causality.
