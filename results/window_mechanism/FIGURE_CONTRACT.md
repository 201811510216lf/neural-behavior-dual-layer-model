# Figure contract

## Scientific claim

The current experiments test whether statistically significant neural time windows are also decisive for fitted behavior decoders, and which model components exploit those windows.

The working interpretation is deliberately asymmetric:

- L2 may contain temporal information shared by the three Stage-2 behaviors.
- L5 may be especially important for Walking and for the alignment between latent states and attention weights.
- L3 currently shows little fitted-model sensitivity, but its mapped-window coverage is too sparse to claim absence of behavioral information.

## Evidence chain and panel map

1. **Fig A — Window x Behavior:** maximum-replacement delta recall for L1-L5 against all five behaviors.
2. **Fig B — Architecture x Perturbation:** Full, NoConv, NoLSTM, and NoAttention under Baseline, Maximum, Attention shuffle, Attention zero, and matched random-window control.
3. **Fig C — Mechanism contrast:** maximum replacement, attention shuffle/zero, and 50/100/200 ms local LSTM shuffle for L2, L3, and L5.
4. **Fig D — Random-window null:** empirical matched-length null distributions, with the observed L2/L3/L5 maximum effects overlaid.
5. **Table A:** fold-level baseline, perturbed value, delta, uncertainty, interaction effect, and exact sign-flip p value.
6. **Table B:** statistical-window evidence, fitted-model sensitivity, architecture dependence, mapping coverage, and interpretation status.
7. **Supplementary Fig E:** Drop-L3 versus Keep-L3 sensitivity/sufficiency diagnostic, explicitly qualified by sparse coverage.

## Figure archetype

Quantitative grid with one dominant interaction heatmap and supporting small-multiple statistical panels. No decorative schematic is needed.

## Backend

Python only, using the user-specified `pytorch-GPU` environment for both model experiments and plotting.

## Export contract

- Nature-style double-column width: 7.2 inch.
- White background, restrained colorblind-safe palette.
- Final text 6-8 pt; editable SVG text (`svg.fonttype = none`).
- Primary SVG plus PDF, 600 dpi PNG, TIFF, and grayscale preview.
- Every plotted value has a CSV source table.

## Statistical contract

- Fold-level points are shown whenever a summary is plotted (n=3 folds).
- Deltas are paired within fold.
- 95% confidence intervals use a deterministic paired bootstrap over folds and are descriptive at n=3.
- Interaction p values use an exact two-sided sign-flip test; Benjamini-Hochberg adjusted values are also reported.
- Random-window specificity uses an empirical one-sided tail probability with the plus-one correction.
- No uncorrected pointwise group p value is treated as proof of behavioral importance.

## Reviewer risks to keep visible

- The legacy 5 s protocol uses overlapping windows and window-level stratified splits; train/test independence is therefore limited.
- Original checkpoints selected the best epoch on the held-out fold; absolute performance is optimistic.
- Structural retraining and frozen-model perturbation answer different questions.
- Only three folds are available, so interaction statistics are low-powered.
- L3 mapped-window coverage is extremely sparse; null or positive effects are not decisive.
- The source time-point p values are uncorrected unless a corrected column is separately supplied.

