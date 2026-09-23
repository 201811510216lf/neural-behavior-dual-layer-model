# Experiment index

This index maps the group-meeting presentation to the public release.

| Presentation section | Public artifact | Notes |
|---|---|---|
| Study and grouping design | `DATA_AVAILABILITY.md` | Describes six groups and the 1/2/4/6 versus 3/5 regrouping without publishing animal-level tables |
| Two-stage model | `src/dual_layer_model/` | Conv, two-layer LSTM, temporal attention, and hierarchical five-class merge |
| Classical model comparison | `results/model/model_comparison_ppt.csv` | Archived presentation snapshot |
| Dataset comparison | `results/model/dataset_performance_ppt.csv` | Control, experimental, and combined results from the presentation |
| Structural ablation | `results/model/structural_ablation_*.csv` | No Attention, No Conv, No LSTM, and full model |
| Two-group p-value figures | `results/group_differences/` | Ten selected time windows and their displayed figures |
| Maximum/mean/minimum replacement | `results/masking/segment_replacement/` | Frozen-model ±1 s replacement around selected time points |
| Attention shuffle and zero | `results/masking/attention_*` | Perturbs attention-to-time correspondence or key attention weights |
| Conv mean replacement | `results/masking/conv_mean_replacement/` | Mean-replacement baseline |
| LSTM shuffle | `results/masking/lstm_shuffle_*` | 50, 100, and 200 ms block-order perturbations used in the presentation |
| Additional neural-model analyses | `results/association/` | Baseline reproduction, RSA, CKA, decision geometry, overlap dependence, and functional interventions |
| Window × behavior and architecture interactions | `results/window_mechanism/` | Figures A–D and Supplementary Figure E, fold-level aggregate source data, exact sign-flip/FDR tables, 500-repeat matched-null controls, and reproducible code |
| 2026-09-21 presentation captions | `docs/GROUP_MEETING_2026-09-21_CAPTIONS.md` | Slide-by-slide figure legends, table titles, statistical wording, and interpretation safeguards |

## Main result snapshots

The presentation contains two evaluation snapshots:

- Early comparison snapshot: proposed model accuracy 77.83%, F1 79.34%, recall 79.57%.
- Reproduced legacy protocol: accuracy 74.70 ± 0.70%, F1 72.80 ± 1.22%, recall 71.85 ± 1.82%.

They come from different experimental snapshots and are kept separate in this repository.

## Masking interpretation

Maximum replacement caused the clearest decrease among the value-replacement tests, with the L5 condition reducing F1 from 72.80% to 70.55%. Mean and minimum replacement produced smaller changes. Attention shuffling generally caused larger losses than zeroing selected attention weights, suggesting that misallocated temporal weighting can be more disruptive than suppressing the selected contribution. Local LSTM shuffling at 50–200 ms produced only small changes and did not show a clear monotonic time-scale effect.

## Updated matched-null conclusion

The new exact-mask circular-shift controls preserve the geometry and bin count of each selected mask. Across L2, L3, and L5, none of the observed maximum-replacement effects was more extreme than this null in any fold. The architecture interaction tests were also non-significant after BH-FDR correction (all q = 1.0). These results retain the perturbation effects as useful exploratory sensitivity findings but do not support a claim that the originally selected time positions are uniquely causal.
