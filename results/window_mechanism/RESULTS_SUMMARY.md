# Results summary

All deltas are perturbation minus baseline in percentage points. Negative values indicate worse performance.

## Window-level findings

- **L2:** mean target-outcome delta -3.63 +/- 2.37 pp across three folds; mapped centers run1/run2 = 665/57. Candidate shared Stage-2 temporal information.
- **L3:** mean target-outcome delta 0.83 +/- 1.44 pp across three folds; mapped centers run1/run2 = 2/3. Inconclusive: weak fitted-model effect but severely limited mapped coverage.
- **L5:** mean target-outcome delta -5.87 +/- 3.51 pp across three folds; mapped centers run1/run2 = 1545/50. Candidate Walking-dominant content and temporal-allocation dependence.

## Random-window specificity

- L2: foldwise one-sided empirical p = 0.9681, 0.9960, 0.8323.
- L3: foldwise one-sided empirical p = 1.0000, 0.9840, 1.0000.
- L5: foldwise one-sided empirical p = 0.7725, 0.5369, 1.0000.

## Mandatory limitations

- n=3 folds makes interaction tests low-powered; exact two-sided sign-flip p cannot be smaller than 0.25.
- Legacy overlapping-window, window-level splits limit train/test independence.
- The legacy checkpoint protocol selects the best epoch on the held-out fold and therefore gives optimistic absolute performance.
- L3 coverage is exceptionally sparse, so a near-zero perturbation effect is not evidence that L3 contains no behavioral information.
- Pointwise group p values are uncorrected in the supplied table and are not equivalent to model-decisive evidence.
