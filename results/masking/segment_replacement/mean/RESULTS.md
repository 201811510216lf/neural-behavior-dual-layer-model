# 5-s window per-neuron mean replacement results

Protocol: frozen 5-s windows, 50% overlap, constant stage labels, saved three-fold indices, and original run1/run2 model merging.
Each p<0.05 10-ms center was expanded to a clipped +/-1-s segment and replaced per neural channel.

## Three-fold summary

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.71 ± 0.38; Recall 71.91 ± 2.24; F1 72.84 ± 1.50.
- Ablate L2: Acc 75.25 ± 1.86; Recall 72.43 ± 2.84; F1 73.12 ± 2.93.
- Ablate L3: Acc 74.47 ± 0.62; Recall 71.67 ± 1.89; F1 72.56 ± 1.24.
- Ablate L4: Acc 74.19 ± 0.95; Recall 71.49 ± 1.62; F1 72.30 ± 0.92.
- Ablate L5: Acc 74.09 ± 1.58; Recall 71.41 ± 0.55; F1 72.33 ± 0.62.

## Change versus baseline

- Ablate L1: Acc drop -0.00 pp; Recall drop -0.06 pp; F1 drop -0.04 pp.
- Ablate L2: Acc drop -0.55 pp; Recall drop -0.58 pp; F1 drop -0.32 pp.
- Ablate L3: Acc drop 0.23 pp; Recall drop 0.18 pp; F1 drop 0.24 pp.
- Ablate L4: Acc drop 0.51 pp; Recall drop 0.37 pp; F1 drop 0.50 pp.
- Ablate L5: Acc drop 0.61 pp; Recall drop 0.44 pp; F1 drop 0.48 pp.
