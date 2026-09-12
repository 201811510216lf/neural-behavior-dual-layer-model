# 5-s window per-neuron minimum replacement results

Protocol: frozen 5-s windows, 50% overlap, constant stage labels, saved three-fold indices, and original run1/run2 model merging.
Each p<0.05 10-ms center was expanded to a clipped +/-1-s segment and replaced per neural channel.

## Three-fold summary

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.47 ± 0.62; Recall 71.70 ± 1.88; F1 72.66 ± 1.22.
- Ablate L2: Acc 75.07 ± 1.82; Recall 72.25 ± 2.45; F1 73.09 ± 2.53.
- Ablate L3: Acc 74.47 ± 0.62; Recall 71.67 ± 1.89; F1 72.56 ± 1.24.
- Ablate L4: Acc 74.35 ± 0.85; Recall 71.60 ± 1.69; F1 72.46 ± 1.05.
- Ablate L5: Acc 74.09 ± 1.62; Recall 71.41 ± 0.48; F1 72.29 ± 0.45.

## Change versus baseline

- Ablate L1: Acc drop 0.23 pp; Recall drop 0.15 pp; F1 drop 0.14 pp.
- Ablate L2: Acc drop -0.37 pp; Recall drop -0.40 pp; F1 drop -0.28 pp.
- Ablate L3: Acc drop 0.23 pp; Recall drop 0.18 pp; F1 drop 0.24 pp.
- Ablate L4: Acc drop 0.36 pp; Recall drop 0.26 pp; F1 drop 0.35 pp.
- Ablate L5: Acc drop 0.61 pp; Recall drop 0.44 pp; F1 drop 0.52 pp.
