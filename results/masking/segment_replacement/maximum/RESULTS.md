# 5-s window per-neuron maximum replacement results

Protocol: frozen 5-s windows, 50% overlap, constant stage labels, saved three-fold indices, and original run1/run2 model merging.
Each p<0.05 10-ms center was expanded to a clipped +/-1-s segment and replaced per neural channel.

## Three-fold summary

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 73.85 ± 0.65; Recall 71.18 ± 2.31; F1 72.30 ± 1.68.
- Ablate L2: Acc 73.20 ± 1.06; Recall 70.82 ± 3.32; F1 71.42 ± 2.74.
- Ablate L3: Acc 74.29 ± 1.05; Recall 71.63 ± 1.43; F1 72.47 ± 0.77.
- Ablate L4: Acc 74.12 ± 1.27; Recall 71.47 ± 1.52; F1 72.22 ± 1.02.
- Ablate L5: Acc 72.93 ± 1.24; Recall 69.89 ± 0.87; F1 70.55 ± 0.35.

## Change versus baseline

- Ablate L1: Acc drop 0.85 pp; Recall drop 0.67 pp; F1 drop 0.51 pp.
- Ablate L2: Acc drop 1.50 pp; Recall drop 1.03 pp; F1 drop 1.38 pp.
- Ablate L3: Acc drop 0.42 pp; Recall drop 0.22 pp; F1 drop 0.33 pp.
- Ablate L4: Acc drop 0.59 pp; Recall drop 0.38 pp; F1 drop 0.58 pp.
- Ablate L5: Acc drop 1.77 pp; Recall drop 1.96 pp; F1 drop 2.26 pp.
