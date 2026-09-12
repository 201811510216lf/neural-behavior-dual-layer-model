# Attention weights shuffled across time

Frozen legacy protocol: 5 s, 50% overlap, constant stage label, saved three-fold splits, no retraining.

## Three-fold summary

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 73.20 ± 0.63; Recall 70.29 ± 1.66; F1 71.13 ± 1.39.
- Ablate L2: Acc 73.30 ± 2.43; Recall 70.94 ± 3.94; F1 71.13 ± 4.89.
- Ablate L3: Acc 74.76 ± 1.09; Recall 72.21 ± 1.45; F1 73.19 ± 0.77.
- Ablate L4: Acc 73.55 ± 0.74; Recall 70.86 ± 2.53; F1 71.27 ± 3.03.
- Ablate L5: Acc 73.58 ± 0.77; Recall 70.58 ± 2.78; F1 70.86 ± 2.35.

## Drop versus baseline

- Ablate L1: Acc 1.50 pp; Recall 1.56 pp; F1 1.68 pp; true-class probability 0.98 pp; logit margin 0.0799.
- Ablate L2: Acc 1.41 pp; Recall 0.91 pp; F1 1.67 pp; true-class probability 2.40 pp; logit margin 0.3317.
- Ablate L3: Acc -0.05 pp; Recall -0.36 pp; F1 -0.39 pp; true-class probability 0.10 pp; logit margin -0.0042.
- Ablate L4: Acc 1.15 pp; Recall 0.99 pp; F1 1.54 pp; true-class probability 1.56 pp; logit margin 0.2251.
- Ablate L5: Acc 1.12 pp; Recall 1.27 pp; F1 1.94 pp; true-class probability 2.15 pp; logit margin 0.2677.
