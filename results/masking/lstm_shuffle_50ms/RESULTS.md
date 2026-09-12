# Critical-region block-order shuffle — block_50ms

Frozen legacy protocol: 5 s, 50% overlap, constant stage labels, saved three-fold splits, no retraining.

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.71 ± 0.38; Recall 71.89 ± 2.26; F1 72.86 ± 1.50.
- Ablate L2: Acc 74.91 ± 0.75; Recall 72.06 ± 2.01; F1 72.95 ± 1.41.
- Ablate L3: Acc 74.52 ± 0.70; Recall 71.81 ± 1.74; F1 72.67 ± 1.11.
- Ablate L4: Acc 74.40 ± 0.92; Recall 71.63 ± 1.65; F1 72.53 ± 0.99.
- Ablate L5: Acc 74.29 ± 1.01; Recall 71.56 ± 1.59; F1 72.46 ± 0.95.

## Drop versus baseline

- Ablate L1: Acc -0.00 pp; Recall -0.03 pp; F1 -0.06 pp; true probability -0.06 pp; logit margin -0.0118.
- Ablate L2: Acc -0.21 pp; Recall -0.21 pp; F1 -0.15 pp; true probability 0.12 pp; logit margin 0.0076.
- Ablate L3: Acc 0.18 pp; Recall 0.04 pp; F1 0.13 pp; true probability 0.20 pp; logit margin 0.0391.
- Ablate L4: Acc 0.31 pp; Recall 0.22 pp; F1 0.27 pp; true probability 0.49 pp; logit margin 0.0740.
- Ablate L5: Acc 0.41 pp; Recall 0.29 pp; F1 0.34 pp; true probability 0.39 pp; logit margin 0.0511.
