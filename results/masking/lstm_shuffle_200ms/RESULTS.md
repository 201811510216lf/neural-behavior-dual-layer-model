# Critical-region block-order shuffle — block_200ms

Frozen legacy protocol: 5 s, 50% overlap, constant stage labels, saved three-fold splits, no retraining.

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.71 ± 0.38; Recall 71.86 ± 2.27; F1 72.87 ± 1.50.
- Ablate L2: Acc 74.33 ± 0.27; Recall 71.63 ± 2.41; F1 72.52 ± 1.69.
- Ablate L3: Acc 74.75 ± 0.75; Recall 71.99 ± 1.64; F1 72.91 ± 1.05.
- Ablate L4: Acc 74.35 ± 0.81; Recall 71.60 ± 1.72; F1 72.46 ± 1.10.
- Ablate L5: Acc 74.40 ± 0.98; Recall 71.63 ± 1.60; F1 72.41 ± 0.93.

## Drop versus baseline

- Ablate L1: Acc -0.00 pp; Recall -0.01 pp; F1 -0.06 pp; true probability 0.13 pp; logit margin 0.0231.
- Ablate L2: Acc 0.38 pp; Recall 0.22 pp; F1 0.29 pp; true probability 0.52 pp; logit margin 0.0818.
- Ablate L3: Acc -0.05 pp; Recall -0.14 pp; F1 -0.11 pp; true probability -0.02 pp; logit margin 0.0067.
- Ablate L4: Acc 0.36 pp; Recall 0.26 pp; F1 0.35 pp; true probability 0.37 pp; logit margin 0.0615.
- Ablate L5: Acc 0.31 pp; Recall 0.22 pp; F1 0.39 pp; true probability 0.34 pp; logit margin 0.0352.
