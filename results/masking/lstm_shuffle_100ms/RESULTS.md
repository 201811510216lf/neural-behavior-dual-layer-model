# Critical-region block-order shuffle — block_100ms

Frozen legacy protocol: 5 s, 50% overlap, constant stage labels, saved three-fold splits, no retraining.

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.24 ± 0.48; Recall 71.47 ± 2.62; F1 72.49 ± 1.81.
- Ablate L2: Acc 74.40 ± 0.26; Recall 71.63 ± 2.52; F1 72.44 ± 1.83.
- Ablate L3: Acc 74.75 ± 0.75; Recall 71.99 ± 1.64; F1 72.91 ± 1.05.
- Ablate L4: Acc 74.35 ± 0.95; Recall 71.60 ± 1.62; F1 72.42 ± 0.99.
- Ablate L5: Acc 74.35 ± 0.95; Recall 71.60 ± 1.62; F1 72.42 ± 0.92.

## Drop versus baseline

- Ablate L1: Acc 0.46 pp; Recall 0.38 pp; F1 0.32 pp; true probability 0.19 pp; logit margin 0.0119.
- Ablate L2: Acc 0.31 pp; Recall 0.22 pp; F1 0.37 pp; true probability 0.42 pp; logit margin 0.0569.
- Ablate L3: Acc -0.05 pp; Recall -0.14 pp; F1 -0.11 pp; true probability 0.19 pp; logit margin 0.0381.
- Ablate L4: Acc 0.36 pp; Recall 0.26 pp; F1 0.39 pp; true probability 0.45 pp; logit margin 0.0745.
- Ablate L5: Acc 0.36 pp; Recall 0.26 pp; F1 0.39 pp; true probability 0.41 pp; logit margin 0.0528.
