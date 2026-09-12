# Critical-region Attention weights zeroed and renormalized

Frozen legacy protocol: 5 s, 50% overlap, constant stage label, saved three-fold splits, no retraining.

## Three-fold summary

- Our Proposed: Acc 74.70 ± 0.70; Recall 71.85 ± 1.82; F1 72.80 ± 1.22.
- Ablate L1: Acc 74.70 ± 0.70; Recall 71.91 ± 1.87; F1 72.77 ± 1.12.
- Ablate L2: Acc 75.23 ± 1.52; Recall 72.37 ± 2.66; F1 73.28 ± 2.56.
- Ablate L3: Acc 74.47 ± 0.62; Recall 71.67 ± 1.89; F1 72.56 ± 1.24.
- Ablate L4: Acc 74.29 ± 0.84; Recall 71.56 ± 1.71; F1 72.42 ± 1.06.
- Ablate L5: Acc 73.85 ± 1.38; Recall 71.02 ± 1.12; F1 71.86 ± 0.98.

## Drop versus baseline

- Ablate L1: Acc 0.00 pp; Recall -0.06 pp; F1 0.04 pp; true-class probability -0.34 pp; logit margin -0.0597.
- Ablate L2: Acc -0.52 pp; Recall -0.52 pp; F1 -0.48 pp; true-class probability -0.67 pp; logit margin -0.0611.
- Ablate L3: Acc 0.23 pp; Recall 0.18 pp; F1 0.24 pp; true-class probability 0.24 pp; logit margin 0.0415.
- Ablate L4: Acc 0.41 pp; Recall 0.29 pp; F1 0.39 pp; true-class probability 0.41 pp; logit margin 0.0743.
- Ablate L5: Acc 0.85 pp; Recall 0.84 pp; F1 0.95 pp; true-class probability 0.54 pp; logit margin 0.0705.
