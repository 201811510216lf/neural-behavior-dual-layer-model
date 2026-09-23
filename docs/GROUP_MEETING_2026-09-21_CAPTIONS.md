# Figure legends and table titles for the 2026-09-21 group meeting

This document maps the figures and tables in `26-9-21组会汇报.pptx` to concise, publication-style English captions. Figure and table numbers are intentionally not fixed because the presentation currently contains non-consecutive numbering.

## Slide 1

**Figure caption**

Experimental workflow and two-stage hierarchical neural-behavior classification model. (A) Synchronized acquisition of neural signals and behavioral video. (B) Hierarchical training and inference with first-stage coarse classification and second-stage five-class behavioral decoding. (C) Detailed network architecture comprising convolutional, LSTM, temporal-attention, and fully connected modules.

## Slide 2

**Table title**

Performance comparison of the proposed hierarchical model with conventional machine-learning classifiers.

## Slide 3

**Upper table**

Overall classification performance of the same model on the full, WT, and KO datasets.

**Lower table**

Per-class recognition accuracy of the same model on the full, WT, and KO datasets.

## Slide 4

**Upper table**

Overall performance of the proposed model and its structural ablation variants under the original evaluation protocol.

**Lower table**

Per-class recognition accuracy of the proposed model and its structural ablation variants under the original evaluation protocol.

## Slide 5

**Upper table**

Composition of the six experimental groups and corresponding animal identifiers.

**Middle table**

Temporal coverage of valid behavioral labels and invalid or unlabeled periods.

**Lower table**

Bout counts, total durations, and temporal fractions of the five behavioral categories.

## Slide 6

**Figure caption**

Behavior-onset-aligned neural activity in the two pooled experimental cohorts. Group 1 comprises original groups 1, 2, 4, and 6, whereas Group 2 comprises groups 3 and 5. Curves show the event-averaged spike z-score from −2 to +2 s relative to behavior onset for the five behavioral categories; shaded bands indicate SEM, and the vertical line denotes behavior onset.

## Slide 7

**Figure caption**

Behavior-onset-aligned neural activity across the six original experimental groups. Curves show the event-averaged spike z-score from −2 to +2 s relative to behavior onset for Hole Exploration, Wrong Hole Exploration, Hesitating, Changing Direction, and Walking; shaded bands indicate SEM, and the vertical line denotes behavior onset.

## Slide 8

**Upper figure**

Exploratory between-group difference in Hole Exploration-related neural activity during the 10-ms window beginning at −0.69 s relative to behavior onset (p = 0.0344). The local trace covers −0.79 to −0.59 s.

**Lower figure**

Exploratory between-group difference in Hole Exploration-related neural activity during the 10-ms window beginning at −0.15 s relative to behavior onset (p = 0.0367). The local trace covers −0.25 to −0.05 s.

## Slide 9

**Upper figure**

Exploratory between-group difference in Wrong Hole Exploration-related neural activity during the 10-ms window beginning at −0.39 s relative to behavior onset (p = 0.0384). The local trace covers −0.49 to −0.29 s.

**Lower figure**

Exploratory between-group difference in Wrong Hole Exploration-related neural activity during the 10-ms window beginning at +0.06 s relative to behavior onset (p = 0.0138). The local trace covers −0.04 to +0.16 s.

## Slide 10

**Upper figure**

Exploratory between-group difference in Wrong Hole Exploration-related neural activity during the 10-ms window beginning at +0.46 s relative to behavior onset (p = 0.0295). The local trace covers +0.36 to +0.56 s.

**Lower figure**

Exploratory between-group difference in Wrong Hole Exploration-related neural activity during the 10-ms window beginning at +0.82 s relative to behavior onset (p = 0.0263). The local trace covers +0.72 to +0.92 s.

## Slide 11

**Figure caption**

Exploratory between-group difference in Hesitating-related neural activity during the 10-ms window beginning at +0.95 s relative to behavior onset (p = 0.0097). The local trace covers +0.85 to +1.05 s.

## Slide 12

**Figure caption**

Exploratory between-group difference in Changing Direction-related neural activity during the 10-ms window beginning at −0.55 s relative to behavior onset (p = 0.0454). The local trace covers −0.65 to −0.45 s.

## Slide 13

**Upper figure**

Exploratory between-group difference in Walking-related neural activity during the 10-ms window beginning at +0.33 s relative to behavior onset (p = 0.0135). The local trace covers +0.23 to +0.43 s.

**Lower figure**

Exploratory between-group difference in Walking-related neural activity during the 10-ms window beginning at +0.70 s relative to behavior onset (p = 0.0326). The local trace covers +0.60 to +0.80 s.

### Common statistical note for slides 8–13

In each figure, panel a shows the complete −2 to +2 s peri-onset trace, panel b shows the corresponding 200-ms local interval, and panel c shows the event-level group mean ± SEM in the tested 10-ms window. The solid black line marks the beginning of the tested window, whereas 0 s denotes behavior onset. P values were obtained using uncorrected two-sided, equal-variance independent-samples t-tests.

## Slide 14

**Table title**

Overall and per-class end-to-end five-class classification counts under baseline and maximum-value masking conditions.

**Table note**

Results are reported as correct/total window counts on the same set of 715 held-out samples.

## Slide 15

**Figure caption**

End-to-end five-class confusion matrices under baseline and maximum-value masking conditions. (a) Baseline; (b) Mask L1; (c) Mask L2; (d) Mask L3; (e) Mask L4; and (f) Mask L5. Rows represent true behavioral labels, columns represent final predicted labels, and cell values indicate window counts. All panels use the same color scale.

## Interpretation safeguards

- Slide 4 reports structural module ablation, whereas slides 14–15 report maximum-value time-window masking. These are distinct experimental protocols and should not share one generic ablation label.
- The time-point results on slides 8–13 are event-level exploratory comparisons selected from multiple uncorrected tests. They should not be described as confirmatory evidence without an animal-level analysis and appropriate multiple-comparison control.
- The source data and plotting code define the shaded traces and bar-plot error bars as SEM.
