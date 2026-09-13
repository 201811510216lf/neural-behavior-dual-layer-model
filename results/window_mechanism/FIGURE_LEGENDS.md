# Figure legends

**Figure A | Significant-window perturbations have behavior-specific effects.**
**a,** Matrix of recall change after replacing each mapped significant segment with the corresponding channel-wise maximum from the original 5 s window. Each cell is the mean of three saved folds; negative values indicate impaired recall. **b,** Fold-level values for the five prespecified cells supporting the L2/L3/L5 interpretation. Open circles are individual folds (n = 3); diamonds are fold means. The perturbations use frozen Full models and do not involve retraining.

**Figure B | Architecture by temporal-perturbation interaction.**
**a-c,** Mean change in the prespecified outcome for Full, NoConv, NoLSTM, and NoAttention models under maximum replacement, attention shuffle, attention zeroing, and a matched-length random-window control. Structural variants were retrained with the saved legacy folds; perturbations were then applied to frozen fitted models. **d-f,** Paired fold-level deltas for the prespecified Full-versus-ablated architecture contrasts. Lines join the same fold. The interaction is defined as delta(Full) - delta(NoModule). Exact two-sided sign-flip tests and Benjamini-Hochberg adjusted values are reported in Table A. With n = 3 folds, these tests are descriptive and have a minimum attainable two-sided p value of 0.25.

**Figure C | Signal content, temporal allocation, and fine-scale local order contribute differently.**
Fold-level outcome changes after maximum replacement, attention-weight shuffling, attention zeroing, and 50/100/200 ms local block-order shuffling for **a,** L2 and Stage-2 macro recall, **b,** L3 and Hesitating recall, and **c,** L5 and Walking recall. Open points are individual folds (n = 3); diamonds are fold means. Negative values indicate worse performance. Attention shuffle averages 20 seeded permutations per affected batch.

**Figure D | Matched random-window null controls.**
The primary null circularly shifts each affected sample's exact binary mask by a nonzero random offset, preserving both the number and geometry of replaced bins; a secondary matched-count control replaces a randomly positioned contiguous segment of equal bin count. Both use the channel-wise maximum from the unchanged 5 s sample and 500 repeats per fold for the frozen Full model. Thin orange lines/triangles show the observed maximum-replacement effect in each fold; the thick line shows their mean. Reported p values refer to the exact-mask shape-matched null and are one-sided empirical tail probabilities with the plus-one correction. Sparse L3 mapped-window coverage can produce a degenerate or weakly informative null and must not be interpreted as evidence of absence.

**Supplementary Figure E | L3 necessity/sufficiency diagnostic.**
Hesitating recall for the frozen Full model at baseline, after L3 maximum replacement (Drop L3), and when only mapped L3 bins are retained while all other bins are replaced by each channel's 5 s mean (Keep L3 only). Lines connect the same saved fold (n = 3); diamonds show means. Because L3 contributes only two Run-1 and three Run-2 center occurrences across all legacy windows before fold selection, this diagnostic is coverage-limited.

**Statistics and scope.**
All performance values are percentages and all deltas are percentage points. Fold summaries report sample standard deviation. Bootstrap intervals use 10,000 deterministic resamples of the three paired folds and are descriptive at this sample size. These experiments quantify fitted-model sensitivity; they do not establish neural causality. The legacy protocol uses overlapping 5 s windows and window-level splits, and the original model-selection procedure selected the best epoch on the held-out fold, so absolute performance is optimistic.
