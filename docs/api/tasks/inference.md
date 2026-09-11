---
title: Statistics & inference
---

# Statistics & inference

Non-parametric group statistics. The one-sample, two-sample, and timeseries permutation tests run on CPU or GPU (`device=`); `phase_randomize` and `circle_shift` are the timeseries null models. `fdr`, `holm_bonf`, `threshold`, and `multi_threshold` correct or threshold the resulting p-maps. `BrainData.ttest`, `Adjacency.ttest`, and `BrainData.bootstrap` call these.

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - one_sample_permutation_test
        - two_sample_permutation_test
        - timeseries_correlation_permutation_test
        - phase_randomize
        - circle_shift
        - fdr
        - holm_bonf
        - threshold
        - multi_threshold
