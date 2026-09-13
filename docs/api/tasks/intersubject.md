---
title: Intersubject correlation
---

# Intersubject correlation

Measure time-locked responses shared across subjects. `isc` correlates subjects with one another — every pair by default, or each subject against the mean of the others with `summary_statistic='leave-one-out'` — and bootstraps a confidence interval. `isfc` does the same across regions, `isps` measures phase synchrony, and `isc_group` compares two groups by permutation.

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - isc
        - isfc
        - isps
        - isc_group
