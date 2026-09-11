---
title: Intersubject correlation
---

# Intersubject correlation

Measure time-locked responses shared across subjects. `isc` correlates each subject's timeseries with the rest of the group and bootstraps a confidence interval. `isfc` does the same across regions, `isps` measures phase synchrony, and `isc_group` compares two groups by permutation. The two `*_permutation_test` functions are the CPU/GPU engines (`device=`) underneath, shared with the [permutation tests](inference.md).

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
        - isc_permutation_test
        - isc_group_permutation_test
