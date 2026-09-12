---
title: Functional alignment
---

# Functional alignment

Put subjects into a shared functional space. `align` is the whole-brain entry point, with `method='procrustes'` or an SRM variant. `SRM` and `DetSRM` are the sklearn-style estimators behind the SRM variants. `align_states` matches state maps across groups. `BrainData.align` calls `align`.

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - SRM
        - DetSRM
        - align
        - align_states
        - procrustes_distance

::: nltools.algorithms.alignment.procrustes.procrustes
    options:
      heading_level: 3
