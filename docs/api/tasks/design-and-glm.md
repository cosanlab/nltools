---
title: Design matrices, HRF & GLM
---

# Design matrices, HRF & GLM

Build a first-level model. `events_to_dm` (from `nltools.io`) turns an events table into a [DesignMatrix](../data/design_matrix.md), which convolves with the canonical Glover HRF; for any other kernel, sample one with [nilearn's HRF functions](https://nilearn.github.io/stable/modules/glm.html) and pass the array to `convolve`. `regress` is the standalone numpy GLM. For 4D data use `BrainData.fit(model='glm')`, which raises `RankDeficientDesignWarning` when a design is rank deficient.

## Events to design matrix

::: nltools.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - events_to_dm

## Regression

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - regress

## Design warnings

::: nltools.data.braindata.modeling
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - RankDeficientDesignWarning

::: nltools.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - DesignMatrixWarning
        - ResamplingWarning
