---
title: Design matrices, HRF & GLM
---

# Design matrices, HRF & GLM

Build a first-level model. `events_to_dm` turns an events table into a [DesignMatrix](../data/design_matrix.md); the HRF functions sample the SPM and Glover responses and their derivatives for convolution. `regress` is the standalone numpy GLM. For 4D data use `BrainData.fit(model='glm')`, which raises the warning classes listed here when a design is rank-deficient or nearly collinear.

## Events to design matrix

::: nltools.data.designmatrix.io
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - events_to_dm

## HRF models and regression

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - spm_hrf
        - spm_time_derivative
        - spm_dispersion_derivative
        - glover_hrf
        - glover_time_derivative
        - glover_dispersion_derivative
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
        - NearCollinearDesignWarning

::: nltools.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - DesignMatrixWarning
        - ResamplingWarning
