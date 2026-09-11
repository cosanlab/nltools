---
title: Preprocessing & signal
---

# Preprocessing & signal

Clean timeseries before modelling. Standardize or trim outliers, flag motion spikes, resample to another sampling rate, build cosine drift regressors. Every function takes and returns numpy arrays or DataFrames. The [BrainData](../data/brain_data.md) and [DesignMatrix](../data/design_matrix.md) methods of the same name call them.

::: nltools.algorithms
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 2
      members:
        - zscore
        - trim
        - winsorize
        - find_spikes
        - downsample
        - upsample
        - make_cosine_basis
        - calc_bpm
