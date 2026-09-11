---
title: Prediction & cross-validation
---

# Prediction & cross-validation

Decode or predict from brain data. `BrainData.predict` runs the workflow. Listed here are the pieces it accepts or returns: the cross-validation schemes (`resolve_cv` turns an int, a name, or an sklearn splitter into one), `Roc` for a classifier's output, and the plots of weights, margins, and predictions. The `Ridge` estimator behind `model='ridge'` (CPU or GPU) is documented with the other [models](../models.md).

## Cross-validation

::: nltools.cross_validation
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - KFoldStratified
        - resolve_cv

## Classifier output

::: nltools.data.roc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - Roc

## Prediction plots

::: nltools.plotting
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - plot_roc
        - plot_dist_from_hyperplane
        - plot_probability
        - plot_scatter
