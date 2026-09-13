---
title: Prediction & cross-validation
---

# Prediction & cross-validation

Decode or predict from brain data. `BrainData.predict` runs the workflow. Listed here are the pieces it accepts or returns: the cross-validation schemes and `Roc` for a classifier's output. Ridge encoding runs through `BrainData.fit(model='ridge')`, on CPU or GPU.

## Cross-validation

::: nltools.cross_validation
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - KFoldStratified

## Classifier output

::: nltools.data.roc
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_category_heading: false
      heading_level: 3
      members:
        - Roc
