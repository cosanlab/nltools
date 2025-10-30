# Analysis Tools

Specialized analysis tools for model evaluation and diagnostics.

## Overview

The `nltools.analysis` module provides tools for evaluating prediction models and computing diagnostic statistics. Currently includes the ROC (Receiver Operating Characteristic) class for binary classification evaluation.

## Key Classes

**Roc** - Receiver Operating Characteristic analysis
- Compute ROC curves for binary classification
- Calculate AUC (Area Under Curve)
- Optimal threshold selection
- Significance testing via permutation
- Visualization of ROC curves

## Quick Start

```python
from nltools.analysis import Roc

# Create ROC object
roc = Roc(predictions, labels)

# Get AUC
auc = roc.auc

# Get optimal threshold
threshold = roc.threshold()

# Plot ROC curve
roc.plot()

# Permutation test
roc_perm = Roc(predictions, labels, permutation=True, n_permutations=5000)
print(f"p-value: {roc_perm.p_value}")
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.analysis
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - Brain_Data.predict() for generating predictions
- {doc}`models` - Ridge and other models
- {doc}`stats` - Statistical functions