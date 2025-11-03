# `nltools.plotting`

Visualization tools for neuroimaging data.

## Overview

The `nltools.plotting` module provides comprehensive plotting functions for visualizing neuroimaging data, including volumetric brain plots, surface plots, statistical maps, and interactive visualizations.

## Key Functions

**Surface Plotting**
- `surface_plot()` - Plot volumetric data on cortical surfaces with intelligent hemisphere parsing

**Volume Plotting**
- `plot_brain()` - Comprehensive brain plotting with glass brain and multi-slice views
- `plot_t_brain()` - T-test visualization with multiple comparison correction
- `plot_interactive_brain()` - Interactive JavaScript-based brain viewer

**Analysis Visualization**
- `scatterplot()` - Prediction scatterplots
- `probability_plot()` - Classification probability plots
- `roc_plot()` - ROC curve visualization
- `dist_from_hyperplane_plot()` - SVM distance from hyperplane

**Adjacency Visualization**
- `plot_stacked_adjacency()` - Stacked adjacency matrix visualization
- `plot_mean_label_distance()` - Within/between label distance violin plots
- `plot_between_label_distance()` - Between-label distance heatmaps
- `plot_silhouette()` - Silhouette plots for clustering validation

## Quick Start

```python
from nltools.plotting import surface_plot, plot_brain
from nltools.data import BrainData

# Load data
brain = BrainData('statistical_map.nii.gz')

# Surface plot with default 2×2 montage
fig = surface_plot(brain)

# Volume plot with glass brain
plot_brain(brain, how='glass')

# Custom surface plot
fig = surface_plot(brain, hemi='left', view='lateral', cmap='hot')
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.plotting
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - BrainData class with plotting methods
- {doc}`stats` - Statistical functions for analysis
- {doc}`analysis` - ROC analysis tools

