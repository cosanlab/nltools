"""nltools.plotting — Visualization utilities for neuroimaging analysis.

This package provides standalone plotting functions organized into
focused submodules:

- **brain**: Surface plots, flatmaps, and interactive brain viewers
- **adjacency**: Adjacency matrix visualizations (stacked, silhouette, distance)
- **prediction**: Model output plots (ROC, SVM margin, regression, logistic)
- **decomposition**: ICA/PCA component viewer

All public functions are re-exported here for convenience:

```python
from nltools.plotting import plot_surf, plot_roc, component_viewer  # all work
```
"""

from .brain import (
    plot_interactive_brain,
    plot_surf,
    plot_flatmap,
)
from .adjacency import (
    plot_stacked_adjacency,
    plot_mean_label_distance,
    plot_between_label_distance,
    plot_silhouette,
)
from .prediction import (
    plot_dist_from_hyperplane,
    plot_scatter,
    plot_probability,
    plot_roc,
)
from .decomposition import component_viewer

__all__ = [
    # decomposition
    "component_viewer",
    "plot_between_label_distance",
    # prediction
    "plot_dist_from_hyperplane",
    "plot_flatmap",
    # brain
    "plot_interactive_brain",
    "plot_mean_label_distance",
    "plot_probability",
    "plot_roc",
    "plot_scatter",
    "plot_silhouette",
    # adjacency
    "plot_stacked_adjacency",
    "plot_surf",
]
