"""
nltools.plotting — Visualization utilities for neuroimaging analysis.

This package provides standalone plotting functions organized into
focused submodules:

- **brain**: Surface plots, flatmaps, and interactive brain viewers
- **adjacency**: Adjacency matrix visualizations (stacked, silhouette, distance)
- **prediction**: Model output plots (ROC, SVM margin, regression, logistic)
- **decomposition**: ICA/PCA component viewer

All public functions are re-exported here for convenience::

    from nltools.plotting import plot_surface, plot_roc, component_viewer  # all work
"""

from .brain import (
    plot_interactive_brain,
    plot_surface,
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
    # brain
    "plot_interactive_brain",
    "plot_surface",
    "plot_flatmap",
    # adjacency
    "plot_stacked_adjacency",
    "plot_mean_label_distance",
    "plot_between_label_distance",
    "plot_silhouette",
    # prediction
    "plot_dist_from_hyperplane",
    "plot_scatter",
    "plot_probability",
    "plot_roc",
    # decomposition
    "component_viewer",
]
