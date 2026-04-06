"""
nltools.plotting — Visualization utilities for neuroimaging analysis.

This package provides standalone plotting functions organized into
focused submodules:

- **brain**: Surface plots, flatmaps, and interactive brain viewers
- **adjacency**: Adjacency matrix visualizations (stacked, silhouette, distance)
- **prediction**: Model output plots (ROC, SVM margin, regression, logistic)
- **decomposition**: ICA/PCA component viewer

All public functions are re-exported here for convenience::

    from nltools.plotting import surface_plot, roc_plot, component_viewer  # all work
"""

from .brain import (
    plot_interactive_brain,
    surface_plot,
    plot_flatmap,
    _get_surface_paths as _get_surface_paths,
    _resolve_brain_input as _resolve_brain_input,
    _get_background_map as _get_background_map,
)
from .adjacency import (
    plot_stacked_adjacency,
    plot_mean_label_distance,
    plot_between_label_distance,
    plot_silhouette,
)
from .prediction import (
    dist_from_hyperplane_plot,
    scatterplot,
    probability_plot,
    roc_plot,
)
from .decomposition import component_viewer

__all__ = [
    # brain
    "plot_interactive_brain",
    "surface_plot",
    "plot_flatmap",
    # adjacency
    "plot_stacked_adjacency",
    "plot_mean_label_distance",
    "plot_between_label_distance",
    "plot_silhouette",
    # prediction
    "dist_from_hyperplane_plot",
    "scatterplot",
    "probability_plot",
    "roc_plot",
    # decomposition
    "component_viewer",
]
