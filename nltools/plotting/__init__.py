"""nltools.plotting — visualization utilities for neuroimaging analysis.

`component_viewer` is the one function users call directly. Everything else
here is the drawing internal behind a data-class method — `BrainData.plot_surf`,
`Roc.plot`, `Adjacency.plot_silhouette` and friends — organized into focused
submodules:

- **brain**: surface plots, flatmaps, and the interactive brain viewer
- **adjacency**: adjacency matrix visualizations (stacked, silhouette, distance)
- **prediction**: model output plots (ROC, SVM margin, regression, logistic)
- **decomposition**: ICA/PCA component viewer
"""

from .brain import (  # noqa: F401
    plot_interactive_brain,
    plot_surf,
    plot_flatmap,
)
from .adjacency import (  # noqa: F401
    plot_stacked_adjacency,
    plot_mean_label_distance,
    plot_between_label_distance,
    plot_silhouette,
)
from .prediction import (  # noqa: F401
    plot_dist_from_hyperplane,
    plot_scatter,
    plot_probability,
    plot_roc,
)
from .decomposition import component_viewer

__all__ = ["component_viewer"]
