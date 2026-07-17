"""nltools: a Python toolbox for analyzing neuroimaging data.

Focused on multivariate analyses and built on top of nilearn and scikit-learn,
nltools provides high-level data classes — `BrainData`, `Adjacency`,
`DesignMatrix`, and `BrainCollection` — that wrap common neuroimaging
workflows, alongside a functional core of statistics and algorithms (e.g.
ridge regression, SRM, hyperalignment, and inference).
"""

__all__ = [
    "SRM",
    "Adjacency",
    "BrainData",
    "BrainSpaceConfig",
    "DesignMatrix",
    "DetSRM",
    "Roc",
    "SimulateGrid",
    "Simulator",
    "__version__",
    "collapse_mask",
    "create_sphere",
    "cross_validation",
    "data",
    "datasets",
    "expand_mask",
    "get_brainspace",
    "io",
    "mask",
    "plotting",
    "reset_brainspace",
    "set_brainspace",
    "stats",
    "templates",
    "utils",
    "with_brainspace",
]

from .data import (
    BrainData,
    Adjacency,
    DesignMatrix,
    Simulator,
    SimulateGrid,
    Roc,
)
from .templates import (
    BrainSpaceConfig,
    get_brainspace,
    set_brainspace,
    reset_brainspace,
    with_brainspace,
)
from .version import __version__
from .mask import expand_mask, collapse_mask, create_sphere
from .algorithms import SRM, DetSRM

# Bind submodules advertised in __all__ so attribute access (e.g.
# nltools.datasets, nltools.cross_validation) works without a prior
# explicit `import nltools.datasets`.
from . import (  # noqa: F401
    cross_validation,
    data,
    datasets,
    io,
    plotting,
    stats,
    utils,
)
