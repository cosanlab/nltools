"""nltools: a Python toolbox for analyzing neuroimaging data.

Focused on multivariate analyses and built on top of nilearn and scikit-learn,
nltools provides high-level data classes — `BrainData`, `Adjacency`, and
`DesignMatrix` — that wrap common neuroimaging workflows, alongside a
functional core of statistics and algorithms (`nltools.algorithms`) that the
data classes delegate to.
"""

__all__ = [
    "Adjacency",
    "BrainData",
    "DesignMatrix",
    "Roc",
    "SimulateGrid",
    "Simulator",
    "__version__",
    "concatenate",
    "get_brainspace",
    "reset_brainspace",
    "set_brainspace",
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
from .data.combine import concatenate
from .templates import (
    get_brainspace,
    set_brainspace,
    reset_brainspace,
    with_brainspace,
)
from .version import __version__

# Bind the submodules users reach through attribute access (e.g.
# nltools.datasets, nltools.cross_validation) so no prior explicit
# `import nltools.datasets` is needed. They are not part of the advertised
# surface; the names in __all__ above are.
from . import (  # noqa: F401
    algorithms,
    cross_validation,
    data,
    datasets,
    io,
    mask,
    plotting,
    utils,
)
