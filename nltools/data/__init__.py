"""Data classes for neuroimaging analysis.

`BrainData` (masked voxel data), `Adjacency` (similarity/distance matrices),
`DesignMatrix` (regressors), `BrainCollection` (per-subject stacks of
`BrainData`), `Fit` (model-fit results), plus the `Roc` analysis class and the
`Simulator`/`SimulateGrid` data simulators.
"""

from .braindata import BrainData
from .adjacency import Adjacency
from .designmatrix import DesignMatrix
from .fitresults import Fit
from .collection import BrainCollection
from .simulator import Simulator, SimulateGrid
from .roc import Roc

__all__ = [
    "Adjacency",
    "BrainCollection",
    "BrainData",
    "DesignMatrix",
    "Fit",
    "Roc",
    "SimulateGrid",
    "Simulator",
]
