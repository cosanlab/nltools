"""Data classes for neuroimaging analysis.

`BrainData` (masked voxel data), `Adjacency` (similarity/distance matrices),
`DesignMatrix` (regressors), the `Roc` analysis class, the
`Simulator`/`SimulateGrid` data simulators, and the frozen result records those
classes return (`Predict`, `BootstrapResult`, `ContrastResult`) plus the
brain-space configuration record (`BrainSpaceConfig`).
"""

from .braindata import BrainData
from .adjacency import Adjacency
from .designmatrix import DesignMatrix
from .results import BootstrapResult, Predict
from .simulator import Simulator, SimulateGrid
from .roc import Roc
from nltools.models.results import ContrastResult
from nltools.templates.config import BrainSpaceConfig

__all__ = [
    "Adjacency",
    "BootstrapResult",
    "BrainData",
    "BrainSpaceConfig",
    "ContrastResult",
    "DesignMatrix",
    "Predict",
    "Roc",
    "SimulateGrid",
    "Simulator",
]
