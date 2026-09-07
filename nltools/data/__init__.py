"""Data classes for neuroimaging analysis.

`BrainData` (masked voxel data), `Adjacency` (similarity/distance matrices),
`DesignMatrix` (regressors), decoding result records, the `Roc` analysis class, and the
`Simulator`/`SimulateGrid` data simulators.
"""

from .braindata import BrainData
from .adjacency import Adjacency
from .designmatrix import DesignMatrix
from .results import Predict
from .simulator import Simulator, SimulateGrid
from .roc import Roc

__all__ = [
    "Adjacency",
    "BrainData",
    "DesignMatrix",
    "Predict",
    "Roc",
    "SimulateGrid",
    "Simulator",
]
