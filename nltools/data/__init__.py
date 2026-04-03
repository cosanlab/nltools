"""
nltools data types.
"""

from .braindata import BrainData
from .adjacency import Adjacency
from .designmatrix import DesignMatrix
from .fitresults import Fit
from .collection import BrainCollection

__all__ = [
    "BrainData",
    "BrainCollection",
    "Adjacency",
    "DesignMatrix",
    "Fit",
]
