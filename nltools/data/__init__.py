"""
nltools data types.
"""

from .brain_data import BrainData
from .adjacency import Adjacency
from .design_matrix import DesignMatrix
from .fit_results import Fit
from .collection import BrainCollection

__all__ = [
    "BrainData",
    "BrainCollection",
    "Adjacency",
    "DesignMatrix",
    "Fit",
]
