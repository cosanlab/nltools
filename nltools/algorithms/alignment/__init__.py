"""Multi-subject functional alignment algorithms.

Algorithms for aligning functional data across subjects:

- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)
- **align**: whole-brain alignment of a group of subjects, by SRM or by
  Procrustes-based hyperalignment (Haxby et al. 2011)
- **procrustes** / **procrustes_distance**: pairwise Procrustes superposition
  and its permutation test
- **align_states**: match two sets of state weight maps
"""

from .srm import SRM, DetSRM
from .procrustes import align, align_states, procrustes, procrustes_distance

__all__ = [
    "SRM",
    "DetSRM",
    "align",
    "align_states",
    "procrustes",
    "procrustes_distance",
]
