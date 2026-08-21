"""Multi-subject functional alignment algorithms.

This package provides algorithms for aligning functional data across subjects:

- **LocalAlignment**: Searchlight/ROI-scale alignment (Bazeille et al. 2021)
- **HyperAlignment**: Iterative Procrustes alignment (Haxby et al. 2011)
- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)
- **align** / **procrustes** / **procrustes_distance** / **align_states**:
  functional entry points for whole-brain alignment (SRM or Procrustes)
"""

from .local import LocalAlignment
from .hyperalignment import HyperAlignment
from .srm import SRM, DetSRM
from .procrustes import align, align_states, procrustes, procrustes_distance

__all__ = [
    "SRM",
    "DetSRM",
    "HyperAlignment",
    "LocalAlignment",
    "align",
    "align_states",
    "procrustes",
    "procrustes_distance",
]
