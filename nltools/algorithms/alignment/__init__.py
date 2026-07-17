"""Multi-subject functional alignment algorithms.

This package provides algorithms for aligning functional data across subjects:

- **LocalAlignment**: Searchlight/ROI-scale alignment (Bazeille et al. 2021)
- **HyperAlignment**: Iterative Procrustes alignment (Haxby et al. 2011)
- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)
"""

from .local import LocalAlignment
from .hyperalignment import HyperAlignment
from .srm import SRM, DetSRM

__all__ = ["SRM", "DetSRM", "HyperAlignment", "LocalAlignment"]
