"""Multi-subject functional alignment algorithms.

Algorithms for aligning functional data across subjects:

- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)
- **align**: whole-brain alignment of a group of subjects, by SRM or by
  Procrustes-based hyperalignment (Haxby et al. 2011)
- **procrustes** / **procrustes_distance**: pairwise Procrustes superposition
  and its permutation test
- **align_states**: match two sets of state weight maps
"""

# Internal package: these imports are re-exports for the rest of nltools, not
# an advertised surface, so there is no `__all__` to mark them as used.
from .srm import _SRM, _DetSRM  # noqa: F401
from .procrustes import (  # noqa: F401
    align,
    align_states,
    procrustes,
    procrustes_distance,
)
