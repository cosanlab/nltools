"""
Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.
"""

# Internal package: these imports are re-exports for the rest of nltools, not
# an advertised surface, so there is no `__all__` to mark them as used.
from .results import ContrastResult  # noqa: F401
from .ridge import Ridge  # noqa: F401
from .glm import Glm  # noqa: F401
