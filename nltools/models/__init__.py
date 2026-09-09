"""
Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.
"""

from .results import ContrastResult
from .ridge import Ridge
from .glm import Glm

__all__ = ["ContrastResult", "Glm", "Ridge"]
