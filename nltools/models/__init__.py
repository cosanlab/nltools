"""
Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.
"""

from .base import BaseModel
from .ridge import Ridge

__all__ = ['BaseModel', 'Ridge']
