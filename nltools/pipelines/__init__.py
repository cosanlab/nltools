"""Pipeline infrastructure for nltools.

This module provides a fluent API for building data processing pipelines
with cross-validation support.

Classes
-------
Pipeline
    Base pipeline for chained transforms with optional CV.
CVScheme
    Cross-validation scheme configuration.
FittedStack
    Collection of fitted transforms for inverse transform support.

Protocols
---------
TransformStep
    Protocol for pipeline transform steps.
FittedTransform
    Protocol for fitted transform objects.
Terminal
    Protocol for terminal operations.

Examples
--------
>>> from nltools.pipelines import Pipeline, CVScheme
>>> cv = CVScheme(scheme='kfold', k=5)
>>> result = (
...     Pipeline(data, cv=cv)
...     .normalize()
...     .reduce(n_components=50)
...     .predict(y)
... )
"""

from .base import (
    CVScheme,
    FittedStack,
    FittedTransform,
    Pipeline,
    Terminal,
    TransformStep,
)
from .cv import CVScheme as CVSchemeImpl
from .results import CVResult, FoldResult
from .steps import AlignStep, FittedAlign, NormalizeStep, PipeStep, ReduceStep
from .terminals import PredictTerminal
from .multi_subject import MultiSubjectPipeline
from .pool import PooledData, ResultDict, StatResult

__all__ = [
    # Core classes
    "Pipeline",
    "CVScheme",
    "CVSchemeImpl",
    "FittedStack",
    # Protocols
    "TransformStep",
    "FittedTransform",
    "Terminal",
    # Steps (Phase 2)
    "NormalizeStep",
    "ReduceStep",
    "PipeStep",
    # Alignment (Phase 7)
    "AlignStep",
    "FittedAlign",
    # Terminals (Phase 3)
    "PredictTerminal",
    # Results (Phase 3)
    "CVResult",
    "FoldResult",
    # Multi-subject (Phase 4)
    "MultiSubjectPipeline",
    # Pool infrastructure (Phase 5)
    "PooledData",
    "StatResult",
    "ResultDict",
]
