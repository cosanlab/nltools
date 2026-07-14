"""Pipeline infrastructure for nltools.

This module provides a fluent API for building data processing pipelines
with cross-validation support.

Classes:
Pipeline
    Base pipeline for chained transforms with optional CV.
CVScheme
    Cross-validation scheme configuration.
FittedStack
    Collection of fitted transforms for inverse transform support.

Protocols:
TransformStep
    Protocol for pipeline transform steps.
FittedTransform
    Protocol for fitted transform objects.
Terminal
    Protocol for terminal operations.

Examples:
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
from .cv import CVScheme as CVSchemeImpl, NestedCVScheme
from .results import CVResult, FoldResult, ISCResult, PermutationResult, RSAResult
from .steps import AlignStep, FittedAlign, NormalizeStep, PipeStep, ReduceStep
from .terminals import ISCTerminal, PredictTerminal, RSATerminal
from .multi_subject import MultiSubjectPipeline
from .pool import PooledData, ResultDict, StatResult

__all__ = [
    # Alignment (Phase 7)
    "AlignStep",
    # Results (Phase 3 + Phase 8)
    "CVResult",
    "CVScheme",
    "CVSchemeImpl",
    "FittedAlign",
    "FittedStack",
    "FittedTransform",
    "FoldResult",
    "ISCResult",
    "ISCTerminal",
    # Multi-subject (Phase 4)
    "MultiSubjectPipeline",
    "NestedCVScheme",
    # Steps (Phase 2)
    "NormalizeStep",
    "PermutationResult",
    "PipeStep",
    # Core classes
    "Pipeline",
    # Pool infrastructure (Phase 5)
    "PooledData",
    # Terminals (Phase 3 + Phase 8)
    "PredictTerminal",
    "RSAResult",
    "RSATerminal",
    "ReduceStep",
    "ResultDict",
    "StatResult",
    "Terminal",
    # Protocols
    "TransformStep",
]
