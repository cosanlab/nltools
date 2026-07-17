"""Low-level pipeline primitives used by `BrainCollection`.

These are the building blocks that back `BrainCollectionPipeline`: transform
steps (`NormalizeStep`, `ReduceStep`, `PipeStep`), the fitted-stack container
(`FittedStack`), the cross-validation scheme (`CVScheme`), and the transform
protocols. This package is internal; the standalone fluent `Pipeline` /
`MultiSubjectPipeline` orchestration was removed in v0.6.0 — multi-subject CV
now lives on `BrainCollection` (`.cv().standardize().reduce().predict()`) and
custom single-dataset preprocessing uses `model=make_pipeline(...)` on
`BrainData.predict`.
"""

from .base import FittedStack, FittedTransform, TransformStep
from .cv import CVScheme
from .steps import NormalizeStep, PipeStep, ReduceStep

__all__ = [
    # Cross-validation scheme
    "CVScheme",
    # Fitted-stack container + transform protocols
    "FittedStack",
    "FittedTransform",
    # Transform steps
    "NormalizeStep",
    "PipeStep",
    "ReduceStep",
    "TransformStep",
]
