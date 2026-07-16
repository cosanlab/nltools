"""Low-level pipeline primitives used by `BrainCollection`.

These are the building blocks that back `BrainCollectionPipeline`: transform
steps (`NormalizeStep`, `ReduceStep`, `PipeStep`, `AlignStep`), the fitted-stack
container (`FittedStack`), the pooled-data aggregator (`PooledData`), and the
transform protocols. The standalone fluent `Pipeline` / `MultiSubjectPipeline`
orchestration was removed in v0.6.0 — multi-subject CV now lives on
`BrainCollection` (`.cv().standardize().reduce().predict()`) and custom
single-dataset preprocessing uses `model=make_pipeline(...)` on `BrainData.predict`.
"""

from .base import (
    CVScheme,
    FittedStack,
    FittedTransform,
    Terminal,
    TransformStep,
)
from .pool import PooledData, ResultDict, StatResult
from .steps import AlignStep, FittedAlign, NormalizeStep, PipeStep, ReduceStep

__all__ = [
    "AlignStep",
    # Transform protocols
    "CVScheme",
    "FittedAlign",
    "FittedStack",
    "FittedTransform",
    # Transform steps
    "NormalizeStep",
    "PipeStep",
    # Pool infrastructure
    "PooledData",
    "ReduceStep",
    "ResultDict",
    "StatResult",
    "Terminal",
    "TransformStep",
]
