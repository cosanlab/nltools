"""Structural result records returned by the nltools estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

__all__ = ["ContrastResult"]

Payload = TypeVar("Payload")


@dataclass(frozen=True)
class ContrastResult(Generic[Payload]):
    """Frozen record of the inferential outputs of one contrast.

    The one result type inferential contrast methods return. Its payload is
    whatever the producer works in: `float` or `np.ndarray` for a `Glm`,
    `BrainData` for the `BrainData` facade.

    Fields cannot be rebound. Array payloads stay mutable, but each result owns
    its arrays: they never alias the input contrast, a model's retained state,
    or another result.

    Every statistic describes the directional hypothesis that `effect` is zero,
    so `p_value` is one-sided; negating the contrast tests the other direction.

    Attributes:
        effect (Payload): The estimated linear combination of coefficients.
        variance (Payload): The estimated variance of `effect`.
        standard_error (Payload): `np.sqrt` of `variance`, with no absolute
            value or clipping, so it may be non-finite.
        statistic (Payload): The signed t-statistic for the null hypothesis
            that `effect` is zero.
        z_score (Payload): The signed normal-score equivalent of the
            directional p-value.
        p_value (Payload): The one-sided upper-tail p-value.
        degrees_of_freedom (float | np.ndarray): The residual degrees of
            freedom used for inference.
    """

    effect: Payload
    variance: Payload
    standard_error: Payload
    statistic: Payload
    z_score: Payload
    p_value: Payload
    degrees_of_freedom: float | np.ndarray
