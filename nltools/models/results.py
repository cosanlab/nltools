"""Structural result records returned by the nltools estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np


Payload = TypeVar("Payload")


@dataclass(frozen=True)
class ContrastResult(Generic[Payload]):
    """Frozen record of the inferential outputs of one contrast.

    The one result type inferential contrast methods return. Its payload is
    whatever the producer works in: `float` or `np.ndarray` for a `_Glm`,
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

    def write(self, directory, prefix=None) -> list:
        """Write the contrast to `directory` as NIfTI maps and a sidecar.

        The whole "fit, contrast, then save" workflow in one call. Each map
        becomes `<prefix>_effect.nii.gz`, `_variance`, `_se`, `_t`, `_z` and
        `_p`, and `<prefix>_contrast.json` records the degrees of freedom.
        Nothing here is BIDS.

        Args:
            directory (str | Path): Where to write. Created if it does not exist.
            prefix (str | None): Prepended to every filename as `<prefix>_`.
                Default None writes the bare names.

        Returns:
            list[Path]: Every file written.

        Raises:
            TypeError: If the payloads are bare arrays rather than brain maps,
                which only a contrast computed outside `BrainData` can be.

        Examples:
            ```python
            result = data.compute_contrasts("a - b", inference=True)
            result.write("derivatives/contrasts", prefix="a-gt-b")
            ```
        """
        from nltools.data.results_io import _write_contrast

        return _write_contrast(self, directory, prefix)
