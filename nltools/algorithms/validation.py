"""The argument vocabulary every algorithms family shares.

`tail` is spoken by inference, alignment and regression alike, and by the two
facade methods that expose a permutation test (`Roc.summary`,
`Adjacency.ttest`), so its validator and the p-value computation built on it
live here rather than inside any one family.

Examples:
    ```python
    from nltools.algorithms.validation import _validate_tail_parameter

    _validate_tail_parameter(2)  # → 'two'
    _validate_tail_parameter("invalid")  # raises ValueError
    ```
"""

import numpy as np


def _validate_tail_parameter(tail: int | str) -> str:
    """Validate the public tail vocabulary and normalize to the internal form.

    The public vocabulary is deliberately two-valued: the *direction* of a
    one-tailed test is fixed by the test's convention, never chosen from the
    data (a data-driven direction would silently halve every p-value). A fixed
    direction across all tests is what keeps multiple-comparison correction
    (FDR, Bonferroni) valid (GH #315).

    Args:
        tail (int | str): `2` or `'two'` (the default everywhere) for a
            two-tailed test (`|obs|` vs `|null|`); `1` or `'one'` for a
            one-tailed test in the test's canonical positive direction
            (correlation/ISC/similarity > 0, mean > popmean, group1 > group2).
            To test the negative direction, negate your data, swap the groups,
            or flip the contrast.

    Returns:
        str: Normalized internal tail, `'two'` or `'upper'`.

    Raises:
        ValueError: If `tail` is not a valid option (including the removed
            `'upper'`/`'lower'`/`-1` forms).
    """
    # One mapping table lives in `_normalize_tail_internal`; the public layer
    # only rejects the internal-only directional forms it must not accept.
    if tail not in (-1, "upper", "lower"):
        try:
            return _normalize_tail_internal(tail)
        except ValueError:
            pass
    raise ValueError(
        f"tail must be 2|'two' (two-tailed) or 1|'one' (one-tailed, the test's "
        f"positive direction), got {tail!r}. The 'upper'/'lower'/-1 forms were "
        "removed in v0.6.0: to test the negative direction, negate your data, "
        "swap the groups, or flip the contrast."
    )


def _normalize_tail_internal(tail: int | str) -> str:
    """Normalize a tail value to the internal 'two'/'upper'/'lower' form.

    Accepts BOTH the public v0.6.0 vocabulary (2 or 'two', 1 or 'one') and the
    internal directional forms ('upper', 'lower', -1) that forced-tail call
    sites use directly. Public entry points must validate with the strict
    `_validate_tail_parameter` first — this permissive form exists only so
    `_compute_pvalue` can serve both layers.
    """
    if tail == 2 or tail == "two":
        return "two"
    if tail == 1 or tail == "one" or tail == "upper":
        return "upper"
    if tail == -1 or tail == "lower":
        return "lower"
    raise ValueError(
        f"tail must be 2|'two', 1|'one' (or internal 'upper'/'lower'/-1), got {tail!r}"
    )


def _compute_pvalue(
    obs_stat: np.ndarray,
    null_dist: np.ndarray,
    tail: int | str = 2,
) -> np.ndarray:
    """Calculate p-values from observed statistic and null distribution.

    Computes the proportion of null-distribution values at least as extreme as
    the observed statistic, with the `(count + 1) / (n_permute + 1)` correction:
    the observed value counts as one draw, so p is never 0 and its minimum is
    `1 / (n_permute + 1)`. Two-tailed tests compare absolute values. The fixed
    per-tail direction keeps multiple-comparison correction (FDR, Bonferroni)
    valid across tests (GH #315).

    Args:
        obs_stat (np.ndarray): Observed statistic(s), shape `()` for a scalar or
            `(n_features,)` for multi-feature.
        null_dist (np.ndarray): Null distribution from permutations, shape
            `(n_permute,)` for a single feature or `(n_permute, n_features)`
            for multi-feature.
        tail (int | str): `2` or `'two'` for a two-tailed test (`|obs|` vs
            `|null|`); `1`, `'one'`, or the internal `'upper'` for a one-tailed
            upper test (`null >= obs`, positive effects); the internal `'lower'`
            or `-1` for a one-tailed lower test (`null <= obs`), reserved for
            forced-tail call sites and not part of the public vocabulary.

    Returns:
        np.ndarray: P-value(s) with the same shape as `obs_stat`.

    Examples:
        ```python
        obs_stat = np.array([2.5])
        null_dist = np.random.randn(1000, 1)
        p = _compute_pvalue(obs_stat, null_dist, tail="two")
        0 < p <= 1  # → True

        # Explicit directions for forced-tail call sites
        p_upper = _compute_pvalue(obs_stat, null_dist, tail="upper")
        p_lower = _compute_pvalue(obs_stat, null_dist, tail="lower")
        ```
    """
    tail_normalized = _normalize_tail_internal(tail)

    # Ensure inputs are numpy arrays (handles Python float/int scalars)
    obs_stat = np.asarray(obs_stat)
    null_dist = np.asarray(null_dist)

    # Handle shape differences
    if null_dist.ndim == 1:
        null_dist = null_dist[:, np.newaxis]
    if obs_stat.ndim == 0:
        obs_stat = obs_stat.reshape(1)
    elif obs_stat.ndim == 1:
        obs_stat = obs_stat.reshape(1, -1)

    n_permute = null_dist.shape[0]
    denom = float(n_permute) + 1.0

    if tail_normalized == "upper":
        # One-tailed upper: count how many null >= observed
        # Tests for positive effects (H1: statistic > 0)
        numer = np.sum(null_dist >= obs_stat, axis=0) + 1.0
    elif tail_normalized == "lower":
        # One-tailed lower: count how many null <= observed
        # Tests for negative effects (H1: statistic < 0)
        numer = np.sum(null_dist <= obs_stat, axis=0) + 1.0
    else:  # tail_normalized == "two"
        # Two-tailed: count how many |null| >= |observed|
        numer = np.sum(np.abs(null_dist) >= np.abs(obs_stat), axis=0) + 1.0

    p_values = numer / denom

    return p_values
