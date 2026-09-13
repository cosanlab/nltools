"""Shared helpers for the permutation tests: sign flips, z-from-p, the stability epsilon and progress bars."""

import numpy as np
from ...utils import _NullProgressBar, _make_progress_bar, _maybe_tqdm  # noqa: F401
from .random import _generate_sign_flips as _generate_sign_flips  # noqa: F401


# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Small constant added to denominators to prevent division by zero
# Value: 1e-10 is standard in scientific computing for float64 precision
# - Well above machine epsilon (2.22e-16 for float64)
# - Small enough not to affect correlation values
# - Matches established practice in neuroimaging libraries
EPSILON = 1e-10


def _signed_z_from_p(t_like_arr, p_arr, tail_internal: str = "two") -> np.ndarray:
    """Compute a signed z-score map from a p-value map.

    The z-from-p conversion used by `BrainData.ttest` and `Adjacency.ttest`.
    The clipping policy below must live in exactly one place.

    Two-tailed p: ``|z| = norm.isf(p/2)`` so that p=0.05 → |z|≈1.96, matching
    nilearn's ``output_type='z_score'`` convention, with the sign copied from
    the accompanying statistic. One-tailed (upper) p: ``z = norm.isf(p)`` —
    a one-sided p already encodes direction, so no sign copy is needed.

    p is clipped to the open interval (0, 1) so z stays finite in BOTH
    directions: the lower bound is the smallest positive normal float64
    (z ≈ +37.7), the upper bound the largest float64 below 1 (z ≈ -8.3).
    The bounds look asymmetric because float64 resolves p near 0 far more
    finely than near 1; both sit beyond any meaningful statistical
    resolution. Without the upper clip, ``p == 1.0`` — reachable on the
    one-tailed path, e.g. ``t.sf(-45, 29) == 1.0`` — maps to ``-inf`` and
    poisons downstream percentiles and plotting.

    Args:
        t_like_arr (np.ndarray): Statistic supplying the sign (t map or similar).
        p_arr (np.ndarray): P-value map (two-tailed, or one-tailed upper).
        tail_internal (str): `'two'` (default) or `'upper'`.

    Returns:
        np.ndarray: Signed z map, finite everywhere.
    """
    from scipy.stats import norm

    p_clipped = np.clip(
        np.asarray(p_arr, dtype=np.float64),
        np.finfo(np.float64).tiny,
        np.nextafter(1.0, 0.0),
    )
    if tail_internal == "upper":
        return norm.isf(p_clipped)
    z_abs = norm.isf(p_clipped / 2.0)
    return np.sign(np.asarray(t_like_arr)) * z_abs
