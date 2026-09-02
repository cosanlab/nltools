"""Group-level reductions and cross-subject analyses for `BrainCollection`.

Voxelwise summaries (`mean`, `std`, `var`, `median`, `sum_`, `min_`,
`max_`, `concat`), group tests (`ttest`, `ttest2`, `anova`,
`permutation_test`, `permutation_test2`), inter-subject correlation (`isc`,
`isc_test`), and functional alignment (`align`). The `BrainCollection`
methods of the same names delegate here. Where the math allows (means,
variances, t-tests, leave-one-out ISC) inputs are streamed one subject at a
time, so peak memory stays near one subject's worth of data; `median`,
`concat`, the permutation tests, pairwise ISC, and `align` load every
subject. Every result is an in-memory `BrainData` (or a dict of them) and is
never cached to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np


if TYPE_CHECKING:
    from ..braindata import BrainData
    from . import BrainCollection


__all__ = [
    "align",
    "anova",
    "concat",
    "isc",
    "isc_test",
    "max_",
    "mean",
    "median",
    "min_",
    "permutation_test",
    "permutation_test2",
    "std",
    "sum_",
    "ttest",
    "ttest2",
    "var",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_arrays(bc: BrainCollection, roi=None):
    """Yield one item's ``data`` array at a time.

    Loads path-backed items on the fly via ``bc._load_item`` (which builds a
    fresh ``BrainData`` without mutating ``bc._items``). Peak RAM stays at
    ~1 subject's worth of data. When *roi* is given (a BrainData already in the
    collection's mask space, from `_resolve_roi`), each item is masked to the
    ROI before being yielded, so downstream math sees only ROI voxels.
    """
    for i in range(len(bc._items)):
        item = bc._load_item(i)
        if roi is not None:
            item = item.apply_mask(roi)
        yield np.asarray(item.data)


def _resolve_roi(bc: BrainCollection, roi_mask):
    """Coerce *roi_mask* into the collection's space once. Returns (roi, out_mask).

    The ROI must be built against ``bc._mask``: passing a raw Niimg straight to
    ``BrainData.apply_mask`` would re-home it onto the *default* MNI152 mask,
    which silently breaks any collection in another space. Resolving here also
    keeps the resample to one pass rather than one per subject.

    ``out_mask`` is the mask scoped results must carry, since ROI-masked arrays
    span the ROI's voxels rather than the collection's whole-brain mask.
    """
    if roi_mask is None:
        return None, bc._mask
    from ..braindata.utils import check_brain_data

    roi = check_brain_data(roi_mask, mask=bc._mask)
    return roi, roi.to_nifti()


def _make_braindata(arr: np.ndarray, mask: nib.Nifti1Image):
    """Wrap an ndarray result as a ``BrainData``. Promotes 1D → ``(1, n_vox)``."""
    from ..braindata import BrainData

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return BrainData(arr.astype(np.float32), mask=mask)


def _check_nonempty(bc: BrainCollection) -> None:
    if len(bc._items) == 0:
        raise ValueError("collection is empty")


def _welford(bc: BrainCollection) -> tuple[int, np.ndarray, np.ndarray]:
    """One-pass Welford accumulator across items. Returns ``(n, mean, M2)``.

    ``M2`` is the sum of squared deviations; ``var = M2 / (n-1)``.
    """
    n = 0
    mean = np.empty(0, dtype=np.float64)
    M2 = np.empty(0, dtype=np.float64)
    for x in _iter_arrays(bc):
        n += 1
        x64 = x.astype(np.float64)
        if n == 1:
            mean = x64.copy()
            M2 = np.zeros_like(mean)
        else:
            delta = x64 - mean
            mean += delta / n
            delta2 = x64 - mean
            M2 += delta * delta2
    return n, mean, M2


# ---------------------------------------------------------------------------
# Stream-friendly reductions (Welford one-pass)
# ---------------------------------------------------------------------------


def concat(bc: BrainCollection) -> BrainData:
    """Stack every subject's rows into one `BrainData` of shape ``(n_total_obs, n_voxels)``.

    Loads every item (the operation *is* materialization). Single-image items
    are promoted to ``(1, n_voxels)`` before concatenation.
    """
    _check_nonempty(bc)
    arrays = []
    for x in _iter_arrays(bc):
        arrays.append(x.reshape(1, -1) if x.ndim == 1 else x)
    return _make_braindata(np.concatenate(arrays, axis=0), bc._mask)


def mean(bc: BrainCollection) -> BrainData:
    """Voxelwise mean across subjects.

    Streams one subject at a time from path-backed input.
    """
    _check_nonempty(bc)
    n, m, _ = _welford(bc)
    return _make_braindata(m, bc._mask)


def std(bc: BrainCollection) -> BrainData:
    """Voxelwise standard deviation across subjects (``ddof=1``).

    Streams one subject at a time via Welford's algorithm.
    """
    _check_nonempty(bc)
    n, _, M2 = _welford(bc)
    var_arr = M2 / max(n - 1, 1)
    return _make_braindata(np.sqrt(var_arr), bc._mask)


def var(bc: BrainCollection) -> BrainData:
    """Voxelwise variance across subjects (``ddof=1``).

    Streams one subject at a time via Welford's algorithm.
    """
    _check_nonempty(bc)
    n, _, M2 = _welford(bc)
    return _make_braindata(M2 / max(n - 1, 1), bc._mask)


def median(bc: BrainCollection) -> BrainData:
    """Voxelwise median across subjects.

    Loads every item into memory (a median cannot be streamed).
    """
    _check_nonempty(bc)
    stack = np.stack(list(_iter_arrays(bc)), axis=0)
    return _make_braindata(np.median(stack, axis=0), bc._mask)


def sum_(bc: BrainCollection) -> BrainData:
    """Voxelwise sum across subjects.

    Streams one subject at a time from path-backed input.
    """
    _check_nonempty(bc)
    total = None
    for x in _iter_arrays(bc):
        x64 = x.astype(np.float64)
        total = x64 if total is None else total + x64
    return _make_braindata(total, bc._mask)


def min_(bc: BrainCollection) -> BrainData:
    """Voxelwise minimum across subjects.

    Streams one subject at a time from path-backed input.
    """
    _check_nonempty(bc)
    cur = None
    for x in _iter_arrays(bc):
        cur = x.copy() if cur is None else np.minimum(cur, x)
    return _make_braindata(cur, bc._mask)


def max_(bc: BrainCollection) -> BrainData:
    """Voxelwise maximum across subjects.

    Streams one subject at a time from path-backed input.
    """
    _check_nonempty(bc)
    cur = None
    for x in _iter_arrays(bc):
        cur = x.copy() if cur is None else np.maximum(cur, x)
    return _make_braindata(cur, bc._mask)


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------


def ttest(
    bc: BrainCollection,
    *,
    popmean: float = 0.0,
    tail: int | str = 2,
) -> dict[str, BrainData]:
    """One-sample t-test across subjects.

    Streams one subject at a time via Welford's algorithm. The z map is
    derived from the reported p, so it matches the requested tail.

    Args:
        bc (BrainCollection): One map per subject.
        popmean (float): Null-hypothesis population mean.
        tail (int | str): ``2``/``'two'`` for two-tailed, or ``1``/``'one'``
            for one-tailed (mean > ``popmean``; negate the data for the other
            direction).

    Returns:
        dict[str, BrainData]: ``{'mean', 't', 'z', 'p'}`` maps — the same
            contract as `BrainData.ttest`.
    """
    from scipy.stats import t as t_dist

    from nltools.algorithms.inference.utils import _signed_z_from_p
    from nltools.algorithms.inference.validation import validate_tail_parameter

    tail_internal = validate_tail_parameter(tail)
    _check_nonempty(bc)
    n, m, M2 = _welford(bc)
    if n < 2:
        raise ValueError("ttest requires at least 2 subjects")

    var_arr = M2 / (n - 1)
    se = np.sqrt(var_arr / n)
    t_stat = np.divide(
        m - popmean,
        se,
        out=np.zeros_like(m),
        where=se > 0,
    )
    df = n - 1
    if tail_internal == "upper":
        p = t_dist.sf(t_stat, df)
    else:
        p = 2.0 * t_dist.sf(np.abs(t_stat), df)
    z = _signed_z_from_p(t_stat, p, tail_internal)

    return {
        "mean": _make_braindata(m, bc._mask),
        "t": _make_braindata(t_stat, bc._mask),
        "z": _make_braindata(z, bc._mask),
        "p": _make_braindata(p, bc._mask),
    }


def ttest2(
    bc: BrainCollection,
    other: BrainCollection,
    *,
    equal_var: bool = True,
    tail: int | str = 2,
) -> dict[str, BrainData]:
    """Two-sample t-test between two collections (subject-level).

    Args:
        bc (BrainCollection): First group, one map per subject.
        other (BrainCollection): Second group.
        equal_var (bool): If True, pooled-variance t-test; if False, Welch's.
        tail (int | str): ``2``/``'two'`` for two-tailed, or ``1``/``'one'``
            for one-tailed (``bc`` > ``other``; swap the operands for the other
            direction).

    Returns:
        dict[str, BrainData]: ``{'mean', 't', 'z', 'p'}`` maps, where
            ``'mean'`` is the group difference.
    """
    from scipy.stats import t as t_dist

    from nltools.algorithms.inference.utils import _signed_z_from_p
    from nltools.algorithms.inference.validation import validate_tail_parameter

    tail_internal = validate_tail_parameter(tail)
    _check_nonempty(bc)
    _check_nonempty(other)
    n1, m1, M2_1 = _welford(bc)
    n2, m2, M2_2 = _welford(other)
    var1 = M2_1 / max(n1 - 1, 1)
    var2 = M2_2 / max(n2 - 1, 1)
    diff = m1 - m2

    if equal_var:
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        se = np.sqrt(var1 / n1 + var2 / n2)
        # Welch–Satterthwaite df
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (var1 / n1 + var2 / n2) ** 2
            den = (var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1)
            df = np.where(den > 0, num / den, 1.0)

    t_stat = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
    if tail_internal == "upper":
        p = t_dist.sf(t_stat, df)
    else:
        p = 2.0 * t_dist.sf(np.abs(t_stat), df)
    z = _signed_z_from_p(t_stat, p, tail_internal)

    return {
        "mean": _make_braindata(diff, bc._mask),
        "t": _make_braindata(t_stat, bc._mask),
        "z": _make_braindata(z, bc._mask),
        "p": _make_braindata(p, bc._mask),
    }


def anova(
    bc: BrainCollection,
    groups: str | list | np.ndarray,
) -> dict[str, BrainData | int]:
    """One-way ANOVA across subjects.

    Args:
        bc (BrainCollection): One map per subject.
        groups (str | list | np.ndarray): A metadata column name, or a list /
            array of length ``n_subjects`` giving each subject's group label.

    Returns:
        dict[str, BrainData | int]: ``{'F', 'p'}`` maps plus the
            ``'df_between'`` and ``'df_within'`` degrees of freedom.
    """
    from scipy.stats import f as f_dist

    _check_nonempty(bc)

    if isinstance(groups, str):
        if bc._metadata is None or groups not in bc._metadata.columns:
            raise ValueError(f"groups column {groups!r} not in metadata")
        labels = np.asarray(bc._metadata[groups].to_list())
    else:
        labels = np.asarray(groups)
    if len(labels) != len(bc):
        raise ValueError(f"groups length ({len(labels)}) != n_subjects ({len(bc)})")

    # Materialize stack — ANOVA needs all data; tiny test data only.
    data = np.stack(list(_iter_arrays(bc)), axis=0)  # (n_subj, ..., n_vox)
    grand_mean = data.mean(axis=0)

    unique = np.unique(labels)
    n_groups = len(unique)
    n = len(labels)
    if n_groups < 2:
        raise ValueError("anova requires at least 2 groups")

    ss_between = np.zeros_like(grand_mean, dtype=np.float64)
    ss_within = np.zeros_like(grand_mean, dtype=np.float64)
    for g in unique:
        mask = labels == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        group_data = data[mask].astype(np.float64)
        group_mean = group_data.mean(axis=0)
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += ((group_data - group_mean) ** 2).sum(axis=0)

    df_between = n_groups - 1
    df_within = n - n_groups
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    f_stat = np.divide(
        ms_between,
        ms_within,
        out=np.zeros_like(ms_between),
        where=ms_within > 0,
    )
    p = f_dist.sf(f_stat, df_between, max(df_within, 1))

    return {
        "F": _make_braindata(f_stat, bc._mask),
        "p": _make_braindata(p, bc._mask),
        "df_between": df_between,
        "df_within": df_within,
    }


def permutation_test(
    bc: BrainCollection,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """One-sample sign-flipping permutation test across subjects.

    Loads every subject (sign-flipping needs the full stack) and delegates to
    `one_sample_permutation_test`, so ``device`` and ``n_jobs`` select the
    execution backend.

    Args:
        bc (BrainCollection): One map per subject.
        n_permute (int): Number of sign-flip permutations.
        tail (int | str): ``1`` for one-tailed, ``2`` for two-tailed.
        device (str): ``'cpu'`` (joblib parallel) or ``'gpu'`` (PyTorch).
        return_null (bool): If True, include the null distribution.
        n_jobs (int): CPU workers when ``device='cpu'`` (``-1`` = all cores).
        random_state (int | None): Seed for the sign-flip RNG.
        progress_bar (bool): If True, show a progress bar.

    Returns:
        dict: ``{'mean', 'p'}`` `BrainData` maps, plus ``'null_dist'`` when
            ``return_null=True``.
    """
    from nltools.algorithms import one_sample_permutation_test

    _check_nonempty(bc)

    data = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    n = data.shape[0]
    if n < 2:
        raise ValueError("permutation_test requires at least 2 subjects")

    item_shape = data.shape[1:]
    result = one_sample_permutation_test(
        data.reshape(n, -1),
        n_permute=n_permute,
        tail=tail,
        device=device,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=True,
        progress_bar=progress_bar,
    )

    out: dict = {
        "mean": _make_braindata(
            np.asarray(result["mean"]).reshape(item_shape), bc._mask
        ),
        "p": _make_braindata(np.asarray(result["p"]).reshape(item_shape), bc._mask),
    }
    if return_null:
        out["null_dist"] = np.asarray(result["null_dist"]).reshape(
            n_permute, *item_shape
        )
    return out


def permutation_test2(
    bc: BrainCollection,
    other: BrainCollection,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Two-sample permutation test by random label shuffling of the pooled subjects.

    Loads every subject and delegates to `two_sample_permutation_test`, so
    ``device`` and ``n_jobs`` select the execution backend.

    Args:
        bc (BrainCollection): First group, one map per subject.
        other (BrainCollection): Second group.
        n_permute (int): Number of label-shuffle permutations.
        tail (int | str): ``1`` for one-tailed, ``2`` for two-tailed.
        device (str): ``'cpu'`` (joblib parallel) or ``'gpu'`` (PyTorch).
        return_null (bool): If True, include the null distribution.
        n_jobs (int): CPU workers when ``device='cpu'`` (``-1`` = all cores).
        random_state (int | None): Seed for the shuffling RNG.
        progress_bar (bool): If True, show a progress bar.

    Returns:
        dict: ``{'mean', 'p'}`` `BrainData` maps (``'mean'`` is the group
            difference), plus ``'null_dist'`` when ``return_null=True``.
    """
    from nltools.algorithms import two_sample_permutation_test

    _check_nonempty(bc)
    _check_nonempty(other)

    data1 = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    data2 = np.stack(list(_iter_arrays(other)), axis=0).astype(np.float64)
    if data1.shape[1:] != data2.shape[1:]:
        raise ValueError(
            "permutation_test2 requires matching item shapes, got "
            f"{data1.shape[1:]} and {data2.shape[1:]}"
        )

    item_shape = data1.shape[1:]
    result = two_sample_permutation_test(
        data1.reshape(data1.shape[0], -1),
        data2.reshape(data2.shape[0], -1),
        n_permute=n_permute,
        tail=tail,
        device=device,
        n_jobs=n_jobs,
        random_state=random_state,
        return_null=True,
        progress_bar=progress_bar,
    )

    out: dict = {
        "mean": _make_braindata(
            np.asarray(result["mean_diff"]).reshape(item_shape), bc._mask
        ),
        "p": _make_braindata(np.asarray(result["p"]).reshape(item_shape), bc._mask),
    }
    if return_null:
        out["null_dist"] = np.asarray(result["null_dist"]).reshape(
            n_permute, *item_shape
        )
    return out


# ---------------------------------------------------------------------------
# Cross-subject ops (inherently multi-subject)
# ---------------------------------------------------------------------------


def _pearson_per_voxel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per voxel between two ``(n_obs, n_vox)`` arrays."""
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    a_norm = np.sqrt((a_c**2).sum(axis=0))
    b_norm = np.sqrt((b_c**2).sum(axis=0))
    den = a_norm * b_norm
    out = np.zeros_like(a_c.sum(axis=0))
    np.divide((a_c * b_c).sum(axis=0), den, out=out, where=den > 0)
    return out


def _aggregate_corrs(corrs: np.ndarray, summary: str) -> np.ndarray:
    """Aggregate per-subject correlations across the leading axis."""
    if summary == "median":
        return np.median(corrs, axis=0)
    if summary == "mean":
        z = np.arctanh(np.clip(corrs, -0.999, 0.999))
        return np.tanh(z.mean(axis=0))
    raise ValueError(f"unknown summary {summary!r}; expected 'median' or 'mean'")


def isc(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    summary: str = "median",
) -> dict:
    """Inter-subject correlation (ISC) across the time dimension.

    ``method='loo'`` correlates each subject with the average of the others
    and streams in two passes (peak memory about two subjects);
    ``method='pairwise'`` correlates every subject pair and loads all
    subjects at once.

    Args:
        bc (BrainCollection): One timeseries per subject, aligned in time.
        method (str): ``'loo'`` or ``'pairwise'``.
        roi_mask (Nifti1Image | Path | str | None): Optional ROI restricting
            the computation; the returned maps then carry the ROI mask rather
            than the collection's whole-brain mask.
        summary (str): How to aggregate across subjects or pairs — ``'median'``
            or ``'mean'`` (Fisher-z averaged).

    Returns:
        dict: ``{'isc', 'per_subject'}`` for ``'loo'`` or ``{'isc', 'pairs'}``
            for ``'pairwise'``, where ``'isc'`` is a `BrainData` map and the
            other entry is the per-subject / per-pair correlation array.
    """
    _check_nonempty(bc)
    if method not in ("loo", "pairwise"):
        raise ValueError(f"method must be 'loo' or 'pairwise', got {method!r}")

    roi, out_mask = _resolve_roi(bc, roi_mask)

    if method == "loo":
        return _isc_loo_streaming(bc, roi, out_mask, summary)

    # pairwise materializes all subjects: it needs every C(n,2) pair, so there is
    # no single-pass streaming form (streaming would cost O(n²) subject reloads).
    # See docs/development/execution-model.md — the streaming rewrite covers loo.
    data = np.stack(list(_iter_arrays(bc, roi)), axis=0).astype(np.float64)
    n_subj = data.shape[0]
    if n_subj < 2:
        raise ValueError("isc requires at least 2 subjects")

    from itertools import combinations

    pairs = list(combinations(range(n_subj), 2))
    pair_corrs = np.empty((len(pairs), *data.shape[2:]), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        pair_corrs[k] = _pearson_per_voxel(data[i], data[j])
    agg = _aggregate_corrs(pair_corrs, summary)
    return {
        "isc": _make_braindata(agg, out_mask),
        "pairs": pair_corrs,
    }


def _isc_loo_streaming(bc, roi, out_mask, summary: str) -> dict:
    """Leave-one-out ISC in two streaming passes — never all subjects at once.

    loo ISC correlates each subject with the mean of the *others*. That template
    is ``(Σ subjects − subjectᵢ) / (n − 1)``, so the only cross-subject quantity
    needed is the running sum. Pass 1 accumulates that sum (one ``T×V`` array);
    pass 2 re-streams, forming each subject's template and per-voxel Pearson r.
    Peak memory is ~2 subjects regardless of N, matching ``mean()``/``std()`` —
    versus the old ``np.stack`` that held all N. Two full reads of path-backed
    input is the memory-for-I/O trade the streaming design intends.
    """
    # Pass 1: subject sum + count (a single subject-sized accumulator).
    total = None
    n_subj = 0
    for x in _iter_arrays(bc, roi):
        x64 = x.astype(np.float64)
        total = x64 if total is None else total + x64
        n_subj += 1
    if n_subj < 2:
        raise ValueError("isc requires at least 2 subjects")

    # Pass 2: leave-one-out template per subject, per-voxel correlation. `corrs`
    # is (n_subj, n_vox) — the small per-voxel result, not the T×V timeseries.
    corrs = np.empty((n_subj, *total.shape[1:]), dtype=np.float64)
    for i, x in enumerate(_iter_arrays(bc, roi)):
        x64 = x.astype(np.float64)
        template = (total - x64) / (n_subj - 1)
        corrs[i] = _pearson_per_voxel(x64, template)
    agg = _aggregate_corrs(corrs, summary)
    return {
        "isc": _make_braindata(agg, out_mask),
        "per_subject": corrs,
    }


def isc_test(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    n_samples: int = 5000,
    summary: str = "median",
    tail: int | str = 2,
    random_state: int | None = None,
) -> dict:
    """Bootstrap inference on ISC (per-voxel p-values).

    Resamples subjects with replacement, recomputes ISC on each draw, and
    derives a per-voxel p-value from the null distribution centered at 0.

    Args:
        bc (BrainCollection): One timeseries per subject, aligned in time.
        method (str): ``'loo'`` or ``'pairwise'`` (as in `isc`).
        roi_mask (Nifti1Image | Path | str | None): Optional ROI restricting
            the computation; the returned maps then carry the ROI mask.
        n_samples (int): Number of bootstrap resamples.
        summary (str): ``'median'`` or ``'mean'`` aggregation (as in `isc`).
        tail (int | str): ``2``/``'two'`` for two-tailed, or ``1``/``'one'``
            for one-tailed (ISC > 0).
        random_state (int | None): Seed for the bootstrap RNG.

    Returns:
        dict: ``{'isc', 'p', 'null_dist'}`` — ``'isc'`` and ``'p'`` are
            `BrainData` maps; ``'null_dist'`` is the bootstrap array.
    """
    from nltools.algorithms.inference.validation import validate_tail_parameter

    tail_internal = validate_tail_parameter(tail)
    rng = np.random.default_rng(random_state)
    observed = isc(bc, method=method, roi_mask=roi_mask, summary=summary)
    obs_map = np.asarray(observed["isc"].data).reshape(-1)
    n_subj = len(bc)

    roi, out_mask = _resolve_roi(bc, roi_mask)
    null = np.empty((n_samples, obs_map.size), dtype=np.float64)
    data = np.stack(list(_iter_arrays(bc, roi)), axis=0).astype(np.float64)
    for k in range(n_samples):
        idx = rng.integers(0, n_subj, size=n_subj)
        sample_data = data[idx]
        if method == "loo":
            total = sample_data.sum(axis=0)
            corrs = np.empty((n_subj, *data.shape[2:]), dtype=np.float64)
            denom = max(n_subj - 1, 1)
            for i in range(n_subj):
                template = (total - sample_data[i]) / denom
                corrs[i] = _pearson_per_voxel(sample_data[i], template)
            null[k] = _aggregate_corrs(corrs, summary).reshape(-1)
        else:
            from itertools import combinations

            pairs = list(combinations(range(n_subj), 2))
            pc = np.empty((len(pairs), *data.shape[2:]), dtype=np.float64)
            for kk, (i, j) in enumerate(pairs):
                pc[kk] = _pearson_per_voxel(sample_data[i], sample_data[j])
            null[k] = _aggregate_corrs(pc, summary).reshape(-1)

    # P-value centered at 0 (ISC null hypothesis: no synchrony → ISC = 0).
    # The subject bootstrap is centered on the OBSERVED ISC, so it must be
    # re-centered at 0 (subtract obs_map) before the comparison — otherwise the
    # null sits on top of the observed value and every voxel gets p ≈ 0.5. This
    # restores the pre-0.6.0 behavior (`_calc_pvalue(all_bootstraps - isc, isc)`)
    # that the collection refactor dropped. The Phipson-Smyth (count+1)/(n+1)
    # form lives in the shared engine helper, not inline.
    from nltools.algorithms.inference.utils import _compute_pvalue

    centered_null = null - obs_map
    p = _compute_pvalue(obs_map, centered_null, tail=tail_internal)
    return {
        "isc": observed["isc"],
        "p": _make_braindata(p.reshape(obs_map.shape), out_mask),
        "null_dist": null,
    }


def align(  # n_iter exemption: solver iterations, not a permutation count (see api-vocabulary.yml)
    bc: BrainCollection,
    *,
    method: str = "procrustes",
    spatial_scale: str = "searchlight",
    radius_mm: float = 10.0,
    roi_mask: nib.Nifti1Image | None = None,
    n_features: int | None = None,
    n_iter: int = 3,
    device: str = "cpu",
    return_model: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = False,
    cache: Literal["auto", True, False] = "auto",
):
    """Functionally align subjects into a common space via `LocalAlignment`.

    Loads every subject — the aligner needs all of them at once. Outputs are
    cached under the collection's cache root by the same ``cache`` rule as
    the per-subject methods.

    Args:
        bc (BrainCollection): The subjects to align.
        method (str): Alignment solver (e.g. ``'procrustes'``).
        spatial_scale (str): ``'searchlight'`` (overlapping spheres) or
            ``'roi'`` (non-overlapping parcels).
        radius_mm (float): Searchlight sphere radius in mm.
        roi_mask (Nifti1Image | None): Parcellation used when
            ``spatial_scale='roi'``.
        n_features (int | None): Optional feature count for the common space.
        n_iter (int): Solver iterations.
        device (str): ``'cpu'`` or ``'gpu'``.
        return_model (bool): If True, also return the fitted `LocalAlignment`.
        n_jobs (int): Parallel worker count (``-1`` uses all cores).
        progress_bar (bool): If True, show a progress bar.
        cache (Literal['auto', True, False]): Cache policy for the result
            (``'auto'`` follows source state).

    Returns:
        BrainCollection | tuple[BrainCollection, LocalAlignment]: The aligned
            collection, or ``(collection, model)`` when ``return_model=True``.
    """
    from ..braindata import BrainData
    from ...algorithms.alignment.local import LocalAlignment
    from . import execution

    _check_nonempty(bc)
    # Validate spatial_scale up front for parity with BrainData.align: each
    # facade honors the canonical 'searchlight'/'roi'/'whole_brain' vocabulary
    # and raises a clear NotImplementedError for the one scale it cannot serve
    # (collection alignment is local-only; whole-brain is a single global
    # transform, which lives on per-subject BrainData.align).
    if spatial_scale == "whole_brain":
        raise NotImplementedError(
            "BrainCollection.align(spatial_scale='whole_brain') is not "
            "implemented: collection alignment uses local schemes "
            "('searchlight' or 'roi') via LocalAlignment. For a single global "
            "transform, align individual BrainData objects with "
            "BrainData.align(spatial_scale='whole_brain')."
        )
    if spatial_scale not in ("searchlight", "roi"):
        raise ValueError(
            "spatial_scale must be one of {'searchlight', 'roi', 'whole_brain'}, "
            f"got {spatial_scale!r}"
        )
    # LocalAlignment operates on (n_voxels, n_samples); BrainData.data is
    # (n_samples, n_voxels), so transpose in and transpose the aligned result
    # back before rewrapping.
    arrays = [np.asarray(x).T for x in _iter_arrays(bc)]

    aligner = LocalAlignment(
        spatial_scale=spatial_scale,
        method=method,
        radius_mm=radius_mm,
        roi_mask=roi_mask,
        n_features=n_features,
        n_iter=n_iter,
        parallel=device,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
    )
    aligner.fit(arrays, mask=bc._mask)
    aligned = aligner.transform(arrays)

    new_brains = [BrainData(arr.T.astype(np.float32), mask=bc._mask) for arr in aligned]
    items, source_paths, step_dir = execution._persist_or_keep(
        bc,
        new_brains,
        op="align",
        op_kwargs={"method": method, "spatial_scale": spatial_scale},
        cache=cache,
    )
    new_bc = bc._clone(
        _items=items,
        _step_id=bc._next_step_id(),
        _step_dirs=bc._step_dirs + ([step_dir] if step_dir else []),
        _source_paths=source_paths,
    )
    if return_model:
        return new_bc, aligner
    return new_bc
