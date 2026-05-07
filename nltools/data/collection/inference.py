"""Group-level reductions and cross-subject ops for BrainCollection.

Module-level functions that the ``BrainCollection`` facade delegates to.
Reductions stream from path-backed inputs (Welford-style) and produce
in-memory ``BrainData`` (or dicts of them); they never path-back their
own output.
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


def _iter_arrays(bc: BrainCollection):
    """Yield one item's ``data`` array at a time.

    Loads path-backed items on the fly via ``bc._load_item`` (which builds a
    fresh ``BrainData`` without mutating ``bc._items``). Peak RAM stays at
    ~1 subject's worth of data.
    """
    for i in range(len(bc._items)):
        yield np.asarray(bc._load_item(i).data)


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
    mean = M2 = None
    for x in _iter_arrays(bc):
        n += 1
        x64 = x.astype(np.float64)
        if mean is None:
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
    """Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.

    Not streamable — the operation *is* materialization. 1D items are
    promoted to ``(1, n_voxels)`` before concatenation.
    """
    _check_nonempty(bc)
    arrays = []
    for x in _iter_arrays(bc):
        arrays.append(x.reshape(1, -1) if x.ndim == 1 else x)
    return _make_braindata(np.concatenate(arrays, axis=0), bc._mask)


def mean(bc: BrainCollection) -> BrainData:
    """Mean across subjects (leading axis). Streams from path-backed input."""
    _check_nonempty(bc)
    n, m, _ = _welford(bc)
    return _make_braindata(m, bc._mask)


def std(bc: BrainCollection) -> BrainData:
    """Std across subjects. Streams via Welford; ddof=1."""
    _check_nonempty(bc)
    n, _, M2 = _welford(bc)
    var_arr = M2 / max(n - 1, 1)
    return _make_braindata(np.sqrt(var_arr), bc._mask)


def var(bc: BrainCollection) -> BrainData:
    """Variance across subjects. Streams via Welford; ddof=1."""
    _check_nonempty(bc)
    n, _, M2 = _welford(bc)
    return _make_braindata(M2 / max(n - 1, 1), bc._mask)


def median(bc: BrainCollection) -> BrainData:
    """Median across subjects. Materializes (not streaming-friendly)."""
    _check_nonempty(bc)
    stack = np.stack(list(_iter_arrays(bc)), axis=0)
    return _make_braindata(np.median(stack, axis=0), bc._mask)


def sum_(bc: BrainCollection) -> BrainData:
    """Sum across subjects. Streams."""
    _check_nonempty(bc)
    total = None
    for x in _iter_arrays(bc):
        x64 = x.astype(np.float64)
        total = x64 if total is None else total + x64
    return _make_braindata(total, bc._mask)


def min_(bc: BrainCollection) -> BrainData:
    """Per-voxel min across subjects. Streams."""
    _check_nonempty(bc)
    cur = None
    for x in _iter_arrays(bc):
        cur = x.copy() if cur is None else np.minimum(cur, x)
    return _make_braindata(cur, bc._mask)


def max_(bc: BrainCollection) -> BrainData:
    """Per-voxel max across subjects. Streams."""
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
) -> dict[str, BrainData]:
    """One-sample t-test across subjects.

    Returns ``{'mean', 't', 'z', 'p'}`` — same shape contract as
    ``BrainData.ttest``. Streams from path-backed input via Welford.
    """
    from scipy.stats import norm, t as t_dist

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
    p = 2.0 * t_dist.sf(np.abs(t_stat), df)
    z = np.sign(t_stat) * norm.isf(np.clip(p / 2.0, 1e-300, 1.0))

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
) -> dict[str, BrainData]:
    """Two-sample t-test between two collections (subject-level)."""
    from scipy.stats import norm, t as t_dist

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
    p = 2.0 * t_dist.sf(np.abs(t_stat), df)
    z = np.sign(t_stat) * norm.isf(np.clip(p / 2.0, 1e-300, 1.0))

    return {
        "mean": _make_braindata(diff, bc._mask),
        "t": _make_braindata(t_stat, bc._mask),
        "z": _make_braindata(z, bc._mask),
        "p": _make_braindata(p, bc._mask),
    }


def anova(
    bc: BrainCollection,
    groups: str | list | np.ndarray,
) -> dict[str, BrainData]:
    """One-way ANOVA across subjects.

    ``groups`` is a metadata column name, a list, or an ndarray of length
    ``n_subjects``. Returns ``{'F', 'p', 'df_between', 'df_within'}``.
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
    tail: int = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """Sign-flipping permutation test across subjects (one-sample).

    Per SPEC streaming-algorithms table, sign-flipping needs all subjects
    in memory by design. ``device`` is currently informational; backend
    selection is deferred to the parametric stats path.
    """
    _check_nonempty(bc)
    if tail not in (1, 2):
        raise ValueError(f"tail must be 1 or 2, got {tail}")

    rng = np.random.default_rng(random_state)
    data = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    n = data.shape[0]
    if n < 2:
        raise ValueError("permutation_test requires at least 2 subjects")

    observed_mean = data.mean(axis=0)

    null = np.empty((n_permute, *observed_mean.shape), dtype=np.float64)
    for k in range(n_permute):
        signs = rng.choice([-1.0, 1.0], size=n).reshape((n,) + (1,) * (data.ndim - 1))
        null[k] = (signs * data).mean(axis=0)

    if tail == 2:
        # Empirical p with +1 numerator/denominator (for unbiased estimation)
        p = (np.sum(np.abs(null) >= np.abs(observed_mean), axis=0) + 1) / (
            n_permute + 1
        )
    else:
        p = (np.sum(null >= observed_mean, axis=0) + 1) / (n_permute + 1)

    out: dict = {
        "mean": _make_braindata(observed_mean, bc._mask),
        "p": _make_braindata(p, bc._mask),
    }
    if return_null:
        out["null_distribution"] = null
    return out


def permutation_test2(
    bc: BrainCollection,
    other: BrainCollection,
    *,
    n_permute: int = 5000,
    tail: int = 2,
    device: str = "cpu",
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> dict:
    """Two-sample permutation test by random label shuffling."""
    _check_nonempty(bc)
    _check_nonempty(other)
    if tail not in (1, 2):
        raise ValueError(f"tail must be 1 or 2, got {tail}")

    rng = np.random.default_rng(random_state)
    data1 = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    data2 = np.stack(list(_iter_arrays(other)), axis=0).astype(np.float64)
    n1, n2 = data1.shape[0], data2.shape[0]
    pooled = np.concatenate([data1, data2], axis=0)
    n_total = n1 + n2

    observed_diff = data1.mean(axis=0) - data2.mean(axis=0)

    null = np.empty((n_permute, *observed_diff.shape), dtype=np.float64)
    for k in range(n_permute):
        idx = rng.permutation(n_total)
        s1 = pooled[idx[:n1]].mean(axis=0)
        s2 = pooled[idx[n1:]].mean(axis=0)
        null[k] = s1 - s2

    if tail == 2:
        p = (np.sum(np.abs(null) >= np.abs(observed_diff), axis=0) + 1) / (
            n_permute + 1
        )
    else:
        p = (np.sum(null >= observed_diff, axis=0) + 1) / (n_permute + 1)

    out: dict = {
        "mean": _make_braindata(observed_diff, bc._mask),
        "p": _make_braindata(p, bc._mask),
    }
    if return_null:
        out["null_distribution"] = null
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


def _aggregate_corrs(corrs: np.ndarray, metric: str) -> np.ndarray:
    """Aggregate per-subject correlations across the leading axis."""
    if metric == "median":
        return np.median(corrs, axis=0)
    if metric == "mean":
        z = np.arctanh(np.clip(corrs, -0.999, 0.999))
        return np.tanh(z.mean(axis=0))
    raise ValueError(f"unknown metric {metric!r}; expected 'median' or 'mean'")


def isc(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    radius_mm: float | None = 6.0,
    metric: str = "median",
    device: str = "cpu",
    n_jobs: int = -1,
    progress_bar: bool = False,
) -> dict:
    """Inter-subject correlation across the time dimension.

    method='loo' uses the leave-one-out template approach (each subject
    correlated with the average of the others). method='pairwise' computes
    all subject pairs. Both materialize all subjects in v0.6.0; the
    streaming rewrite is deferred to a later release.

    Returns ``{'isc', 'per_subject'}`` for ``loo`` or ``{'isc', 'pairs'}``
    for ``pairwise``.
    """
    _check_nonempty(bc)
    if method not in ("loo", "pairwise"):
        raise ValueError(f"method must be 'loo' or 'pairwise', got {method!r}")

    data = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    n_subj = data.shape[0]
    if n_subj < 2:
        raise ValueError("isc requires at least 2 subjects")

    if method == "loo":
        total = data.sum(axis=0)
        corrs = np.empty((n_subj, *data.shape[2:]), dtype=np.float64)
        for i in range(n_subj):
            template = (total - data[i]) / (n_subj - 1)
            corrs[i] = _pearson_per_voxel(data[i], template)
        agg = _aggregate_corrs(corrs, metric)
        return {
            "isc": _make_braindata(agg, bc._mask),
            "per_subject": corrs,
        }

    # pairwise
    from itertools import combinations

    pairs = list(combinations(range(n_subj), 2))
    pair_corrs = np.empty((len(pairs), *data.shape[2:]), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        pair_corrs[k] = _pearson_per_voxel(data[i], data[j])
    agg = _aggregate_corrs(pair_corrs, metric)
    return {
        "isc": _make_braindata(agg, bc._mask),
        "pairs": pair_corrs,
    }


def isc_test(
    bc: BrainCollection,
    *,
    method: str = "loo",
    roi_mask: nib.Nifti1Image | Path | str | None = None,
    radius_mm: float | None = 6.0,
    n_permute: int = 5000,
    permutation_method: str = "bootstrap",
    metric: str = "median",
    device: str = "cpu",
    n_jobs: int = -1,
    progress_bar: bool = False,
    random_state: int | None = None,
) -> dict:
    """Bootstrap inference on ISC.

    Resamples subjects with replacement, recomputes ISC each draw, and
    derives a per-voxel p-value from the null distribution centered at 0.
    """
    rng = np.random.default_rng(random_state)
    observed = isc(bc, method=method, metric=metric)
    obs_map = np.asarray(observed["isc"].data).reshape(-1)
    n_subj = len(bc)

    null = np.empty((n_permute, obs_map.size), dtype=np.float64)
    data = np.stack(list(_iter_arrays(bc)), axis=0).astype(np.float64)
    for k in range(n_permute):
        idx = rng.integers(0, n_subj, size=n_subj)
        sample_data = data[idx]
        if method == "loo":
            total = sample_data.sum(axis=0)
            corrs = np.empty((n_subj, *data.shape[2:]), dtype=np.float64)
            denom = max(n_subj - 1, 1)
            for i in range(n_subj):
                template = (total - sample_data[i]) / denom
                corrs[i] = _pearson_per_voxel(sample_data[i], template)
            null[k] = _aggregate_corrs(corrs, metric).reshape(-1)
        else:
            from itertools import combinations

            pairs = list(combinations(range(n_subj), 2))
            pc = np.empty((len(pairs), *data.shape[2:]), dtype=np.float64)
            for kk, (i, j) in enumerate(pairs):
                pc[kk] = _pearson_per_voxel(sample_data[i], sample_data[j])
            null[k] = _aggregate_corrs(pc, metric).reshape(-1)

    # Two-tailed p centered at 0 (ISC null hypothesis: no synchrony → ISC = 0).
    p = (np.sum(np.abs(null) >= np.abs(obs_map), axis=0) + 1) / (n_permute + 1)
    return {
        "isc": observed["isc"],
        "p": _make_braindata(p.reshape(obs_map.shape), bc._mask),
        "null_distribution": null,
    }


def align(
    bc: BrainCollection,
    *,
    method: str = "procrustes",
    scheme: str = "searchlight",
    radius_mm: float = 10.0,
    parcellation: nib.Nifti1Image | None = None,
    n_features: int | None = None,
    n_iter: int = 3,
    device: str = "cpu",
    return_model: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = False,
    cache: Literal["auto", True, False] = "auto",
):
    """Functional alignment via ``LocalAlignment``.

    Materializes all subjects (algorithm constraint in v0.6.0). Returns
    a new ``BrainCollection`` of aligned data, or
    ``(BrainCollection, LocalAlignment)`` when ``return_model=True``.
    """
    from ..braindata import BrainData
    from ...algorithms.alignment.local import LocalAlignment

    _check_nonempty(bc)
    arrays = [np.asarray(x) for x in _iter_arrays(bc)]

    aligner = LocalAlignment(
        scheme=scheme,
        method=method,
        radius_mm=radius_mm,
        parcellation=parcellation,
        n_features=n_features,
        n_iter=n_iter,
        parallel=device,
        n_jobs=n_jobs,
    )
    aligner.fit(arrays, mask=bc._mask)
    aligned = aligner.transform(arrays)

    new_brains = [BrainData(arr.astype(np.float32), mask=bc._mask) for arr in aligned]
    new_bc = bc._clone(
        _items=new_brains,
        _step_id=bc._next_step_id(),
        _source_paths=[None] * len(new_brains),
    )
    if return_model:
        return new_bc, aligner
    return new_bc
