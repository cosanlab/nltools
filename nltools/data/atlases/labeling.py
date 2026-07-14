"""Coordinate-level atlas labeling.

Adapted from [atlasreader](https://github.com/miykael/atlasreader)
(BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from .loading import Atlas, load_atlas

CoordsLike = ArrayLike | Sequence[Sequence[float]]


def _as_xyz_array(coords: CoordsLike) -> np.ndarray:
    """Coerce coords to an ``(N, 3)`` float64 array."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {arr.shape}")
    return arr


def _xyz_to_ijk(coords_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map MNI mm → integer voxel ijk via the inverse affine.

    Out-of-bounds voxels get clamped to the origin (treated as background).
    """
    homog = np.hstack([coords_xyz, np.ones((coords_xyz.shape[0], 1))])
    ijk = np.linalg.solve(affine, homog.T)[:3].T
    return np.round(ijk).astype(int)


def _clip_to_box(ijk: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Clamp out-of-bounds voxel indices to the origin (background)."""
    box = np.array(shape[:3])
    out = np.any((ijk < 0) | (ijk >= box), axis=1)
    ijk = ijk.copy()
    ijk[out] = 0
    return ijk


def _label_lookup(atlas: Atlas) -> dict[int, str]:
    """Build an ``{integer index → region name}`` dict from atlas labels."""
    return dict(
        zip(
            atlas.labels["index"].to_list(),
            atlas.labels["name"].to_list(),
            strict=True,
        )
    )


def _label_deterministic(atlas: Atlas, ijk: np.ndarray) -> list[str]:
    """Look up region names for voxels in a deterministic atlas."""
    data = atlas.image.get_fdata()
    lut = _label_lookup(atlas)
    out: list[str] = []
    for v in ijk:
        idx = int(data[v[0], v[1], v[2]])
        out.append(lut.get(idx, "no_label"))
    return out


def _format_prob_entry(pct: float, name: str) -> str:
    """Format a single ``"pct%% name"`` entry for probabilistic output."""
    return f"{pct:.1f}% {name}"


def _label_probabilistic(
    atlas: Atlas, ijk: np.ndarray, prob_threshold: float
) -> list[str]:
    """Format ``"X% Foo; Y% Bar"`` strings per voxel for a probabilistic atlas.

    Regions with probability below ``prob_threshold`` are dropped; if no
    region survives, the voxel gets ``"no_label"``.
    """
    data = atlas.image.get_fdata()
    lut = _label_lookup(atlas)
    out: list[str] = []
    for v in ijk:
        probs = np.asarray(data[v[0], v[1], v[2]], dtype=float)
        keep = np.where(probs >= prob_threshold)[0]
        if keep.size == 0:
            out.append("no_label")
            continue
        order = keep[np.argsort(probs[keep])[::-1]]
        out.append(
            "; ".join(
                _format_prob_entry(float(probs[i]), lut.get(int(i), "no_label"))
                for i in order
            )
        )
    return out


def _labels_for_atlas(
    atlas: Atlas, ijk: np.ndarray, prob_threshold: float
) -> list[str]:
    """Dispatch to the deterministic or probabilistic labeling path."""
    if atlas.kind == "probabilistic":
        return _label_probabilistic(atlas, ijk, prob_threshold)
    return _label_deterministic(atlas, ijk)


def label_coords(
    coords: CoordsLike,
    *,
    atlas: str | Sequence[str] = "harvard_oxford",
    prob_threshold: float = 5.0,
) -> pl.DataFrame:
    """Look up anatomical labels for a set of MNI mm coordinates.

    For each coordinate, returns the atlas region(s) it falls in. Works
    for both deterministic atlases (single label per coord) and
    probabilistic atlases (formatted ``"42.0% Foo; 18.0% Bar"`` strings,
    sorted by descending probability).

    Args:
        coords: ``(N, 3)`` array-like of MNI mm coordinates ``(x, y, z)``.
            A single coord like ``(-42, -22, 56)`` is also accepted.
        atlas: Atlas name or list of names from `list_atlases`.
            One column is added to the output per atlas.
        prob_threshold: For probabilistic atlases only — drop regions
            with probability (in percent units) below this threshold.

    Returns:
        Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
        column per atlas. All atlas columns are ``Utf8``.
    """
    xyz = _as_xyz_array(coords)
    atlas_names = [atlas] if isinstance(atlas, str) else list(atlas)

    columns: dict[str, list] = {
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
    }

    for name in atlas_names:
        a = load_atlas(name)
        ijk = _xyz_to_ijk(xyz, a.image.affine)
        ijk = _clip_to_box(ijk, a.image.shape)
        columns[name] = _labels_for_atlas(a, ijk, prob_threshold)

    return pl.DataFrame(columns)
