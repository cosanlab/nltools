"""Cluster reports — peak/cluster geometry plus atlas labels.

The peak/sub-peak geometry comes from `get_clusters_table`;
the cluster masks and mass-weighted labels are computed locally so we can
attribute every voxel of every cluster to one or more atlases.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nb
import nibabel.affines as nb_affines
import numpy as np
import polars as pl
from scipy import ndimage

from .labeling import _clip_to_box, _label_lookup, _xyz_to_ijk, label_coords
from .loading import Atlas, load_atlas
from .registry import DEFAULT_ATLASES

if TYPE_CHECKING:
    from nltools.data import BrainData


@dataclass(frozen=True)
class ClusterReport:
    """Result of `BrainData.cluster_report`.

    Attributes:
        peaks: Polars DataFrame, one row per peak (incl. sub-peaks). Columns
            ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``,
            ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas.
        clusters: Polars DataFrame, one row per cluster. Columns
            ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``,
            ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8
            column per atlas (mass-weighted top regions).
        stat_img: BrainData with the thresholded stat map (sub-cluster
            voxels and clusters smaller than ``cluster_threshold`` zeroed).
    """

    peaks: pl.DataFrame
    clusters: pl.DataFrame
    stat_img: "BrainData"

    def to_csv(self, output_dir: str | Path) -> None:
        """Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.peaks.write_csv(output_dir / "peaks.csv")
        self.clusters.write_csv(output_dir / "clusters.csv")

    def plot(
        self,
        *,
        output_dir: str | Path | None = None,
    ) -> list[tuple[str, Any]] | None:
        """Render an overview glass brain + one slice figure per cluster.

        Args:
            output_dir: If given, save ``overview.png`` and
                ``cluster_NN.png`` files into the directory and return
                ``None``. If omitted, return a list of
                ``(label, matplotlib.figure.Figure)`` tuples without
                writing to disk.

        Returns:
            ``None`` when ``output_dir`` is set, else a list of
            ``(label, figure)`` tuples.
        """
        from matplotlib import pyplot as plt
        from nilearn.plotting import plot_glass_brain, plot_stat_map

        thr_img = self.stat_img.to_nifti()
        figures: list[tuple[str, Any]] = []

        fig_overview = plt.figure(figsize=(10, 4))
        plot_glass_brain(
            thr_img,
            figure=fig_overview,
            display_mode="lyrz",
            colorbar=True,
            plot_abs=False,
        )
        figures.append(("overview", fig_overview))

        for row in self.clusters.iter_rows(named=True):
            cid = int(row["cluster_id"])
            cut = (row["peak_x"], row["peak_y"], row["peak_z"])
            fig = plt.figure(figsize=(10, 3))
            plot_stat_map(
                thr_img,
                figure=fig,
                cut_coords=cut,
                display_mode="ortho",
                colorbar=True,
                title=f"Cluster {cid}: peak ({cut[0]:.0f}, {cut[1]:.0f}, {cut[2]:.0f})",
            )
            figures.append((f"cluster_{cid:02d}", fig))

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for label, fig in figures:
                fig.savefig(output_dir / f"{label}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
            return None
        return figures


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_threshold_mask(
    data: np.ndarray, stat_threshold: float | None, two_sided: bool
) -> np.ndarray:
    """Return a boolean mask of voxels surviving the voxel-level threshold.

    If ``stat_threshold is None``, treat the input as already thresholded
    (only zero vs non-zero matters).
    """
    if stat_threshold is None:
        return data != 0 if two_sided else data > 0
    if two_sided:
        return np.abs(data) >= stat_threshold
    return data >= stat_threshold


def _label_clusters(data: np.ndarray, mask: np.ndarray, two_sided: bool) -> np.ndarray:
    """Sign-aware 26-connected component labeling.

    For ``two_sided=True``, label positive and negative regions
    independently so a positive blob touching a negative blob never
    merges. Negative-cluster IDs are offset above positive IDs.
    """
    structure = np.ones((3, 3, 3), dtype=int)
    if not two_sided:
        labels, _ = ndimage.label(mask, structure=structure)
        return labels
    pos_mask = mask & (data > 0)
    neg_mask = mask & (data < 0)
    pos_labels, n_pos = ndimage.label(pos_mask, structure=structure)
    neg_labels, _ = ndimage.label(neg_mask, structure=structure)
    neg_labels = np.where(neg_labels > 0, neg_labels + n_pos, 0)
    return pos_labels + neg_labels


def _filter_by_size(
    labels: np.ndarray, cluster_threshold: int
) -> tuple[np.ndarray, list[int]]:
    """Zero out clusters smaller than ``cluster_threshold`` voxels.

    Returns the filtered label volume and the list of surviving IDs in
    descending size order.
    """
    if labels.max() == 0:
        return labels, []
    sizes = np.bincount(labels.ravel())
    keep = [i for i in range(1, len(sizes)) if sizes[i] >= cluster_threshold]
    keep.sort(key=lambda i: -sizes[i])
    keep_set = set(keep)
    filtered = np.where(np.isin(labels, list(keep_set)), labels, 0)
    return filtered, keep


def _renumber_labels(
    labels: np.ndarray, ordered_ids: list[int]
) -> tuple[np.ndarray, dict[int, int]]:
    """Remap labels to 1..K following ``ordered_ids`` (largest cluster = 1)."""
    out = np.zeros_like(labels)
    id_map: dict[int, int] = {}
    for new_id, old_id in enumerate(ordered_ids, start=1):
        out[labels == old_id] = new_id
        id_map[old_id] = new_id
    return out, id_map


def _cluster_label_string(
    atlas: Atlas, ijk: np.ndarray, prob_threshold: float, top_k: int = 5
) -> str:
    """Tally regions across all voxels in a cluster, return a formatted string.

    For deterministic atlases, each voxel contributes exactly one region.
    For probabilistic atlases, each voxel contributes its argmax region
    (or ``no_label`` if its top probability is below ``prob_threshold``).
    Output: ``"38.2% Foo; 12.1% Bar; ..."`` sorted by descending share.
    """
    data = atlas.image.get_fdata()
    lut = _label_lookup(atlas)
    n = ijk.shape[0]
    if n == 0:
        return ""

    if atlas.kind == "deterministic":
        ids = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(int)
        names = [lut.get(int(i), "no_label") for i in ids]
    else:  # probabilistic
        probs = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]  # (M, K)
        best = probs.argmax(axis=1)
        max_prob = probs.max(axis=1)
        names = [
            "no_label" if p < prob_threshold else lut.get(int(b), "no_label")
            for b, p in zip(best, max_prob, strict=True)
        ]

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]
    return "; ".join(f"{100 * c / n:.1f}% {name}" for name, c in items)


def _peak_voxel(
    data: np.ndarray, cluster_mask: np.ndarray, two_sided: bool
) -> tuple[int, int, int]:
    """Return ijk of the voxel with the largest |value| (or value) in cluster."""
    masked = np.where(cluster_mask, data, np.nan)
    if two_sided:
        flat = np.nanargmax(np.abs(masked))
    else:
        flat = np.nanargmax(masked)
    i, j, k = np.unravel_index(flat, data.shape)
    return int(i), int(j), int(k)


def _build_clusters_dataframe(
    data: np.ndarray,
    labels: np.ndarray,
    affine: np.ndarray,
    *,
    atlas_objs: list[Atlas],
    prob_threshold: float,
    two_sided: bool,
    voxel_volume_mm3: float,
) -> pl.DataFrame:
    """Build the per-cluster summary DataFrame with mass-weighted labels."""
    rows: list[dict] = []
    n_clusters = int(labels.max())
    for cid in range(1, n_clusters + 1):
        cluster_mask = labels == cid
        ijk = np.argwhere(cluster_mask)
        n_vox = ijk.shape[0]
        peak_ijk = _peak_voxel(data, cluster_mask, two_sided)
        peak_xyz = nb_affines.apply_affine(affine, np.asarray(peak_ijk))
        cluster_vals = data[cluster_mask]
        row: dict = {
            "cluster_id": cid,
            "peak_x": float(peak_xyz[0]),
            "peak_y": float(peak_xyz[1]),
            "peak_z": float(peak_xyz[2]),
            "mean_stat": float(cluster_vals.mean()),
            "volume_mm3": float(n_vox * voxel_volume_mm3),
            "n_voxels": int(n_vox),
        }
        for atlas in atlas_objs:
            atlas_ijk = _xyz_to_ijk(
                nb_affines.apply_affine(affine, ijk), atlas.image.affine
            )
            atlas_ijk = _clip_to_box(atlas_ijk, atlas.image.shape)
            row[atlas.name] = _cluster_label_string(atlas, atlas_ijk, prob_threshold)
        rows.append(row)

    if not rows:
        # Build empty schema-compatible frame
        schema = {
            "cluster_id": pl.Int64,
            "peak_x": pl.Float64,
            "peak_y": pl.Float64,
            "peak_z": pl.Float64,
            "mean_stat": pl.Float64,
            "volume_mm3": pl.Float64,
            "n_voxels": pl.Int64,
            **{a.name: pl.Utf8 for a in atlas_objs},
        }
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows)


def _build_peaks_dataframe(
    thr_img: nb.Nifti1Image,
    *,
    stat_threshold: float | None,
    two_sided: bool,
    min_distance: float,
    atlas_names: list[str],
    prob_threshold: float,
    voxel_volume_mm3: float,
) -> pl.DataFrame:
    """Use nilearn's get_clusters_table for peaks/sub-peaks, then add labels."""
    from nilearn.reporting import get_clusters_table

    thresh = 0.0 if stat_threshold is None else float(stat_threshold)
    table = get_clusters_table(
        thr_img,
        stat_threshold=thresh,
        cluster_threshold=0,
        two_sided=two_sided,
        min_distance=min_distance,
    )
    if len(table) == 0:
        schema = {
            "cluster_id": pl.Utf8,
            "x": pl.Float64,
            "y": pl.Float64,
            "z": pl.Float64,
            "peak_stat": pl.Float64,
            "volume_mm3": pl.Float64,
            "n_voxels": pl.Int64,
            **dict.fromkeys(atlas_names, pl.Utf8),
        }
        return pl.DataFrame(schema=schema)

    # Rename pandas → polars-friendly names. nilearn returns columns:
    #   'Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)'
    coords = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    labels = label_coords(
        coords, atlas=atlas_names, prob_threshold=prob_threshold
    ).drop(["x", "y", "z"])

    base = pl.DataFrame(
        {
            "cluster_id": [str(c) for c in table["Cluster ID"].tolist()],
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "peak_stat": table["Peak Stat"].to_numpy(dtype=float),
            "volume_mm3": table["Cluster Size (mm3)"].to_numpy(dtype=float),
            "n_voxels": (
                table["Cluster Size (mm3)"].to_numpy(dtype=float) / voxel_volume_mm3
            )
            .round()
            .astype(int),
        }
    )
    return pl.concat([base, labels], how="horizontal")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cluster_report_data(
    bd: "BrainData",
    *,
    stat_threshold: float | None = 3.0,
    cluster_threshold: int = 10,
    two_sided: bool = True,
    min_distance: float = 8.0,
    atlas: str | Sequence[str] = DEFAULT_ATLASES,
    prob_threshold: float = 5.0,
) -> tuple[pl.DataFrame, pl.DataFrame, "BrainData"]:
    """Compute cluster report DataFrames + thresholded BrainData.

    Pure function — the BrainData facade `BrainData.cluster_report`
    wraps the result in a `ClusterReport`.

    Args:
        bd: BrainData with a 3D stat map (single sample).
        stat_threshold: Voxel-level threshold. ``None`` means treat ``bd``
            as already thresholded (skip voxel filtering, keep all
            non-zero voxels).
        cluster_threshold: Minimum cluster size in voxels.
        two_sided: Report negative clusters as separate clusters.
        min_distance: Minimum distance (mm) between sub-peaks. Passed to
            `get_clusters_table`.
        atlas: Atlas name or list of names from `list_atlases`.
        prob_threshold: Drop probabilistic-atlas regions below this %.

    Returns:
        Tuple ``(peaks, clusters, thresholded_bd)``.
    """
    from nltools.data import BrainData

    img = bd.to_nifti()
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"cluster_report requires a single 3D stat map; got shape {data.shape}"
        )
    affine = img.affine
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    mask = _build_threshold_mask(data, stat_threshold, two_sided)
    raw_labels = _label_clusters(data, mask, two_sided)
    filtered, kept = _filter_by_size(raw_labels, cluster_threshold)
    renumbered, _ = _renumber_labels(filtered, kept)

    thr_data = np.where(renumbered > 0, data, 0.0).astype(np.float32)
    thr_img = nb.Nifti1Image(thr_data, affine)
    thr_bd = BrainData(thr_img, mask=bd.mask)

    atlas_names = [atlas] if isinstance(atlas, str) else list(atlas)
    atlas_objs = [load_atlas(name) for name in atlas_names]

    peaks = _build_peaks_dataframe(
        thr_img,
        stat_threshold=stat_threshold,
        two_sided=two_sided,
        min_distance=min_distance,
        atlas_names=atlas_names,
        prob_threshold=prob_threshold,
        voxel_volume_mm3=voxel_volume_mm3,
    )
    clusters = _build_clusters_dataframe(
        data,
        renumbered,
        affine,
        atlas_objs=atlas_objs,
        prob_threshold=prob_threshold,
        two_sided=two_sided,
        voxel_volume_mm3=voxel_volume_mm3,
    )

    return peaks, clusters, thr_bd
