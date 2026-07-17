"""Regression test for F042: cluster_report crashes on sub-peak clusters.

nilearn's get_clusters_table emits one row per peak AND per sub-peak; sub-peak
rows carry an empty string '' in the 'Cluster Size (mm3)' column. The old
`_build_peaks_dataframe` did `to_numpy(dtype=float)` on that column, raising
ValueError the moment any cluster had more than one local maximum — the common
case for real fMRI stat maps. This builds a single connected cluster with two
local maxima so nilearn emits sub-peak rows.
"""

import nibabel as nb
import numpy as np
import polars as pl

from nltools.data import BrainData
from nltools.data.atlases.reporting import cluster_report_data


def _gaussian_blob(shape, center_ijk, peak_amp, sigma):
    grid = np.indices(shape).astype(float)
    cz, cy, cx = (grid[i] - center_ijk[i] for i in range(3))
    return peak_amp * np.exp(-(cz**2 + cy**2 + cx**2) / (2 * sigma**2))


def _two_peak_brain():
    """Stat map with two nearby peaks fused into one connected cluster."""
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    shape = (91, 109, 91)
    data = np.zeros(shape, dtype=np.float32)

    def mni_to_ijk(xyz):
        homog = np.append(xyz, 1.0)
        return tuple(int(round(v)) for v in np.linalg.solve(affine, homog)[:3])

    # Two peaks 12mm apart (> default min_distance of 8mm), wide enough that the
    # ridge between them stays above threshold -> one connected component, two
    # local maxima -> nilearn reports a main peak plus a sub-peak.
    data += _gaussian_blob(shape, mni_to_ijk((-6, -22, 56)), 8.0, sigma=2.0)
    data += _gaussian_blob(shape, mni_to_ijk((6, -22, 56)), 7.0, sigma=2.0)

    img = nb.Nifti1Image(data, affine)
    mask_data = (np.abs(data) > 0.01).astype(np.uint8)
    mask = nb.Nifti1Image(mask_data, affine)
    return BrainData(img, mask=mask)


def test_get_clusters_table_emits_subpeaks():
    """Sanity check: the synthetic image really produces sub-peak rows."""
    from nilearn.reporting import get_clusters_table

    bd = _two_peak_brain()
    table = get_clusters_table(
        bd.to_nifti(), stat_threshold=3.0, cluster_threshold=0, two_sided=True
    )
    ids = [str(c) for c in table["Cluster ID"].tolist()]
    # Sub-peak rows are labelled like '1a', '1b' and carry an empty size cell.
    assert any(not c.isdigit() for c in ids)
    assert (table["Cluster Size (mm3)"] == "").any()


def test_cluster_report_survives_subpeaks():
    """cluster_report_data must not crash on sub-peak clusters (F042)."""
    bd = _two_peak_brain()
    peaks, clusters, thr = cluster_report_data(
        bd, stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    assert isinstance(peaks, pl.DataFrame)
    # More than one peak row -> sub-peaks were present and handled.
    assert peaks.height >= 2
    # Sub-peaks inherit their parent cluster's size (forward-filled), so no
    # nulls and integer n_voxels are well-defined (not NaN-cast garbage).
    assert peaks["volume_mm3"].null_count() == 0
    assert (peaks["volume_mm3"].to_numpy() > 0).all()
    assert peaks["n_voxels"].null_count() == 0
    assert (peaks["n_voxels"].to_numpy() > 0).all()
