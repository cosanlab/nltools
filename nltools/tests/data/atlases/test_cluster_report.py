"""Tests for _cluster_report_data + _ClusterReport."""

import nibabel as nb
import numpy as np
import polars as pl
import pytest

from nltools.data import BrainData
from nltools.data.atlases.reporting import (
    _ClusterReport,
    _cluster_report_data,
)


# ---------------------------------------------------------------------------
# Fixture: a synthetic 2mm MNI-space stat map with two Gaussian blobs at
# known locations. Built directly as a Nifti1Image, then wrapped in BrainData.
# ---------------------------------------------------------------------------


def _gaussian_blob(shape, center_ijk, peak_amp, sigma=2.0):
    grid = np.indices(shape).astype(float)
    cz, cy, cx = (grid[i] - center_ijk[i] for i in range(3))
    return peak_amp * np.exp(-(cz**2 + cy**2 + cx**2) / (2 * sigma**2))


@pytest.fixture(scope="module")
def synthetic_stat_brain():
    """A stat map with two positive blobs and one negative blob.

    2mm MNI-space affine, blobs placed at:
    - (-42, -22, 56)  ≈ left M1/S1 hand area, +6.0
    - (+42, -22, 56)  ≈ right M1/S1 hand area, +5.0
    - (  0, -78,  8)  ≈ medial occipital, -4.0
    """
    # 2mm MNI affine — origin at (-90, -126, -72)
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

    data += _gaussian_blob(shape, mni_to_ijk((-42, -22, 56)), 6.0)
    data += _gaussian_blob(shape, mni_to_ijk((42, -22, 56)), 5.0)
    data += _gaussian_blob(shape, mni_to_ijk((0, -78, 8)), -4.0)

    img = nb.Nifti1Image(data, affine)
    # Build a brain mask from non-zero voxels (loose enough to cover blobs)
    mask_data = (np.abs(data) > 0.01).astype(np.uint8)
    mask = nb.Nifti1Image(mask_data, affine)
    return BrainData(img, mask=mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_peaks_dataframe_columns(synthetic_stat_brain):
    peaks, _, _ = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    expected = {
        "cluster_id",
        "x",
        "y",
        "z",
        "peak_stat",
        "volume_mm3",
        "n_voxels",
        "aal",
    }
    assert expected.issubset(set(peaks.columns))


def test_clusters_dataframe_columns(synthetic_stat_brain):
    _, clusters, _ = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    expected = {
        "cluster_id",
        "peak_x",
        "peak_y",
        "peak_z",
        "mean_stat",
        "volume_mm3",
        "n_voxels",
        "aal",
    }
    assert expected.issubset(set(clusters.columns))


def test_two_sided_finds_both_signs(synthetic_stat_brain):
    _, clusters, _ = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        two_sided=True,
        atlas="aal",
    )
    # Should find the two positive + one negative blob → 3 clusters
    assert clusters.height == 3
    # Mean stats should span both signs
    mean_stats = clusters["mean_stat"].to_numpy()
    assert (mean_stats > 0).any()
    assert (mean_stats < 0).any()


def test_one_sided_skips_negatives(synthetic_stat_brain):
    _, clusters, _ = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        two_sided=False,
        atlas="aal",
    )
    # Only the two positive blobs survive
    assert clusters.height == 2
    assert (clusters["mean_stat"].to_numpy() > 0).all()


def test_cluster_threshold_filters_small_clusters(synthetic_stat_brain):
    # Huge cluster_threshold should drop everything
    _, clusters, _ = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=10000,
        atlas="aal",
    )
    assert clusters.height == 0


def test_returned_thresholded_brain_is_BrainData(synthetic_stat_brain):
    _, _, thr = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    assert isinstance(thr, BrainData)


# ---------------------------------------------------------------------------
# _ClusterReport dataclass
# ---------------------------------------------------------------------------


def test_cluster_report_dataclass(synthetic_stat_brain):
    peaks, clusters, thr = _cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    report = _ClusterReport(peaks=peaks, clusters=clusters, stat_img=thr)
    assert report.peaks is peaks
    assert report.clusters is clusters
    assert report.stat_img is thr


# ---------------------------------------------------------------------------
# Sub-peak clusters (F042/F043)
#
# nilearn's get_clusters_table emits one row per peak AND per sub-peak; sub-peak
# rows carry an empty string '' in the 'Cluster Size (mm3)' column. The tests
# below build a single connected cluster with two local maxima so nilearn emits
# sub-peak rows.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_peak_brain():
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


def test_peaks_cluster_id_shares_integer_label_space(two_peak_brain):
    """F043: peaks.cluster_id must use the SAME integer id space as clusters.

    Previously peaks.cluster_id came from nilearn (strings '1'/'1a', ordered by
    peak stat) while clusters.cluster_id was renumbered by size (int) — different
    orderings AND dtypes, so the two tables couldn't be joined. Peak ids are now
    looked up in the renumbered label volume, so they share one integer space and
    sub-peaks inherit their parent cluster's id.
    """
    peaks, clusters, thr = _cluster_report_data(
        two_peak_brain, stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    assert peaks["cluster_id"].dtype == pl.Int64
    assert clusters["cluster_id"].dtype == pl.Int64

    peak_ids = set(peaks["cluster_id"].to_list())
    cluster_ids = set(clusters["cluster_id"].to_list())
    assert peak_ids <= cluster_ids, (
        f"peak cluster_ids {peak_ids} are not a subset of cluster ids "
        f"{cluster_ids} — the tables are not joinable"
    )
    # The two local maxima form ONE connected cluster -> one shared id.
    assert len(peak_ids) == 1
    # And a join actually works.
    joined = peaks.join(clusters, on="cluster_id", how="inner")
    assert joined.height == peaks.height
