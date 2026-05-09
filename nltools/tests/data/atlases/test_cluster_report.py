"""Tests for cluster_report_data + ClusterReport."""

import nibabel as nb
import numpy as np
import polars as pl
import pytest

from nltools.data import BrainData
from nltools.data.atlases.reporting import (
    ClusterReport,
    cluster_report_data,
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


def test_cluster_report_data_returns_polars(synthetic_stat_brain):
    peaks, clusters, thr = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    assert isinstance(peaks, pl.DataFrame)
    assert isinstance(clusters, pl.DataFrame)


def test_peaks_dataframe_columns(synthetic_stat_brain):
    peaks, _, _ = cluster_report_data(
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
    _, clusters, _ = cluster_report_data(
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
    _, clusters, _ = cluster_report_data(
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
    _, clusters, _ = cluster_report_data(
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
    _, clusters, _ = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=10000,
        atlas="aal",
    )
    assert clusters.height == 0


def test_pre_thresholded_input(synthetic_stat_brain):
    """stat_threshold=None should treat input as already thresholded."""
    # First, threshold once via stat_threshold=3
    _, clusters_a, thr_a = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    # Then re-pass the thresholded BrainData with stat_threshold=None
    _, clusters_b, _ = cluster_report_data(
        thr_a,
        stat_threshold=None,
        cluster_threshold=5,
        atlas="aal",
    )
    assert clusters_a.height == clusters_b.height


def test_atlas_columns_appear(synthetic_stat_brain):
    peaks, clusters, _ = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas=["aal", "harvard_oxford"],
    )
    assert "aal" in peaks.columns
    assert "harvard_oxford" in peaks.columns
    assert "aal" in clusters.columns
    assert "harvard_oxford" in clusters.columns


def test_cluster_label_format_is_mass_weighted(synthetic_stat_brain):
    """Cluster-level labels should be ``'XX.X% Region; ...'`` strings."""
    _, clusters, _ = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    for s in clusters["aal"].to_list():
        assert "%" in s


def test_returned_thresholded_brain_is_BrainData(synthetic_stat_brain):
    _, _, thr = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    assert isinstance(thr, BrainData)


# ---------------------------------------------------------------------------
# ClusterReport dataclass
# ---------------------------------------------------------------------------


def test_cluster_report_dataclass(synthetic_stat_brain):
    peaks, clusters, thr = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    report = ClusterReport(peaks=peaks, clusters=clusters, stat_img=thr)
    assert report.peaks is peaks
    assert report.clusters is clusters
    assert report.stat_img is thr


def test_cluster_report_to_csv(synthetic_stat_brain, tmp_path):
    peaks, clusters, thr = cluster_report_data(
        synthetic_stat_brain,
        stat_threshold=3.0,
        cluster_threshold=5,
        atlas="aal",
    )
    report = ClusterReport(peaks=peaks, clusters=clusters, stat_img=thr)
    report.to_csv(tmp_path)
    assert (tmp_path / "peaks.csv").exists()
    assert (tmp_path / "clusters.csv").exists()
