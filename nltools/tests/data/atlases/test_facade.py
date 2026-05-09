"""Tests for BrainData.cluster_report facade + ClusterReport.plot."""

import nibabel as nb
import numpy as np
import pytest

from nltools.data import BrainData
from nltools.data.atlases import ClusterReport


def _gaussian_blob(shape, center_ijk, peak_amp, sigma=2.0):
    grid = np.indices(shape).astype(float)
    cz, cy, cx = (grid[i] - center_ijk[i] for i in range(3))
    return peak_amp * np.exp(-(cz**2 + cy**2 + cx**2) / (2 * sigma**2))


@pytest.fixture(scope="module")
def stat_brain():
    """Same synthetic brain as test_cluster_report."""
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
    mask = nb.Nifti1Image((np.abs(data) > 0.01).astype(np.uint8), affine)
    return BrainData(img, mask=mask)


def test_braindata_cluster_report_returns_ClusterReport(stat_brain):
    report = stat_brain.cluster_report(
        stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    assert isinstance(report, ClusterReport)


def test_braindata_cluster_report_uses_default_atlases(stat_brain):
    report = stat_brain.cluster_report(stat_threshold=3.0, cluster_threshold=5)
    # Default trio should appear as columns
    for name in ("harvard_oxford", "aal", "schaefer_200"):
        assert name in report.clusters.columns


def test_braindata_cluster_report_two_sided_default(stat_brain):
    report = stat_brain.cluster_report(
        stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    # Default two_sided=True picks up the negative blob too → 3 clusters
    assert report.clusters.height == 3


def test_cluster_report_plot_writes_files(stat_brain, tmp_path):
    report = stat_brain.cluster_report(
        stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    report.plot(output_dir=tmp_path)
    assert (tmp_path / "overview.png").exists()
    # One per-cluster figure for each surviving cluster
    for cid in report.clusters["cluster_id"].to_list():
        assert (tmp_path / f"cluster_{cid:02d}.png").exists()


def test_cluster_report_plot_no_dir_returns_figures(stat_brain):
    report = stat_brain.cluster_report(
        stat_threshold=3.0, cluster_threshold=5, atlas="aal"
    )
    figures = report.plot()
    # Returns a list of (label, figure) tuples
    assert isinstance(figures, list)
    assert len(figures) == 1 + report.clusters.height  # overview + per-cluster
    import matplotlib.pyplot as plt

    plt.close("all")
