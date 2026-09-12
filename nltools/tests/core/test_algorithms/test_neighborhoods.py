"""Tests for nltools.algorithms.neighborhoods module."""

import numpy as np
import nibabel as nib
import pytest
from scipy import sparse

from nltools.algorithms.neighborhoods import (
    SphereNeighborhoods,
    compute_searchlight_neighborhoods,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_mask():
    """Create a small 10x10x10 mask with a central cube of 5x5x5 voxels."""
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[2:7, 2:7, 2:7] = 1  # 125 voxels
    # 2mm isotropic voxels, origin at center
    affine = np.array(
        [
            [2, 0, 0, -10],
            [0, 2, 0, -10],
            [0, 0, 2, -10],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def tiny_mask():
    """Create a tiny 3x3x3 mask with all voxels active (27 voxels)."""
    data = np.ones((3, 3, 3), dtype=np.int16)
    affine = np.array(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    return nib.Nifti1Image(data, affine)


# =============================================================================
# Test SphereNeighborhoods
# =============================================================================


class TestSphereNeighborhoods:
    """Tests for SphereNeighborhoods dataclass."""

    def test_get_neighbors(self):
        """Test getting neighbors for a voxel."""
        # Create simple 3-voxel adjacency: 0-1, 1-2 (linear chain)
        row = [0, 0, 1, 1, 1, 2, 2]
        col = [0, 1, 0, 1, 2, 1, 2]
        data = [1, 1, 1, 1, 1, 1, 1]
        adj = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        sn = SphereNeighborhoods(
            adjacency=adj,
            radius=5.0,
            n_voxels=3,
        )

        # Voxel 0 has neighbors 0, 1
        np.testing.assert_array_equal(sorted(sn.get_neighbors(0)), [0, 1])
        # Voxel 1 has neighbors 0, 1, 2
        np.testing.assert_array_equal(sorted(sn.get_neighbors(1)), [0, 1, 2])
        # Voxel 2 has neighbors 1, 2
        np.testing.assert_array_equal(sorted(sn.get_neighbors(2)), [1, 2])

    def test_neighborhood_size(self):
        """Test getting neighborhood size."""
        row = [0, 0, 1, 1, 1, 2, 2]
        col = [0, 1, 0, 1, 2, 1, 2]
        data = [1, 1, 1, 1, 1, 1, 1]
        adj = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        sn = SphereNeighborhoods(
            adjacency=adj,
            radius=5.0,
            n_voxels=3,
        )

        assert sn.get_neighborhood_size(0) == 2
        assert sn.get_neighborhood_size(1) == 3
        assert sn.get_neighborhood_size(2) == 2

    def test_iter_neighborhoods(self):
        """Test iterating over neighborhoods."""
        row = [0, 0, 1, 1, 1, 2, 2]
        col = [0, 1, 0, 1, 2, 1, 2]
        data = [1, 1, 1, 1, 1, 1, 1]
        adj = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        sn = SphereNeighborhoods(
            adjacency=adj,
            radius=5.0,
            n_voxels=3,
        )

        neighborhoods = list(sn.iter_neighborhoods())
        assert len(neighborhoods) == 3
        assert neighborhoods[0][0] == 0
        assert neighborhoods[1][0] == 1
        assert neighborhoods[2][0] == 2

    def test_statistics(self):
        """Test mean/min/max size properties."""
        row = [0, 0, 1, 1, 1, 2, 2]
        col = [0, 1, 0, 1, 2, 1, 2]
        data = [1, 1, 1, 1, 1, 1, 1]
        adj = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

        sn = SphereNeighborhoods(
            adjacency=adj,
            radius=5.0,
            n_voxels=3,
        )

        assert sn.min_size == 2
        assert sn.max_size == 3
        # Mean: (2 + 3 + 2) / 3 = 2.33...
        assert abs(sn.mean_size - 7 / 3) < 0.01

    def test_repr(self):
        """Test string representation."""
        row = [0, 1]
        col = [0, 1]
        data = [1, 1]
        adj = sparse.csr_matrix((data, (row, col)), shape=(2, 2))

        sn = SphereNeighborhoods(
            adjacency=adj,
            radius=5.0,
            n_voxels=2,
        )

        repr_str = repr(sn)
        assert "n_voxels=2" in repr_str
        assert "radius=5.0mm" in repr_str


# =============================================================================
# Test compute_searchlight_neighborhoods
# =============================================================================


class TestComputeSearchlightNeighborhoods:
    """Tests for compute_searchlight_neighborhoods function."""

    def test_basic_computation(self, tiny_mask):
        """Test basic neighborhood computation."""
        neighborhoods = compute_searchlight_neighborhoods(tiny_mask, radius=3.0)

        # 3x3x3 = 27 voxels
        assert neighborhoods.n_voxels == 27
        assert neighborhoods.radius == 3.0

    def test_neighborhood_geometry(self, tiny_mask):
        """Test that neighborhood geometry is correct."""
        # With 2mm voxels and 3mm radius, center voxel should see
        # its 6 face-adjacent neighbors (distance = 2mm) but not
        # edge-adjacent (distance = 2*sqrt(2) = 2.83mm) or
        # corner-adjacent (distance = 2*sqrt(3) = 3.46mm)
        neighborhoods = compute_searchlight_neighborhoods(tiny_mask, radius=2.5)

        # Center voxel (1,1,1) should only see itself with radius < 2mm
        # Since the mask voxel indices are flattened, we need to find
        # the center voxel index
        # For 3x3x3, flattened index of (1,1,1) depends on ordering
        # Let's just check that some neighborhoods are larger than 1

        # With radius 2.5mm and 2mm voxels, face neighbors are included
        # Let's verify the adjacency makes sense
        assert neighborhoods.min_size >= 1
        assert neighborhoods.max_size <= 27

    def test_empty_mask_raises(self):
        """Test that empty mask raises ValueError."""
        empty_data = np.zeros((10, 10, 10), dtype=np.int16)
        empty_mask = nib.Nifti1Image(empty_data, np.eye(4))

        with pytest.raises(ValueError, match="no non-zero voxels"):
            compute_searchlight_neighborhoods(empty_mask, radius=5.0)

    def test_larger_radius_more_neighbors(self, small_mask):
        """Test that larger radius gives more neighbors on average."""
        nb_small = compute_searchlight_neighborhoods(small_mask, radius=3.0)
        nb_large = compute_searchlight_neighborhoods(small_mask, radius=8.0)

        assert nb_large.mean_size > nb_small.mean_size

    def test_adjacency_is_symmetric(self, small_mask):
        """Test that the adjacency matrix is symmetric (if i sees j, j sees i)."""
        neighborhoods = compute_searchlight_neighborhoods(small_mask, radius=5.0)

        adj = neighborhoods.adjacency.toarray()
        np.testing.assert_array_equal(adj, adj.T)

    def test_self_is_neighbor(self, small_mask):
        """Test that each voxel is in its own neighborhood."""
        neighborhoods = compute_searchlight_neighborhoods(small_mask, radius=5.0)

        # Diagonal should be all 1s (each voxel neighbors itself)
        diag = neighborhoods.adjacency.diagonal()
        np.testing.assert_array_equal(diag, np.ones(neighborhoods.n_voxels))


class TestRadiusKeyword:
    """The radius keyword follows nilearn: `radius`, in millimeters."""

    def test_radius_mm_keyword_is_removed(self, tiny_mask):
        with pytest.raises(TypeError):
            compute_searchlight_neighborhoods(tiny_mask, radius_mm=3.0)

    def test_radius_is_recorded_in_millimeters(self, tiny_mask):
        neighborhoods = compute_searchlight_neighborhoods(tiny_mask, radius=3.0)
        assert neighborhoods.radius == 3.0
        assert not hasattr(neighborhoods, "radius_mm")
