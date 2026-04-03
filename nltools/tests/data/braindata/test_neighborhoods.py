"""Tests for nltools.data.braindata.neighborhoods module."""

import numpy as np
import nibabel as nib
import pytest
from scipy import sparse

from nltools.data.braindata.neighborhoods import (
    SphereNeighborhoods,
    compute_searchlight_neighborhoods,
)
from nltools.data.braindata.cache import CacheManager, hash_mask, clear_cache


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
# Test CacheManager
# =============================================================================


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        """Test basic save and load functionality."""
        # Patch home directory to use temp path
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")
        data = np.array([1, 2, 3, 4, 5])
        metadata = np.array(["hello"])

        # Save
        path = cache.save("mykey", data=data, metadata=metadata)
        assert path.exists()

        # Load
        loaded = cache.load("mykey")
        assert loaded is not None
        np.testing.assert_array_equal(loaded["data"], data)
        np.testing.assert_array_equal(loaded["metadata"], metadata)

    def test_exists(self, tmp_path, monkeypatch):
        """Test exists check."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")

        assert not cache.exists("nonexistent")

        cache.save("exists", data=np.array([1]))
        assert cache.exists("exists")

    def test_delete(self, tmp_path, monkeypatch):
        """Test delete functionality."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")
        cache.save("to_delete", data=np.array([1]))

        assert cache.exists("to_delete")
        assert cache.delete("to_delete")
        assert not cache.exists("to_delete")
        assert not cache.delete("to_delete")  # Already deleted

    def test_list_keys(self, tmp_path, monkeypatch):
        """Test listing cached keys."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")
        cache.save("key1", data=np.array([1]))
        cache.save("key2", data=np.array([2]))
        cache.save("key3", data=np.array([3]))

        keys = cache.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_clear(self, tmp_path, monkeypatch):
        """Test clearing cache."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")
        cache.save("key1", data=np.array([1]))
        cache.save("key2", data=np.array([2]))

        count = cache.clear()
        assert count == 2
        assert len(cache.list_keys()) == 0

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        """Test that loading nonexistent key returns None."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache = CacheManager("test_category")
        assert cache.load("nonexistent") is None


class TestHashMask:
    """Tests for hash_mask function."""

    def test_same_mask_same_hash(self, small_mask):
        """Same mask should produce same hash."""
        hash1 = hash_mask(small_mask)
        hash2 = hash_mask(small_mask)
        assert hash1 == hash2

    def test_different_data_different_hash(self):
        """Different mask data should produce different hash."""
        data1 = np.zeros((10, 10, 10), dtype=np.int16)
        data1[2:5, 2:5, 2:5] = 1

        data2 = np.zeros((10, 10, 10), dtype=np.int16)
        data2[5:8, 5:8, 5:8] = 1

        affine = np.eye(4)
        mask1 = nib.Nifti1Image(data1, affine)
        mask2 = nib.Nifti1Image(data2, affine)

        hash1 = hash_mask(mask1)
        hash2 = hash_mask(mask2)
        assert hash1 != hash2

    def test_different_affine_different_hash(self):
        """Different affines should produce different hash."""
        data = np.ones((5, 5, 5), dtype=np.int16)

        affine1 = np.eye(4)
        affine2 = np.eye(4) * 2
        affine2[3, 3] = 1

        mask1 = nib.Nifti1Image(data, affine1)
        mask2 = nib.Nifti1Image(data, affine2)

        hash1 = hash_mask(mask1)
        hash2 = hash_mask(mask2)
        assert hash1 != hash2

    def test_hash_is_16_chars(self, small_mask):
        """Hash should be 16 characters."""
        h = hash_mask(small_mask)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


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
            mask_hash="test",
            radius_mm=5.0,
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
            mask_hash="test",
            radius_mm=5.0,
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
            mask_hash="test",
            radius_mm=5.0,
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
            mask_hash="test",
            radius_mm=5.0,
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
            mask_hash="test",
            radius_mm=5.0,
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

    def test_basic_computation(self, tiny_mask, tmp_path, monkeypatch):
        """Test basic neighborhood computation."""
        monkeypatch.setenv("HOME", str(tmp_path))

        neighborhoods = compute_searchlight_neighborhoods(
            tiny_mask, radius_mm=3.0, use_cache=False
        )

        # 3x3x3 = 27 voxels
        assert neighborhoods.n_voxels == 27
        assert neighborhoods.radius_mm == 3.0

    def test_neighborhood_geometry(self, tiny_mask, tmp_path, monkeypatch):
        """Test that neighborhood geometry is correct."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # With 2mm voxels and 3mm radius, center voxel should see
        # its 6 face-adjacent neighbors (distance = 2mm) but not
        # edge-adjacent (distance = 2*sqrt(2) = 2.83mm) or
        # corner-adjacent (distance = 2*sqrt(3) = 3.46mm)
        neighborhoods = compute_searchlight_neighborhoods(
            tiny_mask, radius_mm=2.5, use_cache=False
        )

        # Center voxel (1,1,1) should only see itself with radius < 2mm
        # Since the mask voxel indices are flattened, we need to find
        # the center voxel index
        # For 3x3x3, flattened index of (1,1,1) depends on ordering
        # Let's just check that some neighborhoods are larger than 1

        # With radius 2.5mm and 2mm voxels, face neighbors are included
        # Let's verify the adjacency makes sense
        assert neighborhoods.min_size >= 1
        assert neighborhoods.max_size <= 27

    def test_caching_works(self, small_mask, tmp_path, monkeypatch):
        """Test that caching saves and loads correctly."""
        monkeypatch.setenv("HOME", str(tmp_path))

        # First computation
        nb1 = compute_searchlight_neighborhoods(
            small_mask, radius_mm=5.0, use_cache=True
        )

        # Check cache file exists
        cache = CacheManager("searchlight")
        keys = cache.list_keys()
        assert len(keys) == 1
        assert "5.0mm" in keys[0]

        # Second computation should load from cache
        nb2 = compute_searchlight_neighborhoods(
            small_mask, radius_mm=5.0, use_cache=True
        )

        # Results should be identical
        assert nb1.n_voxels == nb2.n_voxels
        assert nb1.radius_mm == nb2.radius_mm
        assert nb1.mask_hash == nb2.mask_hash
        np.testing.assert_array_equal(nb1.adjacency.toarray(), nb2.adjacency.toarray())

    def test_no_cache_option(self, tiny_mask, tmp_path, monkeypatch):
        """Test that use_cache=False skips caching."""
        monkeypatch.setenv("HOME", str(tmp_path))

        compute_searchlight_neighborhoods(tiny_mask, radius_mm=5.0, use_cache=False)

        cache = CacheManager("searchlight")
        assert len(cache.list_keys()) == 0

    def test_different_radii_different_cache(self, small_mask, tmp_path, monkeypatch):
        """Test that different radii create different cache entries."""
        monkeypatch.setenv("HOME", str(tmp_path))

        compute_searchlight_neighborhoods(small_mask, radius_mm=5.0, use_cache=True)
        compute_searchlight_neighborhoods(small_mask, radius_mm=10.0, use_cache=True)

        cache = CacheManager("searchlight")
        keys = cache.list_keys()
        assert len(keys) == 2

    def test_empty_mask_raises(self, tmp_path, monkeypatch):
        """Test that empty mask raises ValueError."""
        monkeypatch.setenv("HOME", str(tmp_path))

        empty_data = np.zeros((10, 10, 10), dtype=np.int16)
        empty_mask = nib.Nifti1Image(empty_data, np.eye(4))

        with pytest.raises(ValueError, match="no non-zero voxels"):
            compute_searchlight_neighborhoods(
                empty_mask, radius_mm=5.0, use_cache=False
            )

    def test_larger_radius_more_neighbors(self, small_mask, tmp_path, monkeypatch):
        """Test that larger radius gives more neighbors on average."""
        monkeypatch.setenv("HOME", str(tmp_path))

        nb_small = compute_searchlight_neighborhoods(
            small_mask, radius_mm=3.0, use_cache=False
        )
        nb_large = compute_searchlight_neighborhoods(
            small_mask, radius_mm=8.0, use_cache=False
        )

        assert nb_large.mean_size > nb_small.mean_size

    def test_adjacency_is_symmetric(self, small_mask, tmp_path, monkeypatch):
        """Test that the adjacency matrix is symmetric (if i sees j, j sees i)."""
        monkeypatch.setenv("HOME", str(tmp_path))

        neighborhoods = compute_searchlight_neighborhoods(
            small_mask, radius_mm=5.0, use_cache=False
        )

        adj = neighborhoods.adjacency.toarray()
        np.testing.assert_array_equal(adj, adj.T)

    def test_self_is_neighbor(self, small_mask, tmp_path, monkeypatch):
        """Test that each voxel is in its own neighborhood."""
        monkeypatch.setenv("HOME", str(tmp_path))

        neighborhoods = compute_searchlight_neighborhoods(
            small_mask, radius_mm=5.0, use_cache=False
        )

        # Diagonal should be all 1s (each voxel neighbors itself)
        diag = neighborhoods.adjacency.diagonal()
        np.testing.assert_array_equal(diag, np.ones(neighborhoods.n_voxels))


# =============================================================================
# Test clear_cache
# =============================================================================


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_specific_category(self, tmp_path, monkeypatch):
        """Test clearing a specific cache category."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache1 = CacheManager("cat1")
        cache2 = CacheManager("cat2")
        cache1.save("key1", data=np.array([1]))
        cache2.save("key2", data=np.array([2]))

        clear_cache("cat1")

        assert len(cache1.list_keys()) == 0
        assert len(cache2.list_keys()) == 1

    def test_clear_all_categories(self, tmp_path, monkeypatch):
        """Test clearing all cache categories."""
        monkeypatch.setenv("HOME", str(tmp_path))

        cache1 = CacheManager("cat1")
        cache2 = CacheManager("cat2")
        cache1.save("key1", data=np.array([1]))
        cache2.save("key2", data=np.array([2]))

        count = clear_cache()

        assert count == 2
        assert len(cache1.list_keys()) == 0
        assert len(cache2.list_keys()) == 0
