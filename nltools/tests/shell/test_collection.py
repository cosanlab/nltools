"""
Tests for BrainCollection class.

Tests organized by functionality:
- Construction tests
- Property tests
- Indexing tests
- Lazy loading tests
- Construction class methods
"""

import numpy as np
import pandas as pd
import pytest

from nltools.data import BrainData, BrainCollection


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_mask():
    """Create a small sample mask for testing."""
    import nibabel as nib

    # 10x10x10 mask with 500 voxels
    mask_data = np.zeros((10, 10, 10), dtype=np.int8)
    mask_data[2:8, 2:8, 2:8] = 1  # 216 voxels
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    return nib.Nifti1Image(mask_data, affine)


@pytest.fixture
def sample_brain_data(sample_mask):
    """Create a sample BrainData with 20 observations."""
    n_voxels = int(sample_mask.get_fdata().sum())
    data = np.random.randn(20, n_voxels)
    bd = BrainData(mask=sample_mask)
    bd.data = data
    return bd


@pytest.fixture
def sample_brain_data_list(sample_mask):
    """Create a list of 3 BrainData objects with different observation counts."""
    n_voxels = int(sample_mask.get_fdata().sum())
    bds = []
    for n_obs in [20, 20, 20]:
        bd = BrainData(mask=sample_mask)
        bd.data = np.random.randn(n_obs, n_voxels)
        bds.append(bd)
    return bds


@pytest.fixture
def sample_collection(sample_brain_data_list, sample_mask):
    """Create a sample BrainCollection from BrainData objects."""
    return BrainCollection(sample_brain_data_list, mask=sample_mask)


@pytest.fixture
def temp_nifti_files(sample_mask, tmp_path):
    """Create temporary nifti files for testing lazy loading."""
    import nibabel as nib

    paths = []

    for i in range(3):
        # Create 4D data
        data_4d = np.random.randn(10, 10, 10, 20)
        # Apply mask
        data_4d[sample_mask.get_fdata() == 0] = 0
        img = nib.Nifti1Image(data_4d, sample_mask.affine)
        path = tmp_path / f"sub-{i:02d}_bold.nii.gz"
        nib.save(img, path)
        paths.append(path)

    return paths, sample_mask


# =============================================================================
# Construction Tests
# =============================================================================


class TestBrainCollectionConstruction:
    """Tests for BrainCollection construction."""

    def test_init_from_brain_data_list(self, sample_brain_data_list, sample_mask):
        """Test construction from list of BrainData."""
        bc = BrainCollection(sample_brain_data_list, mask=sample_mask)

        assert len(bc) == 3
        assert bc.n_images == 3
        assert all(bc.is_loaded)

    def test_init_from_paths_lazy(self, temp_nifti_files):
        """Test lazy construction from paths."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=True)

        assert len(bc) == 3
        assert not any(bc.is_loaded)

    def test_init_from_paths_eager(self, temp_nifti_files):
        """Test eager construction from paths."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=False)

        assert len(bc) == 3
        assert all(bc.is_loaded)

    def test_init_with_metadata(self, sample_brain_data_list, sample_mask):
        """Test construction with metadata."""
        metadata = pd.DataFrame({
            "subject": ["01", "02", "03"],
            "group": ["control", "patient", "control"],
        })
        bc = BrainCollection(
            sample_brain_data_list, mask=sample_mask, metadata=metadata
        )

        assert len(bc.metadata) == 3
        assert "subject" in bc.metadata.columns
        assert "group" in bc.metadata.columns

    def test_init_empty_raises(self, sample_mask):
        """Test that empty items raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BrainCollection([], mask=sample_mask)

    def test_init_metadata_length_mismatch_raises(
        self, sample_brain_data_list, sample_mask
    ):
        """Test that metadata length mismatch raises error."""
        metadata = pd.DataFrame({"subject": ["01", "02"]})  # Only 2 rows
        with pytest.raises(ValueError, match="metadata length"):
            BrainCollection(
                sample_brain_data_list, mask=sample_mask, metadata=metadata
            )

    def test_init_invalid_item_type_raises(self, sample_mask):
        """Test that invalid item type raises error."""
        with pytest.raises(TypeError, match="Expected path or BrainData"):
            BrainCollection([1, 2, 3], mask=sample_mask)

    def test_init_file_not_found_raises(self, sample_mask):
        """Test that non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            BrainCollection(["/nonexistent/file.nii.gz"], mask=sample_mask)


# =============================================================================
# Property Tests
# =============================================================================


class TestBrainCollectionProperties:
    """Tests for BrainCollection properties."""

    def test_n_images(self, sample_collection):
        """Test n_images property."""
        assert sample_collection.n_images == 3

    def test_n_voxels(self, sample_collection, sample_mask):
        """Test n_voxels property."""
        expected = int(sample_mask.get_fdata().sum())
        assert sample_collection.n_voxels == expected

    def test_shape_uniform_obs(self, sample_collection):
        """Test shape with uniform observation counts."""
        shape = sample_collection.shape
        assert shape[0] == 3  # n_images
        assert shape[1] == 20  # n_observations (uniform)
        assert shape[2] == sample_collection.n_voxels

    def test_shape_variable_obs(self, sample_mask):
        """Test shape with variable observation counts."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = []
        for n_obs in [10, 20, 30]:  # Different counts
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)
        shape = bc.shape
        assert shape[0] == 3
        assert shape[1] is None  # Variable, so None
        assert shape[2] == n_voxels

    def test_mask(self, sample_collection, sample_mask):
        """Test mask property."""
        assert sample_collection.mask.shape == sample_mask.shape

    def test_metadata_default(self, sample_collection):
        """Test default metadata is empty DataFrame with correct length."""
        assert len(sample_collection.metadata) == 3

    def test_is_loaded(self, temp_nifti_files):
        """Test is_loaded property."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=True)

        assert bc.is_loaded == [False, False, False]

        # Access first item
        _ = bc[0]
        assert bc.is_loaded == [True, False, False]


# =============================================================================
# Indexing Tests
# =============================================================================


class TestBrainCollectionIndexing:
    """Tests for BrainCollection indexing."""

    def test_integer_index(self, sample_collection):
        """Test integer indexing returns BrainData."""
        bd = sample_collection[0]
        assert isinstance(bd, BrainData)
        assert bd.shape == (20, sample_collection.n_voxels)

    def test_negative_index(self, sample_collection):
        """Test negative indexing."""
        bd = sample_collection[-1]
        assert isinstance(bd, BrainData)

    def test_slice_index(self, sample_collection):
        """Test slice indexing returns BrainCollection."""
        bc_subset = sample_collection[0:2]
        assert isinstance(bc_subset, BrainCollection)
        assert len(bc_subset) == 2

    def test_list_index(self, sample_collection):
        """Test list indexing returns BrainCollection."""
        bc_subset = sample_collection[[0, 2]]
        assert isinstance(bc_subset, BrainCollection)
        assert len(bc_subset) == 2

    def test_multidim_single_obs(self, sample_collection):
        """Test bc[i, j] returns single observation."""
        bd = sample_collection[0, 5]
        assert isinstance(bd, BrainData)
        assert bd.data.ndim == 1  # Single observation

    def test_multidim_obs_slice(self, sample_collection):
        """Test bc[i, j:k] returns sliced observations."""
        bd = sample_collection[0, 5:10]
        assert isinstance(bd, BrainData)
        assert bd.shape == (5, sample_collection.n_voxels)

    def test_multidim_all_images_single_obs(self, sample_collection):
        """Test bc[:, j] returns all images at single observation."""
        bc_subset = sample_collection[:, 0]
        assert isinstance(bc_subset, BrainCollection)
        assert len(bc_subset) == 3

    def test_multidim_all_images_obs_slice(self, sample_collection):
        """Test bc[:, j:k] returns all images with sliced observations."""
        bc_subset = sample_collection[:, 5:10]
        assert isinstance(bc_subset, BrainCollection)
        assert len(bc_subset) == 3
        # Check each item has 5 observations
        for bd in bc_subset.to_list():
            assert bd.shape == (5, sample_collection.n_voxels)

    def test_string_index_by_subject(self, sample_brain_data_list, sample_mask):
        """Test string indexing by subject metadata."""
        metadata = pd.DataFrame({"subject": ["01", "02", "03"]})
        bc = BrainCollection(
            sample_brain_data_list, mask=sample_mask, metadata=metadata
        )

        bd = bc["01"]
        assert isinstance(bd, BrainData)

    def test_string_index_not_found_raises(self, sample_collection):
        """Test string indexing raises KeyError when not found."""
        with pytest.raises(KeyError):
            _ = sample_collection["nonexistent"]


# =============================================================================
# Lazy Loading Tests
# =============================================================================


class TestBrainCollectionLazyLoading:
    """Tests for lazy loading functionality."""

    def test_lazy_load_on_access(self, temp_nifti_files):
        """Test that lazy items are loaded on access."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=True)

        assert not any(bc.is_loaded)
        _ = bc[0]
        assert bc.is_loaded[0]
        assert not bc.is_loaded[1]

    def test_load_all(self, temp_nifti_files):
        """Test load() loads all items."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=True)

        bc.load()
        assert all(bc.is_loaded)

    def test_load_specific_indices(self, temp_nifti_files):
        """Test load(indices) loads specific items."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=True)

        bc.load(indices=[0, 2])
        assert bc.is_loaded == [True, False, True]

    def test_memory_estimate(self, sample_collection):
        """Test memory_estimate returns reasonable string."""
        estimate = sample_collection.memory_estimate()
        assert "total" in estimate
        assert "per image" in estimate


# =============================================================================
# Construction Class Methods Tests
# =============================================================================


class TestBrainCollectionClassMethods:
    """Tests for BrainCollection class construction methods."""

    def test_from_glob(self, temp_nifti_files):
        """Test from_glob construction."""
        paths, mask = temp_nifti_files
        pattern = str(paths[0].parent / "sub-*_bold.nii.gz")

        bc = BrainCollection.from_glob(pattern, mask=mask)
        assert len(bc) == 3

    def test_from_glob_with_pattern_groups(self, temp_nifti_files):
        """Test from_glob with regex pattern for metadata."""
        paths, mask = temp_nifti_files
        pattern = str(paths[0].parent / "sub-*_bold.nii.gz")

        bc = BrainCollection.from_glob(
            pattern,
            mask=mask,
            pattern_groups=r"sub-(?P<subject>\d+)",
        )
        assert "subject" in bc.metadata.columns
        assert list(bc.metadata["subject"]) == ["00", "01", "02"]

    def test_from_glob_no_matches_raises(self, sample_mask, tmp_path):
        """Test from_glob raises when no files match."""
        with pytest.raises(ValueError, match="No files found"):
            BrainCollection.from_glob(
                str(tmp_path / "nonexistent_*.nii.gz"), mask=sample_mask
            )

    def test_from_stacked_n_images(self, sample_brain_data, sample_mask):
        """Test from_stacked with n_images."""
        # 20 observations split into 2 images of 10 each
        bc = BrainCollection.from_stacked(sample_brain_data, n_images=2)

        assert len(bc) == 2
        assert bc[0].shape == (10, sample_brain_data.data.shape[1])

    def test_from_stacked_splits(self, sample_brain_data, sample_mask):
        """Test from_stacked with explicit splits."""
        # 20 observations split as 5, 7, 8
        bc = BrainCollection.from_stacked(sample_brain_data, splits=[5, 7, 8])

        assert len(bc) == 3
        assert bc[0].shape[0] == 5
        assert bc[1].shape[0] == 7
        assert bc[2].shape[0] == 8

    def test_from_stacked_uneven_raises(self, sample_brain_data):
        """Test from_stacked raises when can't split evenly."""
        with pytest.raises(ValueError, match="Cannot evenly split"):
            BrainCollection.from_stacked(sample_brain_data, n_images=3)

    def test_from_stacked_splits_sum_mismatch_raises(self, sample_brain_data):
        """Test from_stacked raises when splits don't sum to total."""
        with pytest.raises(ValueError, match="splits sum"):
            BrainCollection.from_stacked(sample_brain_data, splits=[5, 5])


# =============================================================================
# Iteration Tests
# =============================================================================


class TestBrainCollectionIteration:
    """Tests for BrainCollection iteration."""

    def test_iter(self, sample_collection):
        """Test iteration yields BrainData objects."""
        items = list(sample_collection)
        assert len(items) == 3
        assert all(isinstance(item, BrainData) for item in items)

    def test_len(self, sample_collection):
        """Test __len__."""
        assert len(sample_collection) == 3


# =============================================================================
# Repr Tests
# =============================================================================


class TestBrainCollectionRepr:
    """Tests for BrainCollection string representation."""

    def test_repr(self, sample_collection):
        """Test __repr__ returns informative string."""
        r = repr(sample_collection)
        assert "BrainCollection" in r
        assert "shape=" in r
        assert "loaded=" in r


# =============================================================================
# Conversion Tests
# =============================================================================


class TestBrainCollectionConversion:
    """Tests for BrainCollection conversion methods."""

    def test_to_tensor_shape(self, sample_collection):
        """Test to_tensor returns correct shape."""
        tensor = sample_collection.to_tensor()
        assert tensor.shape == (3, 20, sample_collection.n_voxels)

    def test_to_tensor_values(self, sample_collection):
        """Test to_tensor preserves values."""
        tensor = sample_collection.to_tensor()
        # Compare with direct access
        for i in range(len(sample_collection)):
            bd = sample_collection[i]
            np.testing.assert_array_equal(tensor[i], bd.data)

    def test_to_tensor_variable_obs_raises(self, sample_mask):
        """Test to_tensor raises with variable observation counts."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = []
        for n_obs in [10, 20, 30]:  # Different counts
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)
        with pytest.raises(ValueError, match="variable observation counts"):
            bc.to_tensor()

    def test_to_tensor_batched(self, sample_collection):
        """Test to_tensor with batch_size returns generator."""
        batches = list(sample_collection.to_tensor(batch_size=2))
        assert len(batches) == 2  # 3 images / 2 per batch = 2 batches
        assert batches[0].shape == (2, 20, sample_collection.n_voxels)
        assert batches[1].shape == (1, 20, sample_collection.n_voxels)

    def test_to_list(self, sample_collection):
        """Test to_list returns list of BrainData."""
        bd_list = sample_collection.to_list()
        assert len(bd_list) == 3
        assert all(isinstance(bd, BrainData) for bd in bd_list)

    def test_to_stacked_shape(self, sample_collection):
        """Test to_stacked returns correct shape."""
        stacked = sample_collection.to_stacked()
        assert isinstance(stacked, BrainData)
        # 3 images * 20 obs = 60 total observations
        assert stacked.shape == (60, sample_collection.n_voxels)

    def test_to_stacked_values(self, sample_collection):
        """Test to_stacked preserves values."""
        stacked = sample_collection.to_stacked()
        # Check first image's data
        bd0 = sample_collection[0]
        np.testing.assert_array_equal(stacked.data[:20], bd0.data)


# =============================================================================
# Iter Batches Tests
# =============================================================================


class TestBrainCollectionIterBatches:
    """Tests for BrainCollection iter_batches method."""

    def test_iter_batches_axis0(self, sample_collection):
        """Test iter_batches over images (axis=0)."""
        batches = list(sample_collection.iter_batches(batch_size=2, show_progress=False))
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_iter_batches_axis1(self, sample_collection):
        """Test iter_batches over observations (axis=1)."""
        batches = list(
            sample_collection.iter_batches(batch_size=5, axis=1, show_progress=False)
        )
        assert len(batches) == 4  # 20 obs / 5 per batch = 4 batches
        # Each batch should have all 3 images with 5 obs each
        for batch in batches[:-1]:
            assert len(batch) == 3
            for bd in batch.to_list():
                assert bd.shape[0] == 5

    def test_iter_batches_named_axis(self, sample_collection):
        """Test iter_batches with named axis."""
        batches = list(
            sample_collection.iter_batches(
                batch_size=2, axis="images", show_progress=False
            )
        )
        assert len(batches) == 2

    def test_iter_batches_variable_obs_axis1_raises(self, sample_mask):
        """Test iter_batches axis=1 raises with variable obs counts."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = []
        for n_obs in [10, 20, 30]:
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)
        with pytest.raises(ValueError, match="variable counts"):
            list(bc.iter_batches(batch_size=5, axis=1, show_progress=False))


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestBrainCollectionAggregation:
    """Tests for BrainCollection axis aggregation operations."""

    def test_mean_axis0_shape(self, sample_collection):
        """Test mean(axis=0) returns correct shape."""
        result = sample_collection.mean(axis=0)
        assert isinstance(result, BrainData)
        # (n_obs, n_voxels)
        assert result.shape == (20, sample_collection.n_voxels)

    def test_mean_axis0_values(self, sample_collection):
        """Test mean(axis=0) computes correct values."""
        result = sample_collection.mean(axis=0)
        # Compare with numpy
        tensor = sample_collection.to_tensor()
        expected = tensor.mean(axis=0)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_mean_axis1_shape(self, sample_collection):
        """Test mean(axis=1) returns correct shape."""
        result = sample_collection.mean(axis=1)
        assert isinstance(result, BrainCollection)
        assert len(result) == 3
        # Each item should have shape (n_voxels,)
        for bd in result.to_list():
            assert bd.data.ndim == 1
            assert bd.shape == (sample_collection.n_voxels,)

    def test_mean_axis1_values(self, sample_collection):
        """Test mean(axis=1) computes correct values."""
        result = sample_collection.mean(axis=1)
        # Compare with numpy
        tensor = sample_collection.to_tensor()
        expected = tensor.mean(axis=1)
        for i, bd in enumerate(result.to_list()):
            np.testing.assert_array_almost_equal(bd.data, expected[i])

    def test_mean_axis2_shape(self, sample_collection):
        """Test mean(axis=2) returns correct shape."""
        result = sample_collection.mean(axis=2)
        assert isinstance(result, np.ndarray)
        # (n_images, n_obs)
        assert result.shape == (3, 20)

    def test_mean_axis2_values(self, sample_collection):
        """Test mean(axis=2) computes correct values."""
        result = sample_collection.mean(axis=2)
        tensor = sample_collection.to_tensor()
        expected = tensor.mean(axis=2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mean_named_axis(self, sample_collection):
        """Test mean with named axis."""
        result_int = sample_collection.mean(axis=0)
        result_name = sample_collection.mean(axis="images")
        np.testing.assert_array_equal(result_int.data, result_name.data)

    def test_std_axis0(self, sample_collection):
        """Test std(axis=0) computes correct values."""
        result = sample_collection.std(axis=0)
        tensor = sample_collection.to_tensor()
        expected = tensor.std(axis=0, ddof=1)  # Sample std
        np.testing.assert_array_almost_equal(result.data, expected, decimal=5)

    def test_sum_axis0(self, sample_collection):
        """Test sum(axis=0) computes correct values."""
        result = sample_collection.sum(axis=0)
        tensor = sample_collection.to_tensor()
        expected = tensor.sum(axis=0)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_min_axis0(self, sample_collection):
        """Test min(axis=0) computes correct values."""
        result = sample_collection.min(axis=0)
        tensor = sample_collection.to_tensor()
        expected = tensor.min(axis=0)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_max_axis0(self, sample_collection):
        """Test max(axis=0) computes correct values."""
        result = sample_collection.max(axis=0)
        tensor = sample_collection.to_tensor()
        expected = tensor.max(axis=0)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_median_axis0(self, sample_collection):
        """Test median(axis=0) computes correct values."""
        result = sample_collection.median(axis=0)
        tensor = sample_collection.to_tensor()
        expected = np.median(tensor, axis=0)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_mean_variable_obs_raises(self, sample_mask):
        """Test mean(axis=0) raises with variable observation counts."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = []
        for n_obs in [10, 20, 30]:
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)
        with pytest.raises(ValueError, match="variable observation counts"):
            bc.mean(axis=0)
