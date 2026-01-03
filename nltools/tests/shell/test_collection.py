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
import nibabel as nib
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

    @pytest.mark.slow
    def test_init_from_paths_eager(self, temp_nifti_files):
        """Test eager construction from paths."""
        paths, mask = temp_nifti_files
        bc = BrainCollection(paths, mask=mask, lazy=False)

        assert len(bc) == 3
        assert all(bc.is_loaded)

    def test_init_with_metadata(self, sample_brain_data_list, sample_mask):
        """Test construction with metadata."""
        metadata = pd.DataFrame(
            {
                "subject": ["01", "02", "03"],
                "group": ["control", "patient", "control"],
            }
        )
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
            BrainCollection(sample_brain_data_list, mask=sample_mask, metadata=metadata)

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
        batches = list(
            sample_collection.iter_batches(batch_size=2, show_progress=False)
        )
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    @pytest.mark.slow
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


class TestBrainCollectionInference:
    """Tests for group inference methods."""

    def test_ttest_shape(self, sample_collection):
        """Test ttest returns correct shapes."""
        t_stat, p_val = sample_collection.ttest()

        # Should match (n_obs, n_voxels)
        tensor = sample_collection.to_tensor()
        expected_shape = tensor.shape[1:]  # (n_obs, n_voxels)

        assert t_stat.data.shape == expected_shape
        assert p_val.data.shape == expected_shape

    def test_ttest_matches_scipy(self, sample_collection):
        """Test ttest matches scipy.stats.ttest_1samp."""
        from scipy import stats

        t_stat, p_val = sample_collection.ttest(popmean=0)

        tensor = sample_collection.to_tensor()
        expected_t, expected_p = stats.ttest_1samp(tensor, 0, axis=0)

        np.testing.assert_array_almost_equal(t_stat.data, expected_t)
        np.testing.assert_array_almost_equal(p_val.data, expected_p)

    def test_ttest_with_popmean(self, sample_collection):
        """Test ttest with non-zero population mean."""
        from scipy import stats

        popmean = 0.5
        t_stat, p_val = sample_collection.ttest(popmean=popmean)

        tensor = sample_collection.to_tensor()
        expected_t, expected_p = stats.ttest_1samp(tensor, popmean, axis=0)

        np.testing.assert_array_almost_equal(t_stat.data, expected_t)
        np.testing.assert_array_almost_equal(p_val.data, expected_p)

    def test_ttest_axis_not_zero_raises(self, sample_collection):
        """Test ttest raises for axis != 0."""
        with pytest.raises(ValueError, match="only supports axis=0"):
            sample_collection.ttest(axis=1)

    def test_ttest2_shape(self, sample_brain_data_list, sample_mask):
        """Test ttest2 returns correct shapes."""
        bc1 = BrainCollection(sample_brain_data_list[:2], mask=sample_mask)
        bc2 = BrainCollection(sample_brain_data_list[1:], mask=sample_mask)

        t_stat, p_val = bc1.ttest2(bc2)

        tensor1 = bc1.to_tensor()
        expected_shape = tensor1.shape[1:]

        assert t_stat.data.shape == expected_shape
        assert p_val.data.shape == expected_shape

    def test_ttest2_matches_scipy(self, sample_brain_data_list, sample_mask):
        """Test ttest2 matches scipy.stats.ttest_ind."""
        from scipy import stats

        bc1 = BrainCollection(sample_brain_data_list[:2], mask=sample_mask)
        bc2 = BrainCollection(sample_brain_data_list[1:], mask=sample_mask)

        t_stat, p_val = bc1.ttest2(bc2)

        tensor1 = bc1.to_tensor()
        tensor2 = bc2.to_tensor()
        expected_t, expected_p = stats.ttest_ind(tensor1, tensor2, axis=0)

        np.testing.assert_array_almost_equal(t_stat.data, expected_t)
        np.testing.assert_array_almost_equal(p_val.data, expected_p)

    def test_ttest2_welch(self, sample_brain_data_list, sample_mask):
        """Test ttest2 with Welch's t-test."""
        from scipy import stats

        bc1 = BrainCollection(sample_brain_data_list[:2], mask=sample_mask)
        bc2 = BrainCollection(sample_brain_data_list[1:], mask=sample_mask)

        t_stat, p_val = bc1.ttest2(bc2, equal_var=False)

        tensor1 = bc1.to_tensor()
        tensor2 = bc2.to_tensor()
        expected_t, expected_p = stats.ttest_ind(
            tensor1, tensor2, axis=0, equal_var=False
        )

        np.testing.assert_array_almost_equal(t_stat.data, expected_t)
        np.testing.assert_array_almost_equal(p_val.data, expected_p)

    def test_ttest2_mask_mismatch_raises(self, sample_brain_data_list, sample_mask):
        """Test ttest2 raises for mismatched masks."""
        bc1 = BrainCollection(sample_brain_data_list, mask=sample_mask)

        # Create collection with different mask
        different_mask = np.zeros((5, 5, 5), dtype=np.int8)
        different_mask[1:4, 1:4, 1:4] = 1
        different_mask_nii = nib.Nifti1Image(different_mask, np.eye(4))

        n_voxels2 = int(different_mask.sum())
        bd = BrainData(mask=different_mask_nii)
        bd.data = np.random.randn(5, n_voxels2)
        bc2 = BrainCollection([bd], mask=different_mask_nii)

        with pytest.raises(ValueError, match="same mask shape"):
            bc1.ttest2(bc2)

    @pytest.mark.slow
    def test_anova_from_list(self, sample_mask):
        """Test ANOVA with group labels as list."""
        from scipy import stats

        n_voxels = int(sample_mask.get_fdata().sum())
        n_images = 9
        n_obs = 1

        # Create 3 groups of 3 images each
        bds = []
        for i in range(n_images):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels) + i // 3  # Group mean offset
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)
        groups = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]

        f_stat, p_val = bc.anova(groups)

        # Verify shapes
        assert f_stat.data.shape == (n_voxels,)
        assert p_val.data.shape == (n_voxels,)

        # Verify against scipy
        tensor = bc.to_tensor().squeeze(axis=1)  # (n_images, n_voxels)
        group_a = tensor[0:3]
        group_b = tensor[3:6]
        group_c = tensor[6:9]
        expected_f, expected_p = stats.f_oneway(group_a, group_b, group_c)

        np.testing.assert_array_almost_equal(f_stat.data, expected_f)
        np.testing.assert_array_almost_equal(p_val.data, expected_p)

    @pytest.mark.slow
    def test_anova_from_metadata(self, sample_mask):
        """Test ANOVA with group from metadata column."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_images = 6
        n_obs = 1

        bds = []
        for _ in range(n_images):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        metadata = pd.DataFrame({"condition": ["A", "A", "B", "B", "C", "C"]})
        bc = BrainCollection(bds, mask=sample_mask, metadata=metadata)

        f_stat, p_val = bc.anova("condition")

        assert f_stat.data.shape == (n_voxels,)
        assert p_val.data.shape == (n_voxels,)

    def test_anova_missing_column_raises(self, sample_collection):
        """Test ANOVA raises for missing metadata column."""
        with pytest.raises(KeyError, match="not found in metadata"):
            sample_collection.anova("nonexistent_column")

    def test_anova_wrong_length_raises(self, sample_collection):
        """Test ANOVA raises for wrong group length."""
        wrong_length = ["A", "B"]  # Only 2 elements, but 3 images
        with pytest.raises(ValueError, match="must match"):
            sample_collection.anova(wrong_length)

    def test_anova_one_group_raises(self, sample_collection):
        """Test ANOVA raises for single group."""
        groups = ["A"] * sample_collection.n_images
        with pytest.raises(ValueError, match="at least 2 groups"):
            sample_collection.anova(groups)

    @pytest.mark.slow
    def test_permutation_test_shape(self, sample_mask):
        """Test permutation_test returns correct shapes."""
        n_voxels = int(sample_mask.get_fdata().sum())

        # Create collection with single observation for speed
        bds = []
        for _ in range(5):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(1, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Use small n_permute for speed
        result = bc.permutation_test(n_permute=10, parallel=None)

        assert "mean" in result
        assert "p" in result
        assert result["mean"].data.shape == (n_voxels,)
        assert result["p"].data.shape == (n_voxels,)

    @pytest.mark.slow
    def test_permutation_test_return_null(self, sample_mask):
        """Test permutation_test with return_null=True."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_permute = 10

        bds = []
        for _ in range(5):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(1, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.permutation_test(
            n_permute=n_permute, parallel=None, return_null=True
        )

        assert "null_dist" in result
        # Shape: (n_obs, n_permute, n_voxels)
        assert result["null_dist"].shape[1] == n_permute

    @pytest.mark.slow
    def test_permutation_test2_shape(self, sample_mask):
        """Test permutation_test2 returns correct shapes."""
        n_voxels = int(sample_mask.get_fdata().sum())

        # Create two collections
        bds1 = []
        bds2 = []
        for _ in range(3):
            bd1 = BrainData(mask=sample_mask)
            bd1.data = np.random.randn(1, n_voxels)
            bds1.append(bd1)

            bd2 = BrainData(mask=sample_mask)
            bd2.data = np.random.randn(1, n_voxels)
            bds2.append(bd2)

        bc1 = BrainCollection(bds1, mask=sample_mask)
        bc2 = BrainCollection(bds2, mask=sample_mask)

        result = bc1.permutation_test2(bc2, n_permute=10, parallel=None)

        assert "mean_diff" in result
        assert "p" in result
        assert result["mean_diff"].data.shape == (n_voxels,)
        assert result["p"].data.shape == (n_voxels,)


class TestBrainCollectionTransformations:
    """Tests for map, filter, and convenience methods."""

    def test_map_axis0_basic(self, sample_collection):
        """Test map over images (axis=0)."""

        # Simple identity-like transform
        def double_data(bd):
            result = BrainData(mask=bd.mask)
            result.data = bd.data * 2
            return result

        result = sample_collection.map(double_data, axis=0, show_progress=False)

        assert isinstance(result, BrainCollection)
        assert result.n_images == sample_collection.n_images

        # Check data was doubled
        orig = sample_collection[0].data
        transformed = result[0].data
        np.testing.assert_array_almost_equal(transformed, orig * 2)

    def test_map_axis0_preserves_metadata(self, sample_mask):
        """Test that map preserves metadata."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = [BrainData(mask=sample_mask) for _ in range(3)]
        for bd in bds:
            bd.data = np.random.randn(5, n_voxels)

        metadata = pd.DataFrame({"subject": ["A", "B", "C"]})
        bc = BrainCollection(bds, mask=sample_mask, metadata=metadata)

        result = bc.map(lambda bd: bd, axis=0, show_progress=False)

        assert list(result.metadata["subject"]) == ["A", "B", "C"]

    def test_map_axis2_per_voxel(self, sample_mask):
        """Test map over voxels (axis=2) applies function to each timeseries."""
        from scipy.signal import detrend

        n_voxels = int(sample_mask.get_fdata().sum())
        n_obs = 10

        bd = BrainData(mask=sample_mask)
        # Add linear trend to each voxel
        trend = np.linspace(0, 1, n_obs)[:, np.newaxis]
        bd.data = np.random.randn(n_obs, n_voxels) + trend

        bc = BrainCollection([bd], mask=sample_mask)

        result = bc.map(detrend, axis=2, show_progress=False)

        # Detrended data should have near-zero mean per voxel
        assert result.n_images == 1
        detrended = result[0].data
        col_means = np.abs(detrended.mean(axis=0))
        assert np.all(col_means < 0.1)  # Should be close to 0

    def test_map_axis0_returns_braindata(self, sample_collection):
        """Test that map axis=0 function must return BrainData."""
        result = sample_collection.map(lambda bd: bd, axis=0, show_progress=False)
        assert all(isinstance(result[i], BrainData) for i in range(result.n_images))

    def test_filter_callable(self, sample_mask):
        """Test filter with callable predicate."""
        n_voxels = int(sample_mask.get_fdata().sum())

        bds = []
        for i in range(5):
            bd = BrainData(mask=sample_mask)
            bd.data = np.ones((3, n_voxels)) * (i - 2)  # Means: -2, -1, 0, 1, 2
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Filter to positive mean images
        result = bc.filter(lambda bd: bd.data.mean() > 0)

        assert result.n_images == 2  # Only images with mean > 0

    def test_filter_boolean_array(self, sample_collection):
        """Test filter with boolean array."""
        mask = [True, False, True]
        result = sample_collection.filter(mask)

        assert result.n_images == 2

    def test_filter_series(self, sample_mask):
        """Test filter with pandas Series."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bds = [BrainData(mask=sample_mask) for _ in range(3)]
        for bd in bds:
            bd.data = np.random.randn(5, n_voxels)

        metadata = pd.DataFrame({"group": ["A", "B", "A"]})
        bc = BrainCollection(bds, mask=sample_mask, metadata=metadata)

        result = bc.filter(bc.metadata["group"] == "A")

        assert result.n_images == 2

    def test_filter_length_mismatch_raises(self, sample_collection):
        """Test filter raises for wrong predicate length."""
        with pytest.raises(ValueError, match="must match"):
            sample_collection.filter([True, False])  # Wrong length

    def test_standardize_zscore(self, sample_collection):
        """Test standardize with zscore method."""
        result = sample_collection.standardize(method="zscore", show_progress=False)

        # Each image should be z-scored: mean ~0, std ~1 per voxel
        for i in range(result.n_images):
            data = result[i].data
            # Check mean is close to 0 for each voxel
            assert np.allclose(data.mean(axis=0), 0, atol=1e-10)
            # Check std is close to 1 for each voxel
            assert np.allclose(data.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_standardize_axis1(self, sample_mask):
        """Test standardize across voxels (axis=1)."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bd = BrainData(mask=sample_mask)
        bd.data = np.random.randn(5, n_voxels) + 10  # Offset mean

        bc = BrainCollection([bd], mask=sample_mask)

        result = bc.standardize(axis=1, method="center", show_progress=False)

        # Each observation should have mean ~0 across voxels
        data = result[0].data
        for obs in range(data.shape[0]):
            assert np.abs(data[obs].mean()) < 1e-10

    def test_threshold_upper(self, sample_mask):
        """Test threshold with upper bound (zeros values below upper)."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bd = BrainData(mask=sample_mask)
        bd.data = np.linspace(-1, 1, n_voxels)

        bc = BrainCollection([bd], mask=sample_mask)

        # upper=0 zeros values below 0, keeping positive values
        result = bc.threshold(upper=0, show_progress=False)

        data = result[0].data
        # Non-zero values should all be >= 0
        assert np.all(data[data != 0] >= 0)

    def test_threshold_binarize(self, sample_mask):
        """Test threshold with binarize."""
        n_voxels = int(sample_mask.get_fdata().sum())
        bd = BrainData(mask=sample_mask)
        bd.data = np.linspace(-1, 1, n_voxels)

        bc = BrainCollection([bd], mask=sample_mask)

        # upper=0 zeros values below 0, binarize makes remaining 1
        result = bc.threshold(upper=0, binarize=True, show_progress=False)

        # Should be binary
        data = result[0].data
        unique_vals = np.unique(data)
        assert len(unique_vals) <= 2
        assert all(v in [0, 1] for v in unique_vals)

    def test_detrend(self, sample_mask):
        """Test detrend removes linear trend."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_obs = 20

        bd = BrainData(mask=sample_mask)
        # Create data with strong linear trend
        trend = np.linspace(0, 10, n_obs)[:, np.newaxis]
        bd.data = np.random.randn(n_obs, n_voxels) * 0.1 + trend

        bc = BrainCollection([bd], mask=sample_mask)

        result = bc.detrend(show_progress=False)

        # Trend should be removed
        data = result[0].data
        # Check that correlation with linear trend is low
        trend_flat = np.linspace(0, 1, n_obs)
        for v in range(min(10, n_voxels)):  # Check first 10 voxels
            corr = np.corrcoef(data[:, v], trend_flat)[0, 1]
            assert np.abs(corr) < 0.3  # Low correlation

    def test_chaining(self, sample_mask):
        """Test method chaining works."""
        n_voxels = int(sample_mask.get_fdata().sum())

        bds = []
        for i in range(4):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(10, n_voxels) + i  # Different means
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Chain: filter -> standardize -> aggregate
        result = (
            bc.filter(lambda bd: bd.data.mean() > 0.5)
            .standardize(method="zscore", show_progress=False)
            .mean(axis=0)
        )

        assert isinstance(result, BrainData)
        # Should have fewer than 4 images in filtered collection
        # Result is mean across filtered images


# =============================================================================
# ISC Tests
# =============================================================================


class TestBrainCollectionISC:
    """Tests for ISC methods on BrainCollection."""

    def test_extract_voxelwise_shape(self, sample_mask):
        """Test voxelwise extraction returns correct shape."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 20

        # Create collection with uniform observations
        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Extract voxelwise
        extracted, info = bc._extract_for_isc(
            roi_mask=None, radius=None, show_progress=False
        )

        assert extracted.shape == (n_obs, n_subjects, n_voxels)
        assert info["mode"] == "voxelwise"
        assert info["n_features"] == n_voxels

    def test_extract_searchlight_shape(self, sample_mask):
        """Test searchlight extraction returns correct shape."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 3
        n_obs = 10

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Extract with searchlight
        extracted, info = bc._extract_for_isc(
            roi_mask=None, radius=5.0, show_progress=False
        )

        assert extracted.shape == (n_obs, n_subjects, n_voxels)
        assert info["mode"] == "searchlight"
        assert info["radius"] == 5.0
        assert "neighborhoods" in info

    def test_project_to_brain_voxelwise(self, sample_mask):
        """Test projecting voxelwise values back to brain."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 3
        n_obs = 10

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        # Extract
        _, info = bc._extract_for_isc(roi_mask=None, radius=None, show_progress=False)

        # Project random values
        values = np.random.randn(n_voxels)
        result = bc._project_to_brain(values, info)

        assert isinstance(result, BrainData)
        np.testing.assert_array_equal(result.data, values)

    @pytest.mark.slow
    def test_isc_voxelwise_shape(self, sample_mask):
        """Test ISC computation returns BrainData with correct shape."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 20

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc(method="loo", radius=None, show_progress=False)

        assert "isc" in result
        assert isinstance(result["isc"], BrainData)
        assert result["isc"].data.shape == (n_voxels,)
        assert result["method"] == "loo"
        assert result["extraction"] == "voxelwise"
        assert result["n_subjects"] == n_subjects

    def test_isc_pairwise_method(self, sample_mask):
        """Test pairwise ISC method."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 15

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc(method="pairwise", radius=None, show_progress=False)

        assert result["method"] == "pairwise"
        assert isinstance(result["isc"], BrainData)

    @pytest.mark.slow
    def test_isc_correlated_subjects(self, sample_mask):
        """Test ISC is high when subjects are correlated."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 50

        # Create shared signal
        shared_signal = np.random.randn(n_obs, n_voxels)

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            # Each subject = shared signal + small noise
            bd.data = shared_signal + 0.1 * np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc(method="loo", radius=None, show_progress=False)

        # ISC should be high (>0.8) for correlated subjects
        assert result["isc"].data.mean() > 0.8

    def test_isc_uncorrelated_subjects(self, sample_mask):
        """Test ISC is near zero for uncorrelated subjects."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 50

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            # Each subject = independent random
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc(method="loo", radius=None, show_progress=False)

        # ISC should be near zero for uncorrelated subjects
        assert np.abs(result["isc"].data.mean()) < 0.3

    def test_isc_variable_obs_raises(self, sample_mask):
        """Test that variable observation counts raise an error."""
        n_voxels = int(sample_mask.get_fdata().sum())

        bds = []
        for i in range(3):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(10 + i, n_voxels)  # Different lengths
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        with pytest.raises(ValueError, match="uniform observation counts"):
            bc.isc(radius=None, show_progress=False)

    @pytest.mark.slow
    def test_isc_mean_metric(self, sample_mask):
        """Test ISC with mean (Fisher z) metric."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 20

        shared = np.random.randn(n_obs, n_voxels)
        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = shared + 0.1 * np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result_median = bc.isc(metric="median", radius=None, show_progress=False)
        result_mean = bc.isc(metric="mean", radius=None, show_progress=False)

        # Both should give high ISC for correlated data
        assert result_median["isc"].data.mean() > 0.7
        assert result_mean["isc"].data.mean() > 0.7

    @pytest.mark.slow
    def test_isc_test_returns_correct_keys(self, sample_mask):
        """Test isc_test returns all expected keys."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 20

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc_test(
            radius=None, n_permute=100, show_progress=False, random_state=42
        )

        # Check all expected keys
        assert "isc" in result
        assert "p" in result
        assert "ci" in result
        assert "method" in result
        assert "permutation_method" in result
        assert "extraction" in result
        assert "n_subjects" in result
        assert "n_permute" in result

        # Check types
        assert isinstance(result["isc"], BrainData)
        assert isinstance(result["p"], BrainData)
        assert isinstance(result["ci"], tuple)
        assert len(result["ci"]) == 2
        assert isinstance(result["ci"][0], BrainData)
        assert isinstance(result["ci"][1], BrainData)

    @pytest.mark.slow
    def test_isc_test_correlated_significant(self, sample_mask):
        """Test that correlated subjects produce significant ISC."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 50

        # Create shared signal
        shared_signal = np.random.randn(n_obs, n_voxels)

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = shared_signal + 0.1 * np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc_test(
            radius=None, n_permute=500, show_progress=False, random_state=42
        )

        # Most p-values should be significant for highly correlated data
        sig_voxels = (result["p"].data < 0.05).sum()
        total_voxels = n_voxels
        assert sig_voxels / total_voxels > 0.9  # >90% significant

    @pytest.mark.slow
    def test_isc_test_uncorrelated_not_significant(self, sample_mask):
        """Test that uncorrelated subjects produce non-significant ISC."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 5
        n_obs = 50

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc_test(
            radius=None, n_permute=500, show_progress=False, random_state=42
        )

        # Very few p-values should be significant for uncorrelated data
        # (expect ~5% by chance at alpha=0.05)
        sig_voxels = (result["p"].data < 0.05).sum()
        total_voxels = n_voxels
        assert sig_voxels / total_voxels < 0.15  # <15% (allow some slack)

    @pytest.mark.slow
    def test_isc_test_permutation_methods(self, sample_mask):
        """Test that all permutation methods run without error."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 20

        shared = np.random.randn(n_obs, n_voxels)
        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = shared + 0.2 * np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        for method in ["bootstrap", "circle_shift", "phase_randomize"]:
            result = bc.isc_test(
                radius=None,
                n_permute=50,
                permutation_method=method,
                show_progress=False,
                random_state=42,
            )
            assert result["permutation_method"] == method
            assert isinstance(result["isc"], BrainData)
            assert isinstance(result["p"], BrainData)

    @pytest.mark.slow
    def test_isc_test_return_null(self, sample_mask):
        """Test that return_null includes null distribution."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 15

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc_test(
            radius=None,
            n_permute=100,
            return_null=True,
            show_progress=False,
            random_state=42,
        )

        assert "null_dist" in result
        assert result["null_dist"] is not None
        assert result["null_dist"].shape[0] == 100  # n_permute

    @pytest.mark.slow
    def test_isc_test_pairwise_method(self, sample_mask):
        """Test isc_test with pairwise method."""
        n_voxels = int(sample_mask.get_fdata().sum())
        n_subjects = 4
        n_obs = 20

        bds = []
        for _ in range(n_subjects):
            bd = BrainData(mask=sample_mask)
            bd.data = np.random.randn(n_obs, n_voxels)
            bds.append(bd)

        bc = BrainCollection(bds, mask=sample_mask)

        result = bc.isc_test(
            method="pairwise",
            radius=None,
            n_permute=100,
            show_progress=False,
            random_state=42,
        )

        assert result["method"] == "pairwise"
        assert isinstance(result["isc"], BrainData)


class TestGLMHelpers:
    """Tests for GLM/Ridge helper functions."""

    def test_resolve_save_path_basic(self, tmp_path):
        """Test basic path resolution with metadata."""
        from nltools.data.collection import _resolve_save_path

        row = pd.Series({"subject": "sub-01", "session": "ses-01"})
        template = str(tmp_path / "{subject}_{session}_betas.nii.gz")

        result = _resolve_save_path(template, row, idx=0)

        assert result == tmp_path / "sub-01_ses-01_betas.nii.gz"

    def test_resolve_save_path_idx(self, tmp_path):
        """Test path resolution with {idx} placeholder."""
        from nltools.data.collection import _resolve_save_path

        row = pd.Series({"subject": "sub-01"})
        template = str(tmp_path / "sub_{idx}_betas.nii.gz")

        result = _resolve_save_path(template, row, idx=5)

        assert result == tmp_path / "sub_5_betas.nii.gz"

    def test_resolve_save_path_creates_dirs(self, tmp_path):
        """Test that parent directories are created."""
        from nltools.data.collection import _resolve_save_path

        row = pd.Series({"subject": "sub-01"})
        template = str(tmp_path / "nested" / "dirs" / "{subject}_betas.nii.gz")

        result = _resolve_save_path(template, row, idx=0)

        assert result.parent.exists()

    def test_resolve_save_path_missing_placeholder(self, tmp_path):
        """Test error on missing placeholder."""
        from nltools.data.collection import _resolve_save_path

        row = pd.Series({"subject": "sub-01"})
        template = str(tmp_path / "{subject}_{missing}_betas.nii.gz")

        with pytest.raises(KeyError, match="missing"):
            _resolve_save_path(template, row, idx=0)

    def test_build_design_matrix_basic(self):
        """Test basic design matrix construction."""
        from nltools.data.collection import _build_subject_design_matrix

        events = pd.DataFrame(
            {
                "onset": [0, 10, 20, 30],
                "duration": [2, 2, 2, 2],
                "trial_type": ["face", "house", "face", "house"],
            }
        )

        dm, task_cols = _build_subject_design_matrix(
            events=events,
            n_scans=50,
            t_r=2.0,
        )

        # Should have task columns
        assert "face" in task_cols
        assert "house" in task_cols
        assert len(task_cols) == 2

        # Design matrix should have correct number of rows
        assert len(dm) == 50

        # Should have drift and constant
        assert "constant" in dm.columns

    def test_build_design_matrix_with_confounds(self, tmp_path):
        """Test design matrix with confounds."""
        from nltools.data.collection import _build_subject_design_matrix

        events = pd.DataFrame(
            {
                "onset": [0, 10, 20],
                "duration": [2, 2, 2],
                "trial_type": ["task", "task", "task"],
            }
        )

        # Create confounds DataFrame
        confounds = pd.DataFrame(
            {
                "motion_x": np.random.randn(30),
                "motion_y": np.random.randn(30),
                "spike_01": np.zeros(30),
            }
        )

        dm, task_cols = _build_subject_design_matrix(
            events=events,
            n_scans=30,
            t_r=2.0,
            confounds=confounds,
            confound_columns=["motion_x", "motion_y"],
        )

        # Task columns should NOT include confounds
        assert "task" in task_cols
        assert "motion_x" not in task_cols

        # But design matrix should have confounds
        assert "motion_x" in dm.columns
        assert "motion_y" in dm.columns
        assert "spike_01" not in dm.columns  # Not selected

    def test_build_design_matrix_confounds_from_file(self, tmp_path):
        """Test loading confounds from TSV file."""
        from nltools.data.collection import _build_subject_design_matrix

        events = pd.DataFrame(
            {
                "onset": [0, 10],
                "duration": [2, 2],
                "trial_type": ["cond", "cond"],
            }
        )

        # Write confounds file
        confounds_path = tmp_path / "confounds.tsv"
        confounds_df = pd.DataFrame(
            {
                "trans_x": np.random.randn(20),
                "trans_y": np.random.randn(20),
            }
        )
        confounds_df.to_csv(confounds_path, sep="\t", index=False)

        dm, task_cols = _build_subject_design_matrix(
            events=events,
            n_scans=20,
            t_r=2.0,
            confounds=confounds_path,
        )

        assert "trans_x" in dm.columns
        assert "trans_y" in dm.columns

    def test_build_design_matrix_missing_confound_column(self):
        """Test error on missing confound column."""
        from nltools.data.collection import _build_subject_design_matrix

        events = pd.DataFrame(
            {
                "onset": [0],
                "duration": [2],
                "trial_type": ["task"],
            }
        )

        confounds = pd.DataFrame(
            {
                "motion_x": np.random.randn(20),
            }
        )

        with pytest.raises(ValueError, match="not found"):
            _build_subject_design_matrix(
                events=events,
                n_scans=20,
                t_r=2.0,
                confounds=confounds,
                confound_columns=["motion_x", "missing_col"],
            )

    def test_build_design_matrix_length_mismatch(self):
        """Test error on confounds length mismatch."""
        from nltools.data.collection import _build_subject_design_matrix

        events = pd.DataFrame(
            {
                "onset": [0],
                "duration": [2],
                "trial_type": ["task"],
            }
        )

        confounds = pd.DataFrame(
            {
                "motion_x": np.random.randn(15),  # Wrong length
            }
        )

        with pytest.raises(ValueError, match="Lengths must match"):
            _build_subject_design_matrix(
                events=events,
                n_scans=20,
                t_r=2.0,
                confounds=confounds,
            )


class TestBrainCollectionFitGLM:
    """Tests for BrainCollection.fit_glm() method."""

    @pytest.fixture
    def small_collection(self, sample_brain_data, sample_mask):
        """Create a small BrainCollection with 3 subjects."""
        # Create collection with 3 subjects, each with ~20 timepoints
        subjects = []
        for i in range(3):
            # Create timeseries data (20 timepoints)
            bd = sample_brain_data.copy()
            data = np.random.randn(20, bd.shape[1]) * 100 + 1000
            bd.data = data
            subjects.append(bd)

        return BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03"]}),
        )

    @pytest.fixture
    def simple_events(self):
        """Create simple task events."""
        return pd.DataFrame(
            {
                "onset": [2, 10, 18, 26],
                "duration": [2, 2, 2, 2],
                "trial_type": ["face", "house", "face", "house"],
            }
        )

    def test_fit_glm_basic(self, small_collection, simple_events):
        """Test basic GLM fitting."""
        betas = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            show_progress=False,
        )

        # Should return BrainCollection
        assert isinstance(betas, BrainCollection)
        # Should have same number of subjects
        assert len(betas) == 3
        # Each subject should have betas for 2 conditions (face, house)
        assert betas[0].shape[0] == 2

    def test_fit_glm_no_scale(self, small_collection, simple_events):
        """Test GLM fitting without scaling."""
        betas = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            scale=False,
            show_progress=False,
        )

        assert isinstance(betas, BrainCollection)
        assert len(betas) == 3

    def test_fit_glm_return_stats(self, small_collection, simple_events):
        """Test GLM with return_stats."""
        result = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            return_stats=["t", "r2"],
            show_progress=False,
        )

        # Should return dict
        assert isinstance(result, dict)
        assert "betas" in result
        assert "t" in result
        assert "r2" in result

        # Check shapes
        assert len(result["betas"]) == 3
        assert len(result["t"]) == 3
        assert len(result["r2"]) == 3

        # t-stats should have same shape as betas
        assert result["t"][0].shape == result["betas"][0].shape
        # r2 should have shape (1, n_voxels)
        assert result["r2"][0].shape[0] == 1

    def test_fit_glm_return_residuals(self, small_collection, simple_events):
        """Test GLM with return_residuals shorthand."""
        result = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            return_residuals=True,
            show_progress=False,
        )

        assert isinstance(result, dict)
        assert "residual" in result
        # Residuals should have shape (n_scans, n_voxels)
        assert result["residual"][0].shape[0] == 20

    def test_fit_glm_with_confounds_list(self, small_collection, simple_events):
        """Test GLM with confounds as list of DataFrames."""
        confounds_list = [
            pd.DataFrame({"motion": np.random.randn(20)}),
            pd.DataFrame({"motion": np.random.randn(20)}),
            pd.DataFrame({"motion": np.random.randn(20)}),
        ]

        betas = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            confounds=confounds_list,
            show_progress=False,
        )

        assert isinstance(betas, BrainCollection)
        assert len(betas) == 3
        # Still only task betas, not confound betas
        assert betas[0].shape[0] == 2

    def test_fit_glm_with_confounds_metadata_column(
        self, sample_brain_data, sample_mask, simple_events, tmp_path
    ):
        """Test GLM with confounds from metadata column."""
        # Create confound files
        confound_paths = []
        for i in range(3):
            path = tmp_path / f"sub-0{i + 1}_confounds.tsv"
            confounds = pd.DataFrame({"motion": np.random.randn(20)})
            confounds.to_csv(path, sep="\t", index=False)
            confound_paths.append(str(path))

        # Create collection with confound paths in metadata
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            bd.data = np.random.randn(20, bd.shape[1]) * 100 + 1000
            subjects.append(bd)

        bc = BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame(
                {
                    "subject": ["sub-01", "sub-02", "sub-03"],
                    "confound_file": confound_paths,
                }
            ),
        )

        betas = bc.fit_glm(
            events=simple_events,
            t_r=2.0,
            confounds="confound_file",
            show_progress=False,
        )

        assert isinstance(betas, BrainCollection)
        assert len(betas) == 3

    def test_fit_glm_save(self, small_collection, simple_events, tmp_path):
        """Test GLM with saving intermediates."""
        save_path = str(tmp_path / "{subject}_betas.nii.gz")

        betas = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            save={"betas": save_path},
            show_progress=False,
        )

        # Check result and files were created
        assert len(betas) == 3
        for subj in ["sub-01", "sub-02", "sub-03"]:
            assert (tmp_path / f"{subj}_betas.nii.gz").exists()

    def test_fit_glm_design_columns_stored(self, small_collection, simple_events):
        """Test that design columns are stored for contrast parsing."""
        betas = small_collection.fit_glm(
            events=simple_events,
            t_r=2.0,
            show_progress=False,
        )

        # Collection should have _design_columns
        assert hasattr(betas, "_design_columns")
        assert "face" in betas._design_columns
        assert "house" in betas._design_columns

        # Individual BrainData should also have it
        assert hasattr(betas[0], "_design_columns")

    def test_fit_glm_invalid_return_stats(self, small_collection, simple_events):
        """Test error on invalid return_stats."""
        with pytest.raises(ValueError, match="Invalid return_stats"):
            small_collection.fit_glm(
                events=simple_events,
                t_r=2.0,
                return_stats=["invalid_stat"],
                show_progress=False,
            )

    def test_resolve_confounds_missing_column(self, small_collection):
        """Test error when confounds column doesn't exist."""
        with pytest.raises(KeyError, match="not found in metadata"):
            small_collection._resolve_confounds("nonexistent_column")

    def test_resolve_confounds_wrong_length(self, small_collection):
        """Test error when confounds list has wrong length."""
        with pytest.raises(ValueError, match="must match collection length"):
            small_collection._resolve_confounds([pd.DataFrame()] * 5)  # Wrong length


class TestBrainCollectionFitRidge:
    """Tests for BrainCollection.fit_ridge() method."""

    @pytest.fixture
    def small_collection(self, sample_brain_data, sample_mask):
        """Create a small BrainCollection with 3 subjects."""
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            data = np.random.randn(20, bd.shape[1]) * 100 + 1000
            bd.data = data
            subjects.append(bd)

        return BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03"]}),
        )

    @pytest.fixture
    def shared_features(self):
        """Create shared feature matrix for all subjects."""
        return np.random.randn(20, 5)  # 20 samples, 5 features

    def test_fit_ridge_basic(self, small_collection, shared_features):
        """Test basic ridge fitting with shared features."""
        weights = small_collection.fit_ridge(
            X=shared_features,
            alpha=1.0,
            show_progress=False,
        )

        # Should return BrainCollection
        assert isinstance(weights, BrainCollection)
        # Should have same number of subjects
        assert len(weights) == 3
        # Each subject should have weights for 5 features
        assert weights[0].shape[0] == 5

    def test_fit_ridge_no_scale(self, small_collection, shared_features):
        """Test ridge fitting without scaling."""
        weights = small_collection.fit_ridge(
            X=shared_features,
            alpha=1.0,
            scale=False,
            show_progress=False,
        )

        assert isinstance(weights, BrainCollection)
        assert len(weights) == 3

    def test_fit_ridge_return_stats(self, small_collection, shared_features):
        """Test ridge with return_stats."""
        result = small_collection.fit_ridge(
            X=shared_features,
            alpha=1.0,
            return_stats=["scores"],
            show_progress=False,
        )

        # Should return dict
        assert isinstance(result, dict)
        assert "weights" in result
        assert "scores" in result

        # Check shapes
        assert len(result["weights"]) == 3
        assert len(result["scores"]) == 3

        # Scores should have shape (1, n_voxels)
        assert result["scores"][0].shape[0] == 1

    def test_fit_ridge_with_cv(self, small_collection, shared_features):
        """Test ridge with cross-validation."""
        weights = small_collection.fit_ridge(
            X=shared_features,
            alpha=1.0,
            cv=3,
            show_progress=False,
        )

        assert isinstance(weights, BrainCollection)
        # Should have CV results
        assert hasattr(weights, "cv_results_")
        assert len(weights.cv_results_) == 3

    def test_fit_ridge_with_features_list(self, small_collection):
        """Test ridge with per-subject features as list."""
        features_list = [
            np.random.randn(20, 5),
            np.random.randn(20, 5),
            np.random.randn(20, 5),
        ]

        weights = small_collection.fit_ridge(
            X=features_list,
            alpha=1.0,
            show_progress=False,
        )

        assert isinstance(weights, BrainCollection)
        assert len(weights) == 3
        assert weights[0].shape[0] == 5

    def test_fit_ridge_with_features_metadata_column(
        self, sample_brain_data, sample_mask, tmp_path
    ):
        """Test ridge with features from metadata column."""
        # Create feature files
        feature_paths = []
        for i in range(3):
            path = tmp_path / f"sub-0{i + 1}_features.npy"
            np.save(path, np.random.randn(20, 5))
            feature_paths.append(str(path))

        # Create collection with feature paths in metadata
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            bd.data = np.random.randn(20, bd.shape[1]) * 100 + 1000
            subjects.append(bd)

        bc = BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame(
                {
                    "subject": ["sub-01", "sub-02", "sub-03"],
                    "features_file": feature_paths,
                }
            ),
        )

        weights = bc.fit_ridge(
            X="features_file",
            alpha=1.0,
            show_progress=False,
        )

        assert isinstance(weights, BrainCollection)
        assert len(weights) == 3

    def test_fit_ridge_save(self, small_collection, shared_features, tmp_path):
        """Test ridge with saving intermediates."""
        save_path = str(tmp_path / "{subject}_weights.nii.gz")

        weights = small_collection.fit_ridge(
            X=shared_features,
            alpha=1.0,
            save={"weights": save_path},
            show_progress=False,
        )

        # Check result and files were created
        assert len(weights) == 3
        for subj in ["sub-01", "sub-02", "sub-03"]:
            assert (tmp_path / f"{subj}_weights.nii.gz").exists()

    def test_fit_ridge_invalid_return_stats(self, small_collection, shared_features):
        """Test error on invalid return_stats."""
        with pytest.raises(ValueError, match="Invalid return_stats"):
            small_collection.fit_ridge(
                X=shared_features,
                alpha=1.0,
                return_stats=["invalid_stat"],
                show_progress=False,
            )

    def test_resolve_features_missing_column(self, small_collection):
        """Test error when features column doesn't exist."""
        with pytest.raises(KeyError, match="not found in metadata"):
            small_collection._resolve_features("nonexistent_column")

    def test_resolve_features_wrong_length(self, small_collection):
        """Test error when features list has wrong length."""
        with pytest.raises(ValueError, match="must match collection length"):
            small_collection._resolve_features([np.array([])] * 5)  # Wrong length


class TestBrainCollectionComputeContrasts:
    """Tests for BrainCollection.compute_contrasts() method."""

    @pytest.fixture
    def betas_collection(self, sample_brain_data, sample_mask):
        """Create a BrainCollection simulating fit_glm output."""
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            # 2 regressors (face, house) x n_voxels
            bd.data = np.random.randn(2, bd.shape[1])
            subjects.append(bd)

        bc = BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03"]}),
        )
        bc._design_columns = ["face", "house"]
        return bc

    def test_compute_contrasts_string(self, betas_collection):
        """Test contrast with string specification."""
        contrast = betas_collection.compute_contrasts("face - house")

        assert isinstance(contrast, BrainCollection)
        assert len(contrast) == 3
        # Each subject should have 1D contrast values
        assert contrast[0].data.ndim == 1

    def test_compute_contrasts_array(self, betas_collection):
        """Test contrast with array specification."""
        contrast = betas_collection.compute_contrasts([1, -1])

        assert isinstance(contrast, BrainCollection)
        assert len(contrast) == 3

    def test_compute_contrasts_dict(self, betas_collection):
        """Test multiple contrasts with dict."""
        contrasts = betas_collection.compute_contrasts(
            {
                "face_vs_house": "face - house",
                "face_only": [1, 0],
            }
        )

        assert isinstance(contrasts, dict)
        assert "face_vs_house" in contrasts
        assert "face_only" in contrasts
        assert len(contrasts["face_vs_house"]) == 3
        assert len(contrasts["face_only"]) == 3

    def test_compute_contrasts_with_coefficient(self, betas_collection):
        """Test contrast with coefficient in string."""
        contrast = betas_collection.compute_contrasts("2*face - house")

        assert isinstance(contrast, BrainCollection)
        # Values should reflect the 2x weighting
        # Just check it runs without error

    def test_compute_contrasts_no_design_columns(self, sample_brain_data, sample_mask):
        """Test error when _design_columns not set."""
        subjects = [sample_brain_data.copy() for _ in range(3)]
        bc = BrainCollection(subjects, mask=sample_mask)

        with pytest.raises(RuntimeError, match="No design columns found"):
            bc.compute_contrasts("face - house")

    def test_compute_contrasts_wrong_length(self, betas_collection):
        """Test error when contrast vector has wrong length."""
        with pytest.raises(ValueError, match="must match number of regressors"):
            betas_collection.compute_contrasts(
                [1, -1, 0]
            )  # 3 elements, but 2 regressors

    def test_compute_contrasts_unknown_column(self, betas_collection):
        """Test error when contrast string references unknown column."""
        with pytest.raises(ValueError, match="not found in design columns"):
            betas_collection.compute_contrasts("face - unknown")


class TestBrainCollectionSelectFeature:
    """Tests for BrainCollection.select_feature() method."""

    @pytest.fixture
    def weights_collection(self, sample_brain_data, sample_mask):
        """Create a BrainCollection simulating fit_ridge output."""
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            # 5 features x n_voxels
            bd.data = np.random.randn(5, bd.shape[1])
            subjects.append(bd)

        bc = BrainCollection(
            subjects,
            mask=sample_mask,
            metadata=pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03"]}),
        )
        bc._feature_names = ["f0", "f1", "f2", "f3", "f4"]
        return bc

    def test_select_feature_by_index(self, weights_collection):
        """Test selecting feature by integer index."""
        feature_0 = weights_collection.select_feature(0)

        assert isinstance(feature_0, BrainCollection)
        assert len(feature_0) == 3
        # Each subject should have 1D weights
        assert feature_0[0].data.ndim == 1

    def test_select_feature_by_name(self, weights_collection):
        """Test selecting feature by name."""
        feature_f2 = weights_collection.select_feature("f2")

        assert isinstance(feature_f2, BrainCollection)
        assert len(feature_f2) == 3

    def test_select_feature_preserves_metadata(self, weights_collection):
        """Test that metadata is preserved."""
        feature_0 = weights_collection.select_feature(0)

        assert "subject" in feature_0.metadata.columns
        assert list(feature_0.metadata["subject"]) == ["sub-01", "sub-02", "sub-03"]

    def test_select_feature_index_out_of_range(self, weights_collection):
        """Test error when feature index out of range."""
        with pytest.raises(IndexError, match="out of range"):
            weights_collection.select_feature(10)

    def test_select_feature_name_not_found(self, weights_collection):
        """Test error when feature name not found."""
        with pytest.raises(KeyError, match="not found"):
            weights_collection.select_feature("unknown_feature")

    def test_select_feature_no_feature_names(self, sample_brain_data, sample_mask):
        """Test error when selecting by name but _feature_names not set."""
        subjects = []
        for i in range(3):
            bd = sample_brain_data.copy()
            bd.data = np.random.randn(5, bd.shape[1])
            subjects.append(bd)

        bc = BrainCollection(subjects, mask=sample_mask)

        # Integer index should still work
        feature_0 = bc.select_feature(0)
        assert len(feature_0) == 3

        # String should fail
        with pytest.raises(RuntimeError, match="_feature_names not set"):
            bc.select_feature("f0")
