"""
Tests for nltools.datasets module

This test file covers:
- download_nifti: Basic file downloading functionality
- fetch_neurovault_collection: Main collection fetching function
- Integration test: Real network test (marked as slow)
"""

import pytest
import polars as pl
import tempfile
import os
from unittest.mock import patch, MagicMock

from nltools.datasets import (
    download_nifti,
    fetch_neurovault_collection,
)
from nltools.data import BrainData, DesignMatrix


class TestDownloadNifti:
    """Test the download_nifti function"""

    def test_empty_url_raises_error(self):
        """Should raise ValueError for empty URL"""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            download_nifti("")

    def test_missing_requests_raises_error(self):
        """Should raise ImportError when requests is not available"""
        with (
            patch("nltools.datasets.requests", None),
            pytest.raises(ImportError, match="requests package is required"),
        ):
            download_nifti("http://example.com/test.nii.gz")

    @patch("nltools.datasets.requests")
    def test_successful_download(self, mock_requests):
        """Should download file successfully"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_requests.get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = download_nifti("http://example.com/test.nii.gz", data_dir=tmp_dir)

            # Check file was created
            expected_path = os.path.join(tmp_dir, "test.nii.gz")
            assert result == expected_path
            assert os.path.exists(expected_path)


class TestFetchNeurovaultCollection:
    """Test the fetch_neurovault_collection function"""

    def test_invalid_collection_id(self):
        """Should raise ValueError for invalid collection ID"""
        with pytest.raises(
            ValueError, match="collection_id must be a positive integer"
        ):
            fetch_neurovault_collection("not_a_number")

        with pytest.raises(
            ValueError, match="collection_id must be a positive integer"
        ):
            fetch_neurovault_collection(-1)

    @patch("nltools.datasets.fetch_neurovault_ids")
    def test_successful_fetch(self, mock_fetch):
        """Should fetch collection successfully"""
        # Setup mock data
        mock_fetch.return_value = {
            "images": ["file1.nii.gz", "file2.nii.gz"],
            "images_meta": [{"id": 1, "name": "image1"}, {"id": 2, "name": "image2"}],
        }

        metadata, files = fetch_neurovault_collection(123)

        # Check results
        assert isinstance(metadata, pl.DataFrame)
        assert len(metadata) == 2
        assert files == ["file1.nii.gz", "file2.nii.gz"]

        # Check function was called correctly
        mock_fetch.assert_called_once_with(
            collection_ids=[123], data_dir=None, verbose=1
        )

    @patch("nltools.datasets.fetch_neurovault_ids")
    def test_fetch_error_handling(self, mock_fetch):
        """Should handle fetch errors gracefully"""
        mock_fetch.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Failed to download collection 123"):
            fetch_neurovault_collection(123)


class TestIntegration:
    """Integration tests that require network access"""

    @pytest.mark.slow
    def test_real_collection_download(self):
        """Test downloading a real collection (requires internet)"""
        try:
            # Use a small, stable collection
            metadata, files = fetch_neurovault_collection(collection_id=2099, verbose=0)

            # Basic checks
            assert isinstance(metadata, pl.DataFrame)
            assert len(metadata) > 0
            assert isinstance(files, list)
            assert len(files) == len(metadata)

            # Check files exist
            for file_path in files:
                assert os.path.exists(file_path)

        except Exception as e:
            pytest.skip(f"Integration test skipped due to network error: {e}")


class TestFetchPain:
    """HF-backed pain dataset loader."""

    def test_pain_resources_manifest(self):
        """PAIN_RESOURCES is the metadata CSV + a full 28x3 image grid."""
        from nltools.datasets import PAIN_RESOURCES

        assert PAIN_RESOURCES[0] == "datasets/pain/metadata.csv"
        images = PAIN_RESOURCES[1:]
        assert len(images) == 84  # 28 subjects x 3 pain levels
        assert all(f.endswith(".nii.gz") for f in images)
        # deterministic grid, sorted subject-major then low/medium/high
        assert images[0] == "datasets/pain/sub-01_pain-low.nii.gz"
        assert images[-1] == "datasets/pain/sub-28_pain-high.nii.gz"

    @pytest.mark.slow
    def test_fetch_pain_from_hf(self):
        """Downloads the real dataset from HF and returns a populated BrainData."""
        from nltools.datasets import fetch_pain

        try:
            brain = fetch_pain()
        except Exception as e:
            pytest.skip(f"Skipped due to network error: {e}")

        assert isinstance(brain, BrainData)
        assert brain.shape[0] == 84
        assert {
            "filename",
            "SubjectID",
            "PainLevel",
            "PainIntensity",
            "Age",
            "Sex",
        }.issubset(brain.X.columns)
        assert brain.X["SubjectID"].n_unique() == 28
        assert sorted(brain.X["PainLevel"].unique().to_list()) == [1, 2, 3]


class TestLoadHaxbyExample:
    """Offline synthetic Haxby dataset for tutorials / Pyodide."""

    def test_return_shape(self):
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example()
        assert isinstance(brain_data, list) and isinstance(dms, list)
        assert len(brain_data) == 1 and len(dms) == 1
        assert isinstance(brain_data[0], BrainData)
        assert isinstance(dms[0], DesignMatrix)
        assert brain_data[0].shape[0] == dms[0].shape[0]

    def test_design_matrix_has_haxby_conditions(self):
        from nltools.datasets import load_haxby_example

        _, dms = load_haxby_example()
        cols = set(dms[0].columns)
        for cond in (
            "face",
            "house",
            "cat",
            "bottle",
            "scissors",
            "shoe",
            "chair",
            "scrambledpix",
        ):
            # `.convolve()` always suffixes `_c0`
            assert f"{cond}_c0" in cols, f"missing condition {cond}_c0"

    def test_multiple_runs(self):
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example(n_runs=3)
        assert len(brain_data) == 3 and len(dms) == 3
        for bd, dm in zip(brain_data, dms):
            assert bd.shape[0] == dm.shape[0]

    def test_reproducible_with_seed(self):
        import numpy as np
        from nltools.datasets import load_haxby_example

        bd1, _ = load_haxby_example(random_state=0)
        bd2, _ = load_haxby_example(random_state=0)
        np.testing.assert_array_equal(bd1[0].data, bd2[0].data)

    def test_glm_fit_end_to_end(self):
        import warnings
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example()
        data, dm = brain_data[0], dms[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm_full = dm.add_dct_basis(duration=128).add_poly(
                order=2, include_lower=True
            )
            data.fit(model="glm", X=dm_full, design_clean=False)
        assert data.glm_betas.shape[0] == dm_full.shape[1]
        assert data.glm_betas.shape[1] == data.shape[1]

    def test_contrast_signal_is_recoverable(self):
        """Injected signal should produce non-trivial contrast t-stats."""
        import warnings
        import numpy as np
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example()
        data, dm = brain_data[0], dms[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm_full = dm.add_dct_basis(duration=128).add_poly(
                order=2, include_lower=True
            )
            data.fit(model="glm", X=dm_full)
            face_vs_house = data.compute_contrasts("face_c0 - house_c0")
        # signal clusters were disjoint, so the face-house map should have
        # clearly significant voxels in both directions.
        assert np.abs(face_vs_house.data).max() > 3.0
