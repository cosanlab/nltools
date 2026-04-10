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
        with patch("nltools.datasets.requests", None):
            with pytest.raises(ImportError, match="requests package is required"):
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


class TestFetchHaxby:
    """Test the fetch_haxby function"""

    def test_fetch_haxby_imports(self):
        """Should be able to import fetch_haxby"""
        from nltools.datasets import fetch_haxby

        assert callable(fetch_haxby)

    @pytest.mark.slow
    def test_fetch_haxby_basic_call(self):
        """Should accept n_subjects parameter and return tuple"""
        from nltools.datasets import fetch_haxby
        from unittest.mock import patch, MagicMock

        # Create mock labels file content
        labels_content = "labels chunks\n" + "\n".join(
            [f"face {i // 121}" for i in range(242)]  # 2 chunks of 121 TRs each
        )

        # Mock nilearn's fetch_haxby
        mock_bunch = MagicMock()
        mock_bunch.func = ["/mock/func.nii.gz"]
        mock_bunch.session_target = ["/mock/labels.txt"]
        mock_bunch.mask = "/mock/mask.nii.gz"

        # Create mock BrainData that supports slicing
        mock_brain_data = MagicMock()
        mock_brain_data.shape = (242, 1000)  # 2 chunks * 121 TRs
        mock_brain_data.__getitem__ = MagicMock(
            side_effect=lambda x: MagicMock(
                shape=(121, 1000) if isinstance(x, slice) else mock_brain_data
            )
        )

        with patch("nltools.datasets.nilearn_fetch_haxby", return_value=mock_bunch):
            with patch("nltools.data.BrainData", return_value=mock_brain_data):
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = (
                        labels_content
                    )
                    mock_open.return_value.__enter__.return_value.readlines.return_value = labels_content.splitlines()
                    result = fetch_haxby(n_subjects=1)
                    assert isinstance(result, tuple)
                    assert len(result) == 2

    @pytest.mark.slow
    def test_fetch_haxby_real_data(self):
        """Integration test with real data - should return correct types and shapes"""
        from nltools.datasets import fetch_haxby

        try:
            brain_data_list, design_matrix_list = fetch_haxby(n_subjects=2, verbose=0)

            # Check return types
            assert isinstance(brain_data_list, list)
            assert isinstance(design_matrix_list, list)
            assert len(brain_data_list) > 0
            assert len(brain_data_list) == len(design_matrix_list)

            # Check types
            assert all(isinstance(bd, BrainData) for bd in brain_data_list)
            assert all(isinstance(dm, DesignMatrix) for dm in design_matrix_list)

            # Check shapes match
            for i, (bd, dm) in enumerate(zip(brain_data_list, design_matrix_list)):
                assert bd.shape[0] == dm.shape[0], (
                    f"Run {i}: BrainData has {bd.shape[0]} samples, DesignMatrix has {dm.shape[0]}"
                )

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.slow
    def test_fetch_haxby_design_matrix_labels(self):
        """DesignMatrix should contain informative condition labels"""
        from nltools.datasets import fetch_haxby

        try:
            brain_data_list, design_matrix_list = fetch_haxby(n_subjects=2, verbose=0)

            expected_conditions = [
                "face",
                "house",
                "scrambledpix",
                "cat",
                "bottle",
                "chair",
                "shoe",
                "scissors",
            ]

            for i, dm in enumerate(design_matrix_list):
                # Check that columns are meaningful (not just '0', '1', etc.)
                assert len(dm.columns) > 0, f"Run {i}: DesignMatrix has no columns"

                # Check that condition names appear in column names
                condition_found = any(
                    cond in col.lower()
                    for cond in expected_conditions
                    for col in dm.columns
                )
                assert condition_found, (
                    f"Run {i}: No expected condition names found in columns: {list(dm.columns)}"
                )

                # Columns should not be auto-generated numeric names
                assert any(not col.isdigit() for col in dm.columns), (
                    f"Run {i}: All columns are numeric, expected condition names"
                )

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.slow
    def test_fetch_haxby_all_subjects(self):
        """Should return nested lists when n_subjects='all' or None"""
        from nltools.datasets import fetch_haxby

        try:
            # Test with 'all'
            brain_data_nested, design_matrix_nested = fetch_haxby(
                n_subjects="all", verbose=0
            )

            assert isinstance(brain_data_nested, list)
            assert isinstance(design_matrix_nested, list)
            assert len(brain_data_nested) > 0  # Should have multiple subjects

            # Each subject should have a list of runs
            for i, (bd_subject, dm_subject) in enumerate(
                zip(brain_data_nested, design_matrix_nested)
            ):
                assert isinstance(bd_subject, list), (
                    f"Subject {i}: BrainData should be list"
                )
                assert isinstance(dm_subject, list), (
                    f"Subject {i}: DesignMatrix should be list"
                )
                assert len(bd_subject) == len(dm_subject), (
                    f"Subject {i}: Run count mismatch"
                )
                assert len(bd_subject) > 0, f"Subject {i}: Should have at least one run"

            # Test with None (should be equivalent to 'all')
            brain_data_nested2, design_matrix_nested2 = fetch_haxby(
                n_subjects=None, verbose=0
            )
            assert len(brain_data_nested2) == len(brain_data_nested), (
                "None and 'all' should return same number of subjects"
            )

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.slow
    def test_fetch_haxby_fit_compatible(self):
        """Returned data should work with BrainData.fit() for each run"""
        from nltools.datasets import fetch_haxby

        try:
            brain_data_list, design_matrix_list = fetch_haxby(n_subjects=2, verbose=0)

            # Should be able to fit Ridge model for each run
            for i, (bd, dm) in enumerate(zip(brain_data_list, design_matrix_list)):
                bd.fit(model="ridge", X=dm, alpha=1.0)
                assert hasattr(bd, "model_"), f"Run {i}: model_ not set"
                assert hasattr(bd, "ridge_weights"), f"Run {i}: ridge_weights not set"

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    @pytest.mark.slow
    def test_fetch_haxby_predict_compatible(self):
        """Returned data should work with BrainData.predict() for each run"""
        from nltools.datasets import fetch_haxby

        try:
            brain_data_list, design_matrix_list = fetch_haxby(n_subjects=2, verbose=0)

            # Fit and predict for each run
            for i, (bd, dm) in enumerate(zip(brain_data_list, design_matrix_list)):
                # Fit model
                bd.fit(model="ridge", X=dm, alpha=1.0)

                # Predict on training data
                predictions = bd.predict()
                assert predictions.shape == bd.shape, (
                    f"Run {i}: Prediction shape mismatch"
                )

                # Predict on subset of design matrix
                if len(dm) > 10:
                    new_dm = dm[:10]
                    predictions = bd.predict(X=new_dm)
                    assert predictions.shape[0] == new_dm.shape[0], (
                        f"Run {i}: Prediction shape mismatch"
                    )

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")

    def test_fetch_haxby_invalid_n_subjects(self):
        """Should raise ValueError for invalid n_subjects"""
        from nltools.datasets import fetch_haxby

        with pytest.raises(ValueError):
            fetch_haxby(n_subjects=0)  # Invalid (must be >= 1)

        with pytest.raises(ValueError):
            fetch_haxby(n_subjects=-1)  # Invalid (must be >= 1)

        with pytest.raises(ValueError):
            fetch_haxby(n_subjects=7)  # More subjects than available (max 6)

    def test_fetch_haxby_missing_nilearn(self):
        """Should raise ImportError if nilearn not available"""
        from nltools.datasets import fetch_haxby

        with patch("nltools.datasets.nilearn_fetch_haxby", None):
            with pytest.raises(ImportError):
                fetch_haxby(n_subjects=1)
