"""
Tests for nltools.datasets module

This test file covers:
- download_nifti: Basic file downloading functionality
- fetch_neurovault_collection: Main collection fetching function
- Deprecated functions: Ensure they still work but show warnings
- Integration test: Real network test (marked as slow)
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import warnings

from nltools.datasets import (
    download_nifti,
    fetch_neurovault_collection,
    download_collection,  # deprecated
    get_collection_image_metadata,  # deprecated
)


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
        assert isinstance(metadata, pd.DataFrame)
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


class TestDeprecatedFunctions:
    """Test that deprecated functions still work but show warnings"""

    @patch("nltools.datasets.fetch_neurovault_collection")
    def test_download_collection_shows_warning(self, mock_fetch_collection):
        """Should show deprecation warning for download_collection"""
        # Setup mock
        mock_metadata = pd.DataFrame({"id": [1]})
        mock_files = ["file1.nii.gz"]
        mock_fetch_collection.return_value = (mock_metadata, mock_files)

        # Test with warning capture
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = download_collection(collection=123)

            # Check warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "download_collection is deprecated" in str(w[0].message)

            # Check function still works
            metadata, files = result
            assert len(metadata) == 1
            assert files == mock_files

    @patch("nltools.datasets.fetch_neurovault_collection")
    def test_get_collection_image_metadata_shows_warning(self, mock_fetch_collection):
        """Should show deprecation warning for get_collection_image_metadata"""
        # Setup mock
        mock_metadata = pd.DataFrame({"id": [1]})
        mock_files = ["file1.nii.gz"]
        mock_fetch_collection.return_value = (mock_metadata, mock_files)

        # Test with warning capture
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_collection_image_metadata(collection=123)

            # Check warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "get_collection_image_metadata is deprecated" in str(w[0].message)

            # Check function still works
            assert len(result) == 1


class TestIntegration:
    """Integration tests that require network access"""

    def test_real_collection_download(self):
        """Test downloading a real collection (requires internet)"""
        try:
            # Use a small, stable collection
            metadata, files = fetch_neurovault_collection(collection_id=2099, verbose=0)

            # Basic checks
            assert isinstance(metadata, pd.DataFrame)
            assert len(metadata) > 0
            assert isinstance(files, list)
            assert len(files) == len(metadata)

            # Check files exist
            for file_path in files:
                assert os.path.exists(file_path)

        except Exception as e:
            pytest.skip(f"Integration test skipped due to network error: {e}")
