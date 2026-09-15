"""
Tests for nltools.datasets module

This test file covers:
- download_nifti: Basic file downloading functionality
- fetch_neurovault_collection: Main collection fetching function
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

    @patch("nltools.datasets.requests")
    def test_successful_download(self, mock_requests):
        """Should download file successfully"""
        # Setup mock response (used as a context manager)
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test data"]
        mock_response.__enter__.return_value = mock_response
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

    @patch("nltools.datasets.fetch_neurovault_ids")
    def test_empty_images_raises_runtime_error(self, mock_fetch):
        """Should raise RuntimeError naming the collection when nothing downloaded"""
        mock_fetch.return_value = {"images": [], "images_meta": []}

        with pytest.raises(RuntimeError, match="collection 123 returned no images"):
            fetch_neurovault_collection(123)


class TestFetchPain:
    """HF-backed pain dataset loader."""

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


class TestFetchEmotionRatings:
    """HF-backed IAPS emotion-rating dataset loader."""

    @pytest.mark.slow
    def test_fetch_emotion_ratings_from_hf(self):
        """Downloads the real dataset from HF and returns a populated BrainData."""
        from nltools.datasets import fetch_emotion_ratings

        try:
            brain = fetch_emotion_ratings()
        except Exception as e:
            pytest.skip(f"Skipped due to network error: {e}")

        assert isinstance(brain, BrainData)
        assert brain.shape[0] == 679
        assert {"filename", "SubjectID", "Rating", "Holdout"}.issubset(brain.X.columns)
        assert sorted(brain.X["Rating"].unique().to_list()) == [1, 2, 3, 4, 5]


class TestLoadHaxbyExample:
    """Offline synthetic Haxby dataset for tutorials and tests."""

    def test_return_shape(self):
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example(space="grid")
        assert isinstance(brain_data, list) and isinstance(dms, list)
        assert len(brain_data) == 1 and len(dms) == 1
        assert isinstance(brain_data[0], BrainData)
        assert isinstance(dms[0], DesignMatrix)
        assert brain_data[0].shape[0] == dms[0].shape[0]

    def test_design_matrix_has_haxby_conditions(self):
        from nltools.datasets import load_haxby_example

        _, dms = load_haxby_example(space="grid")
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

    def test_shared_block_order_repeats_one_order_across_runs(self):
        """`block_order='shared'` gives every run the same condition sequence."""
        from nltools.datasets import load_haxby_example

        shared, _ = load_haxby_example(n_runs=2, space="grid", block_order="shared")
        assert shared[0].Y["condition"].to_list() == shared[1].Y["condition"].to_list()

        independent, _ = load_haxby_example(n_runs=2, space="grid")
        assert (
            independent[0].Y["condition"].to_list()
            != independent[1].Y["condition"].to_list()
        )

    def test_reproducible_with_seed(self):
        import numpy as np
        from nltools.datasets import load_haxby_example

        bd1, _ = load_haxby_example(random_state=0, space="grid")
        bd2, _ = load_haxby_example(random_state=0, space="grid")
        np.testing.assert_array_equal(bd1[0].data, bd2[0].data)

    def test_default_space_is_the_3mm_mni_grid(self):
        """The default data live on the package's 3 mm MNI mask."""
        import numpy as np
        import nibabel as nib
        from nltools.datasets import load_haxby_example
        from nltools.templates import BrainSpaceConfig

        template_mask = nib.load(
            BrainSpaceConfig(template="default", resolution=3).mask
        )
        n_voxels = int((np.asarray(template_mask.dataobj) > 0).sum())

        brain_data, _ = load_haxby_example()
        data = brain_data[0]
        assert data.mask.shape == template_mask.shape
        np.testing.assert_array_equal(data.mask.affine, template_mask.affine)
        assert data.shape == (72, n_voxels)

    def test_grid_space_keeps_the_tiny_volume(self):
        """`space='grid'` still returns the 500-voxel synthetic volume."""
        from nltools.datasets import load_haxby_example

        brain_data, _ = load_haxby_example(space="grid")
        assert brain_data[0].shape == (72, 500)

    def test_labels_align_with_the_design_matrix(self):
        """Each condition's `.Y` label sits on the peak of its own regressor."""
        import numpy as np
        from nltools.datasets import load_haxby_example

        brain_data, dms = load_haxby_example(space="grid")
        data, dm = brain_data[0], dms[0]
        labels = data.Y["condition"].to_numpy()
        assert len(labels) == data.shape[0]
        for condition in ("face", "house", "cat", "scrambledpix"):
            peak_tr = int(np.argmax(np.asarray(dm[f"{condition}_c0"])))
            assert labels[peak_tr] == condition

    def test_glass_brain_plot_runs(self):
        """The data are standard-space enough for a glass brain."""
        import matplotlib.pyplot as plt
        from nltools.datasets import load_haxby_example

        brain_data, _ = load_haxby_example()
        brain_data[0].mean().plot(method="glass")
        plt.close("all")

    @pytest.mark.slow
    def test_contrast_peak_lands_in_the_face_sphere(self):
        """A face > house contrast peaks inside the simulated face ROI."""
        import numpy as np
        from nibabel.affines import apply_affine
        from nltools.datasets import load_haxby_example
        from nltools.data.simulator.haxby import _ROI_CENTERS_MNI, _ROI_RADIUS_MM

        brain_data, dms = load_haxby_example()
        data = brain_data[0]
        data.fit("glm", X=dms[0])
        contrast = data.compute_contrasts("face_c0 - house_c0")

        img = contrast.to_nifti()
        volume = img.get_fdata()
        peak_ijk = np.unravel_index(int(np.argmax(volume)), volume.shape)
        peak_mm = apply_affine(img.affine, peak_ijk)
        distance = np.linalg.norm(peak_mm - np.array(_ROI_CENTERS_MNI["face"]))
        assert distance <= _ROI_RADIUS_MM

    @pytest.mark.slow
    def test_face_vs_house_decoding_is_above_chance_but_not_perfect(self):
        """Decoding inside the response spheres beats chance without hitting 1.0."""
        from nltools.datasets import load_haxby_example
        from nltools.data.simulator.haxby import _ROI_CENTERS_MNI, _ROI_RADIUS_MM
        from nltools.mask import create_sphere

        brain_data, _ = load_haxby_example()
        data = brain_data[0]
        ventral_temporal = create_sphere(
            [list(_ROI_CENTERS_MNI["face"]), list(_ROI_CENTERS_MNI["house"])],
            radius=_ROI_RADIUS_MM,
            mask=data.mask,
        )
        trs = data[data.Y["condition"].is_in(["face", "house"])]
        result = trs.apply_mask(ventral_temporal).predict(y="condition", cv=3)
        assert 0.6 <= result.mean_score < 1.0
