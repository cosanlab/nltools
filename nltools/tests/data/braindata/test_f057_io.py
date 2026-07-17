"""Regression tests for F057 / F058 / F059 in braindata/io.py.

- F057: ``upload_neurovault`` left ``collection`` unbound when
  ``create_collection`` raised ValueError, so it fell through to an
  ``UnboundLocalError`` instead of surfacing a clean error.
- F058: ``load_from_url`` never removed the temp dir it created (leak).
- F059: both helpers named their temp dir from ``os.times()[-1]``
  (collision-prone; ``os.makedirs`` without ``exist_ok`` could crash).
  The fix uses ``tempfile`` so the dir is unique and cleaned up.
"""

import os

import pytest

from nltools.data.braindata import io as io_mod


class _FakeClientCreateFails:
    """pynv.Client stub whose create_collection always raises ValueError."""

    def __init__(self, *args, **kwargs):
        pass

    def create_collection(self, name):
        raise ValueError("Collection name already exists")


class TestUploadNeurovaultUnbound:
    def test_create_collection_failure_raises_cleanly(
        self, minimal_brain_data, monkeypatch
    ):
        """A failed create_collection must not fall through to UnboundLocalError."""
        import pynv

        monkeypatch.setattr(pynv, "Client", _FakeClientCreateFails)

        with pytest.raises(ValueError):
            io_mod.upload_neurovault(
                minimal_brain_data,
                access_token="token",
                collection_name="dupe",
                img_type="Z",
                img_modality="fMRI-BOLD",
            )


class TestLoadFromUrlTempDir:
    def test_temp_dir_cleaned_up(self, minimal_brain_data, monkeypatch):
        """load_from_url must remove the temp dir it downloads into."""
        seen = {}

        def fake_download_nifti(url, data_dir=None):
            seen["data_dir"] = data_dir
            # Directory must already exist (created by tempfile) when we write.
            assert data_dir is not None and os.path.isdir(data_dir)
            path = os.path.join(data_dir, "img.nii.gz")
            with open(path, "wb") as f:
                f.write(b"stub")
            return path

        monkeypatch.setattr("nltools.datasets.download_nifti", fake_download_nifti)
        monkeypatch.setattr(io_mod, "load_from_file", lambda bd, data: None)

        import nibabel as nib

        monkeypatch.setattr(nib, "load", lambda path: object())

        io_mod.load_from_url(minimal_brain_data, "http://example.com/img.nii.gz")

        assert "data_dir" in seen
        assert not os.path.exists(seen["data_dir"]), (
            "load_from_url leaked its temp directory"
        )

    def test_repeated_calls_do_not_collide(self, minimal_brain_data, monkeypatch):
        """Two back-to-back calls must both succeed (no FileExistsError)."""
        dirs = []

        def fake_download_nifti(url, data_dir=None):
            dirs.append(data_dir)
            path = os.path.join(data_dir, "img.nii.gz")
            with open(path, "wb") as f:
                f.write(b"stub")
            return path

        monkeypatch.setattr("nltools.datasets.download_nifti", fake_download_nifti)
        monkeypatch.setattr(io_mod, "load_from_file", lambda bd, data: None)

        import nibabel as nib

        monkeypatch.setattr(nib, "load", lambda path: object())

        for _ in range(3):
            io_mod.load_from_url(minimal_brain_data, "http://example.com/img.nii.gz")

        assert len(dirs) == 3
        for d in dirs:
            assert not os.path.exists(d)
