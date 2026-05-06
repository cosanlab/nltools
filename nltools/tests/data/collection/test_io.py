"""IO / cleanup / from_bids / load / unload / write / read."""

from __future__ import annotations


import pytest

from nltools.data import BrainCollection
from nltools.tests.data.collection.conftest import HAS_FAKE_BIDS


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


class TestFromPaths:
    def test_from_paths_basic(self, tiny_mask, tiny_nifti_paths):
        bc = BrainCollection.from_paths(
            tiny_nifti_paths,
            mask=tiny_mask,
            cache_dir=None,
        )
        assert bc.n_subjects == 3
        assert not any(bc.is_loaded)  # path-backed

    def test_from_paths_with_metadata(self, tiny_mask, tiny_nifti_paths):
        bc = BrainCollection.from_paths(
            tiny_nifti_paths,
            mask=tiny_mask,
            metadata={"subject": ["s1", "s2", "s3"], "group": [0, 1, 0]},
            cache_dir=None,
        )
        assert "group" in bc.metadata.columns


class TestFromGlob:
    @XFAIL
    def test_from_glob_basic(self, tiny_mask, tiny_nifti_paths, tmp_path):
        bc = BrainCollection.from_glob(
            str(tmp_path / "sub-*.nii.gz"),
            mask=tiny_mask,
            cache_dir=None,
        )
        assert bc.n_subjects == 3

    @XFAIL
    def test_from_glob_pattern_groups_extract_metadata(
        self,
        tiny_mask,
        tiny_nifti_paths,
        tmp_path,
    ):
        bc = BrainCollection.from_glob(
            str(tmp_path / "sub-*.nii.gz"),
            mask=tiny_mask,
            pattern_groups={"subject": 0},
            cache_dir=None,
        )
        assert "subject" in bc.metadata.columns


@pytest.mark.skipif(not HAS_FAKE_BIDS, reason="nilearn fake-BIDS not available")
class TestFromBIDS:
    """SPEC §"`from_bids` — concrete design"."""

    @XFAIL
    def test_from_bids_basic(self, fake_bids_root, tiny_mask):
        bc = BrainCollection.from_bids(
            fake_bids_root,
            mask=tiny_mask,
            task="task01",
            cache_dir=None,
        )
        assert bc.n_subjects > 0
        # populated metadata columns
        for col in ("subject", "task", "bold_path", "TR"):
            assert col in bc.metadata.columns

    @XFAIL
    def test_from_bids_pairs_events(self, fake_bids_root, tiny_mask):
        bc = BrainCollection.from_bids(
            fake_bids_root,
            mask=tiny_mask,
            task="task01",
            pair_events=True,
            cache_dir=None,
        )
        # at least one design paired
        assert any(d is not None for d in bc.designs)

    @XFAIL
    def test_from_bids_no_task_downgrades_pair_events(
        self,
        fake_bids_root,
        tiny_mask,
    ):
        """SPEC §"Edge cases": ``task=None + pair_events=True`` → silently downgrade."""
        bc = BrainCollection.from_bids(
            fake_bids_root,
            mask=tiny_mask,
            task=None,
            pair_events=True,
            cache_dir=None,
        )
        assert all(d is None for d in bc.designs)

    @XFAIL
    def test_from_bids_TR_from_sidecar(self, fake_bids_root, tiny_mask):
        bc = BrainCollection.from_bids(
            fake_bids_root,
            mask=tiny_mask,
            task="task01",
            TR="infer",
            cache_dir=None,
        )
        # all items have a numeric TR
        assert all(isinstance(t, (int, float)) for t in bc.metadata["TR"].to_list())


class TestLoadUnload:
    @XFAIL
    def test_load_materializes_paths(self, bc_pathbacked):
        bc_pathbacked.load()
        assert all(bc_pathbacked.is_loaded)

    @XFAIL
    def test_load_returns_self_for_chaining(self, bc_pathbacked):
        out = bc_pathbacked.load()
        assert out is bc_pathbacked

    @XFAIL
    def test_unload_drops_inmem_data_when_path_exists(self, bc_pathbacked):
        bc_pathbacked.load()
        bc_pathbacked.unload()
        assert not any(bc_pathbacked.is_loaded)

    @XFAIL
    def test_unload_noop_for_inmem_only(self, bc_inmem):
        """No backing path → unload would lose data; should be a no-op."""
        bc_inmem.unload()
        assert all(bc_inmem.is_loaded)

    @XFAIL
    def test_load_indices_subset(self, bc_pathbacked):
        bc_pathbacked.load(indices=[0])
        loaded = bc_pathbacked.is_loaded
        assert loaded[0] and not any(loaded[1:])


class TestWriteRead:
    @XFAIL
    def test_write_creates_one_file_per_item(self, bc_inmem, tmp_path):
        out_dir = tmp_path / "out"
        paths = bc_inmem.write(out_dir)
        assert len(paths) == bc_inmem.n_subjects
        assert all(p.exists() for p in paths)

    @XFAIL
    def test_write_emits_metadata_csv(self, bc_inmem, tmp_path):
        out_dir = tmp_path / "out"
        bc_inmem.write(out_dir)
        assert (out_dir / "metadata.csv").exists()

    @XFAIL
    def test_read_roundtrip(self, bc_inmem, tmp_path):
        out_dir = tmp_path / "out"
        bc_inmem.write(out_dir)
        bc2 = BrainCollection.read(out_dir, mask=bc_inmem.mask, cache_dir=None)
        assert bc2.n_subjects == bc_inmem.n_subjects


class TestCleanup:
    @XFAIL
    def test_cleanup_removes_cache_root(self, bc_pathbacked):
        root = bc_pathbacked.cache_root
        # Some step subdirs should exist after a parallel op
        bc_pathbacked.smooth(fwhm=6.0)
        assert root.exists()
        bc_pathbacked.cleanup()
        assert not root.exists()

    @XFAIL
    def test_cleanup_invalidates_clones(self, bc_pathbacked):
        clone = bc_pathbacked.smooth(fwhm=6.0)
        bc_pathbacked.cleanup()
        # Operations on the clone should fail since the cache is gone.
        with pytest.raises(FileNotFoundError):
            clone.load()

    @XFAIL
    def test_cleanup_all_classmethod(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Should not raise even if no caches present.
        BrainCollection.cleanup_all(tmp_path)
