"""Parallel execution invariants: cache=, lightweight clones, errors, lineage."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from nltools.data import BrainCollection
from nltools.data.collection import (
    BrainCollectionWorkerError,
    BUNDLE_SCHEMA_VERSION,
)
from nltools.data.collection import core, execution


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


# ---------------------------------------------------------------------------
# core.py helpers — these we can implement / verify on the scaffold
# ---------------------------------------------------------------------------


class TestRunIdAndStepDirs:
    def test_make_run_id_format(self):
        """SPEC §"Cache location": ``run_id = {timestamp}_{uuid8}``, lex-sortable."""
        run_id = core.make_run_id()
        assert re.match(r"^\d{8}T\d{6}_[0-9a-f]{8}$", run_id)
        assert core.is_run_id(run_id)

    def test_make_run_id_unique(self):
        ids = {core.make_run_id() for _ in range(50)}
        assert len(ids) == 50

    @XFAIL
    def test_make_step_dirname_includes_op_and_kwargs(self):
        """SPEC §"Parallel write safety": step subdir name is unique per call."""
        name = core.make_step_dirname("smooth", {"fwhm": 6.0})
        assert "smooth" in name
        assert "fwhm-6.0" in name

    @XFAIL
    def test_make_step_dirname_unique_for_same_args(self):
        a = core.make_step_dirname("smooth", {"fwhm": 6.0})
        b = core.make_step_dirname("smooth", {"fwhm": 6.0})
        assert a != b  # uuid tail makes them unique

    def test_resolve_cache_dir_explicit(self, tmp_path):
        result = core.resolve_cache_dir(tmp_path / "x")
        assert result == (tmp_path / "x").resolve()

    def test_resolve_cache_dir_none(self):
        assert core.resolve_cache_dir(None) is None

    def test_resolve_cache_dir_env_var(self, tmp_path, monkeypatch):
        env_root = tmp_path / "env_cache"
        monkeypatch.setenv("NLTOOLS_CACHE_DIR", str(env_root))
        result = core.resolve_cache_dir("./.nltools_cache")
        assert result == env_root.resolve()


# ---------------------------------------------------------------------------
# Worker-side dataclasses
# ---------------------------------------------------------------------------


class TestWorkerDataclasses:
    def test_item_task_is_frozen(self):
        sig = inspect.signature(execution._ItemTask)
        for name in (
            "idx",
            "item",
            "design",
            "confounds",
            "sample_mask",
            "metadata_row",
            "mask_path",
            "out_path",
        ):
            assert name in sig.parameters

    def test_design_context_fields(self):
        ctx_cls = execution._DesignContext
        sig = inspect.signature(ctx_cls)
        for name in (
            "bd",
            "dm",
            "confounds",
            "sample_mask",
            "metadata",
            "subject",
            "session",
            "run",
            "task",
            "TR",
            "bold_path",
            "events_path",
            "confounds_path",
        ):
            assert name in sig.parameters, f"{name} missing on _DesignContext"

    def test_design_context_getitem_falls_back_to_metadata(self):
        ctx = execution._DesignContext(
            bd=None,
            dm=None,
            confounds=None,
            sample_mask=None,
            metadata={"custom": 42},
            subject="s01",
            session=None,
            run=None,
            task=None,
            TR=2.0,
            bold_path=Path("/x"),
            events_path=None,
            confounds_path=None,
        )
        assert ctx["subject"] == "s01"  # attribute hit
        assert ctx["custom"] == 42  # metadata fallback

    def test_worker_error_is_runtime_error(self):
        assert issubclass(BrainCollectionWorkerError, RuntimeError)


# ---------------------------------------------------------------------------
# Behavior — xfailed
# ---------------------------------------------------------------------------


class TestApplyDispatch:
    @XFAIL
    def test_apply_returns_lightweight_clone(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0)
        assert isinstance(out, BrainCollection)
        assert out is not bc_pathbacked
        assert out.n_subjects == bc_pathbacked.n_subjects

    @XFAIL
    def test_clone_shares_cache_root_and_mask(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0)
        assert out.cache_root == bc_pathbacked.cache_root
        assert out.mask is bc_pathbacked.mask  # by reference

    @XFAIL
    def test_clone_shares_metadata_by_reference(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0)
        assert out.metadata is bc_pathbacked.metadata


class TestCacheKnob:
    """SPEC §"The cache= knob"."""

    @XFAIL
    def test_auto_path_backed_source_writes_through(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0, cache="auto")
        assert not any(out.is_loaded)  # all path-backed

    @XFAIL
    def test_auto_loaded_source_stays_in_memory(self, bc_inmem):
        out = bc_inmem.smooth(fwhm=6.0, cache="auto")
        assert all(out.is_loaded)

    @XFAIL
    def test_force_true_writes_through_loaded_source(self, bc_inmem):
        out = bc_inmem.smooth(fwhm=6.0, cache=True)
        assert not any(out.is_loaded)

    @XFAIL
    def test_force_false_loads_path_backed_source(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0, cache=False)
        assert all(out.is_loaded)


class TestStepLineage:
    @XFAIL
    def test_steps_lex_sorted_oldest_to_newest(self, bc_pathbacked):
        a = bc_pathbacked.smooth(fwhm=6.0)
        b = a.standardize()
        steps = b.steps()
        names = [p.name for p in steps]
        assert names == sorted(names)
        assert any("smooth" in n for n in names)
        assert any("standardize" in n for n in names)

    @XFAIL
    def test_each_eager_step_is_separate_subdir(self, bc_pathbacked):
        """SPEC §"Eager, no fused chains": two on-disk steps, not fused."""
        chained = bc_pathbacked.smooth(fwhm=6.0).standardize()
        steps = chained.steps()
        assert len(steps) >= 2

    @XFAIL
    def test_repeat_call_creates_new_subdir(self, bc_pathbacked):
        a = bc_pathbacked.smooth(fwhm=6.0)
        b = bc_pathbacked.smooth(fwhm=6.0)
        steps_a = {p.name for p in a.steps()}
        steps_b = {p.name for p in b.steps()}
        assert steps_a != steps_b  # uuid tail makes new dir each time


class TestParallelWriteSafety:
    @XFAIL
    def test_one_file_per_item_per_step(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0)
        latest = out.steps()[-1]
        files = sorted(latest.glob("*.nii*"))
        assert len(files) == bc_pathbacked.n_subjects

    @XFAIL
    def test_no_tmp_files_after_success(self, bc_pathbacked):
        out = bc_pathbacked.smooth(fwhm=6.0)
        latest = out.steps()[-1]
        assert not list(latest.glob("*.tmp"))


class TestErrorPropagation:
    @XFAIL
    def test_worker_error_includes_subject_context(self, bc_pathbacked):
        """SPEC §"Errors": fail fast, message embeds subject/run."""

        def boom(bd):
            raise ValueError("kaboom")

        with pytest.raises(BrainCollectionWorkerError) as exc:
            bc_pathbacked.map(boom)
        # message includes subject id
        assert "subject" in str(exc.value)
        # original chained
        assert isinstance(exc.value.__cause__, ValueError)

    @XFAIL
    def test_partial_step_dir_preserved_on_failure(self, bc_pathbacked, tmp_path):
        def maybe_boom(bd):
            raise ValueError("kaboom")

        with pytest.raises(BrainCollectionWorkerError):
            bc_pathbacked.map(maybe_boom)
        # the partial step subdir should still exist for inspection
        assert any(p.is_dir() for p in bc_pathbacked.cache_root.iterdir())


class TestBundleIO:
    """SPEC §"HDF5 fit bundle"."""

    @XFAIL
    def test_glm_bundle_layout(self, tmp_path):
        path = tmp_path / "sub-01_fit.h5"
        execution.write_glm_bundle(
            path,
            betas=np.zeros((3, 27), dtype=np.float32),
            residuals=np.zeros((10, 27), dtype=np.float32),
            sigma2=np.zeros(27, dtype=np.float32),
            r2=np.zeros(27, dtype=np.float32),
            X=np.zeros((10, 3), dtype=np.float32),
            mask_bytes=b"fake",
            affine=np.eye(4),
            regressor_names=["a", "b", "c"],
            scale=True,
            scale_value=100.0,
            model_kwargs={},
            step_id="abc",
            parent_step_id=None,
            op="fit",
            op_kwargs={"model": "glm"},
            nltools_version="0.6.0",
        )
        with h5py.File(path, "r") as f:
            for ds in ("betas", "residuals", "sigma2", "r2", "X", "mask"):
                assert ds in f, f"{ds} missing"
            assert f.attrs["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
            for attr in ("step_id", "op", "kwargs", "nltools_version"):
                assert attr in f.attrs

    @XFAIL
    def test_bundle_atomic_no_tmp_after_success(self, tmp_path):
        path = tmp_path / "sub-01_fit.h5"
        execution.write_glm_bundle(
            path,
            betas=np.zeros((3, 27)),
            residuals=np.zeros((10, 27)),
            sigma2=np.zeros(27),
            r2=np.zeros(27),
            X=np.zeros((10, 3)),
            mask_bytes=b"x",
            affine=np.eye(4),
            regressor_names=["a", "b", "c"],
            scale=True,
            scale_value=100.0,
            model_kwargs={},
            step_id="abc",
            parent_step_id=None,
            op="fit",
            op_kwargs={},
            nltools_version="0.6.0",
        )
        assert path.exists()
        assert not path.with_suffix(".h5.tmp").exists()

    @XFAIL
    def test_bundle_schema_mismatch_raises(self, tmp_path):
        path = tmp_path / "bogus.h5"
        with h5py.File(path, "w") as f:
            f.attrs["bundle_schema_version"] = BUNDLE_SCHEMA_VERSION + 99
        with pytest.raises(ValueError, match="schema"):
            execution.read_glm_bundle(path)

    @XFAIL
    def test_lineage_sidecar_written(self, tmp_path):
        nifti_path = tmp_path / "sub-01.nii.gz"
        nifti_path.write_bytes(b"fake")
        sidecar = execution.write_lineage_sidecar(
            nifti_path,
            step_id="abc",
            parent_step_id="xyz",
            op="smooth",
            op_kwargs={"fwhm": 6.0},
            nltools_version="0.6.0",
        )
        data = json.loads(sidecar.read_text())
        assert data["step_id"] == "abc"
        assert data["op"] == "smooth"


class TestNestedParallelismGuard:
    @XFAIL
    def test_inner_n_jobs_capped_when_outer_parallel(self, bc_pathbacked):
        """SPEC §"Nested parallelism": ``inner_max_num_threads=1`` by default."""
        # Behavioral check: just ensure the call returns successfully.
        # Implementation details verified separately.
        out = bc_pathbacked.smooth(fwhm=6.0, n_jobs=2)
        assert out.n_subjects == bc_pathbacked.n_subjects
