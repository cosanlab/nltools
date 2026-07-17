"""Construction, classmethod factories, and property surface for BrainCollection.

Signature tests pass on the scaffold; behavior tests are xfailed until impl.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from nltools.data import BrainCollection
from nltools.data.collection import BrainCollectionWorkerError, BUNDLE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Public surface — these pass on the scaffold
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """Module-level public exports and class signatures match the spec."""

    def test_public_exports(self):
        from nltools.data import collection as pkg

        assert pkg.BrainCollection is BrainCollection
        assert issubclass(BrainCollectionWorkerError, RuntimeError)
        assert isinstance(BUNDLE_SCHEMA_VERSION, int) and BUNDLE_SCHEMA_VERSION >= 1

    def test_classmethod_factories_exist(self):
        for name in ("from_bids", "from_glob", "from_paths", "read"):
            assert callable(getattr(BrainCollection, name))

    @pytest.mark.parametrize(
        "method",
        ["smooth", "standardize", "detrend", "threshold", "resample"],
    )
    def test_per_subject_methods_exist(self, method):
        assert callable(getattr(BrainCollection, method))

    @pytest.mark.parametrize(
        "method",
        ["fit", "compute_contrasts", "predict", "transform_designs"],
    )
    def test_modeling_methods_exist(self, method):
        assert callable(getattr(BrainCollection, method))

    @pytest.mark.parametrize(
        "method",
        [
            "concat",
            "mean",
            "std",
            "var",
            "median",
            "sum",
            "min",
            "max",
            "ttest",
            "ttest2",
            "anova",
            "permutation_test",
            "permutation_test2",
        ],
    )
    def test_reductions_exist(self, method):
        assert callable(getattr(BrainCollection, method))

    @pytest.mark.parametrize("method", ["isc", "isc_test", "align", "cv"])
    def test_cross_subject_methods_exist(self, method):
        assert callable(getattr(BrainCollection, method))

    @pytest.mark.parametrize(
        "method",
        [
            "map",
            "apply",
            "load",
            "unload",
            "steps",
            "write",
            "cleanup",
            "cleanup_all",
            "memory_estimate",
            "filter",
            "iter_pairs",
        ],
    )
    def test_io_and_composition_methods_exist(self, method):
        assert callable(getattr(BrainCollection, method))


class TestSignatures:
    """Spec-mandated signature properties (kwarg presence, defaults)."""

    def _params(self, fn) -> dict[str, inspect.Parameter]:
        return dict(inspect.signature(fn).parameters)

    @pytest.mark.parametrize(
        "method",
        [
            "smooth",
            "standardize",
            "detrend",
            "threshold",
            "resample",
            "transform_designs",
            "fit",
            "compute_contrasts",
            "predict",
            "map",
            "apply",
            "align",
        ],
    )
    def test_collection_returning_ops_accept_cache(self, method):
        """SPEC §"Coverage rule": methods that return a BrainCollection accept cache=."""
        params = self._params(getattr(BrainCollection, method))
        assert "cache" in params, f"{method} missing cache="
        assert params["cache"].default == "auto"

    @pytest.mark.parametrize(
        "method",
        [
            "mean",
            "std",
            "var",
            "median",
            "sum",
            "min",
            "max",
            "ttest",
            "ttest2",
            "anova",
            "concat",
        ],
    )
    def test_reductions_do_not_accept_cache(self, method):
        """SPEC §"Coverage rule": reductions don't accept cache=."""
        params = self._params(getattr(BrainCollection, method))
        assert "cache" not in params, f"{method} should not accept cache="

    @pytest.mark.parametrize(
        "method",
        [
            "smooth",
            "standardize",
            "detrend",
            "threshold",
            "resample",
            "transform_designs",
            "fit",
            "compute_contrasts",
            "predict",
            "map",
            "apply",
            "align",
            "permutation_test",
            "permutation_test2",
        ],
    )
    def test_parallel_ops_accept_n_jobs(self, method):
        params = self._params(getattr(BrainCollection, method))
        assert "n_jobs" in params, f"{method} missing n_jobs"
        assert params["n_jobs"].default == -1

    @pytest.mark.parametrize(
        "method",
        [
            "smooth",
            "standardize",
            "detrend",
            "threshold",
            "resample",
            "transform_designs",
            "fit",
            "compute_contrasts",
            "predict",
            "map",
            "apply",
            "align",
        ],
    )
    def test_parallel_ops_accept_progress_bar(self, method):
        params = self._params(getattr(BrainCollection, method))
        assert "progress_bar" in params
        assert params["progress_bar"].default is False

    @pytest.mark.parametrize("method", ["isc", "isc_test"])
    def test_serial_isc_ops_do_not_advertise_parallelism(self, method):
        """F068: isc/isc_test are serial — they must not accept parallel kwargs.

        They never route through the joblib machinery in execution.py, so
        n_jobs/progress_bar/device were accepted-and-ignored no-ops. radius_mm
        likewise only ever meant searchlight ISC, which does not exist.
        """
        params = self._params(getattr(BrainCollection, method))
        for kwarg in ("n_jobs", "progress_bar", "device", "radius_mm"):
            assert kwarg not in params, f"{method} should not advertise {kwarg}"

    def test_fit_default_model_is_glm(self):
        params = self._params(BrainCollection.fit)
        assert params["model"].default == "glm"

    def test_compute_contrasts_default_type_is_beta(self):
        """SPEC §1023: statistic defaults to 'beta'."""
        params = self._params(BrainCollection.compute_contrasts)
        assert params["statistic"].default == "beta"

    def test_ttest_returns_dict_signature(self):
        sig = inspect.signature(BrainCollection.ttest)
        assert "popmean" in sig.parameters
        assert sig.parameters["popmean"].default == 0.0

    def test_predict_dispatch_args_present(self):
        params = self._params(BrainCollection.predict)
        assert "y" in params and "X_new" in params
        assert params["y"].default is None and params["X_new"].default is None

    def test_predict_default_model_and_cv(self):
        params = self._params(BrainCollection.predict)
        assert params["model"].default == "svm"
        assert params["cv"].default == "loso"

    def test_isc_default_method_loo(self):
        """SPEC §"Cross-subject ops": ``method='loo'`` default."""
        params = self._params(BrainCollection.isc)
        assert params["method"].default == "loo"


# ---------------------------------------------------------------------------
# Behavior tests — xfail until impl
# ---------------------------------------------------------------------------


class TestInit:
    """Construction from explicit lists."""

    def test_init_with_loaded_brain_data(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=i) for i in range(3)]
        bc = BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
        assert bc.n_subjects == 3
        assert all(bc.is_loaded)

    def test_init_with_metadata_dict(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=i) for i in range(2)]
        bc = BrainCollection(
            brains,
            mask=tiny_mask,
            metadata={"subject": ["s01", "s02"], "group": [0, 1]},
            cache_dir=None,
        )
        assert isinstance(bc.metadata, pl.DataFrame)
        assert "group" in bc.metadata.columns

    def test_init_metadata_length_mismatch_raises(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=i) for i in range(2)]
        with pytest.raises(ValueError):
            BrainCollection(
                brains,
                mask=tiny_mask,
                metadata={"subject": ["s01", "s02", "s03"]},
                cache_dir=None,
            )

    def test_init_designs_length_mismatch_raises(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=i) for i in range(2)]
        with pytest.raises(ValueError):
            BrainCollection(brains, mask=tiny_mask, designs=[None], cache_dir=None)

    def test_init_cache_root_resolved_at_construction(
        self,
        tiny_mask,
        tiny_brain_factory,
        tmp_path,
        monkeypatch,
    ):
        """SPEC §"Capture timing": cache root is frozen at construction."""
        brains = [tiny_brain_factory(seed=0)]
        monkeypatch.chdir(tmp_path)
        bc = BrainCollection(
            brains,
            mask=tiny_mask,
            cache_dir=tmp_path / "cache",
        )
        original = bc.cache_root
        monkeypatch.chdir(tmp_path.parent)
        assert bc.cache_root == original  # not relocated by chdir

    def test_init_cache_dir_none_uses_tempdir(self, tiny_mask, tiny_brain_factory):
        brains = [tiny_brain_factory(seed=0)]
        bc = BrainCollection(brains, mask=tiny_mask, cache_dir=None)
        assert bc._cache_root is None or bc._cache_root.exists()

    def test_init_cache_dir_env_var_precedence(
        self,
        tiny_mask,
        tiny_brain_factory,
        tmp_path,
        monkeypatch,
    ):
        """SPEC §"Cache location": NLTOOLS_CACHE_DIR env overrides default."""
        env_root = tmp_path / "env_cache"
        monkeypatch.setenv("NLTOOLS_CACHE_DIR", str(env_root))
        brains = [tiny_brain_factory(seed=0)]
        bc = BrainCollection(brains, mask=tiny_mask)  # default cache_dir
        assert env_root in bc.cache_root.parents or bc.cache_root.parent == env_root


class TestProperties:
    def test_n_subjects_n_voxels_shape(self, bc_inmem):
        assert bc_inmem.n_subjects == 3
        assert bc_inmem.n_voxels == 27  # 3*3*3 mask
        n_sub, n_obs, n_vox = bc_inmem.shape
        assert n_sub == 3 and n_vox == 27

    def test_is_loaded_inmem(self, bc_inmem):
        assert all(bc_inmem.is_loaded)

    def test_is_loaded_pathbacked(self, bc_pathbacked):
        assert not any(bc_pathbacked.is_loaded)

    def test_metadata_is_polars(self, bc_inmem):
        assert isinstance(bc_inmem.metadata, pl.DataFrame)

    def test_designs_returns_list(self, bc_inmem):
        d = bc_inmem.designs
        assert isinstance(d, list) and len(d) == bc_inmem.n_subjects

    def test_cache_root_under_cwd_default(self, bc_pathbacked, tmp_path):
        assert tmp_path in bc_pathbacked.cache_root.parents

    def test_memory_estimate_returns_string(self, bc_inmem):
        assert isinstance(bc_inmem.memory_estimate(), str)


# ---------------------------------------------------------------------------
# Lifecycle invariants
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_del_is_noop(self):
        """SPEC §"`__del__` is a no-op."""
        bc = BrainCollection.__new__(BrainCollection)
        # Should not raise even on uninitialized instance.
        assert bc.__del__() is None

    def test_repr_uninitialized(self):
        bc = BrainCollection.__new__(BrainCollection)
        assert "uninitialized" in repr(bc)

    def test_clone_helper_exists(self):
        """Internal _clone helper used by every parallel op."""
        assert callable(getattr(BrainCollection, "_clone"))
        assert callable(getattr(BrainCollection, "_next_step_id"))
