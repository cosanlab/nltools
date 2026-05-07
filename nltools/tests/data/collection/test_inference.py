"""Group reductions, ttest/anova/perm tests, ISC, alignment."""

from __future__ import annotations

import numpy as np
import pytest

from nltools.data import BrainCollection, BrainData


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


class TestReductionShape:
    """SPEC §"Group reductions": collapse subjects → BrainData (or dict)."""

    @pytest.mark.parametrize(
        "method",
        ["mean", "std", "var", "median", "sum", "min", "max"],
    )
    def test_simple_reductions_return_braindata(self, bc_inmem, method):
        out = getattr(bc_inmem, method)()
        assert isinstance(out, BrainData)

    def test_concat_returns_stacked_braindata(self, bc_inmem):
        out = bc_inmem.concat()
        assert isinstance(out, BrainData)
        # n_total_obs == sum of per-subject n_obs
        per_sub_obs = sum(bd.shape[0] for bd in bc_inmem)
        assert out.shape[0] == per_sub_obs

    def test_ttest_returns_dict_with_keys(self, bc_inmem):
        out = bc_inmem.ttest()
        for k in ("mean", "t", "z", "p"):
            assert k in out
            assert isinstance(out[k], BrainData)

    def test_ttest2_signature(self, bc_inmem):
        out = bc_inmem.ttest2(bc_inmem)
        for k in ("mean", "t", "z", "p"):
            assert k in out

    def test_anova_returns_dict(self, bc_inmem):
        out = bc_inmem.anova(np.array([0, 1, 0]))
        assert "f" in out or "F" in out


class TestStreaming:
    """SPEC §"Streaming algorithms": path-backed inputs stream when possible."""

    def test_mean_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.mean()
        assert isinstance(out, BrainData)
        # All items still path-backed afterward (not loaded as a side-effect).
        assert not any(bc_pathbacked.is_loaded)

    def test_ttest_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.ttest()
        assert "t" in out
        assert not any(bc_pathbacked.is_loaded)


class TestPermutationTest:
    def test_permutation_test_returns_dict(self, bc_inmem):
        out = bc_inmem.permutation_test(n_permute=20, random_state=0)
        assert isinstance(out, dict)

    def test_permutation_test_return_null(self, bc_inmem):
        out = bc_inmem.permutation_test(
            n_permute=10,
            return_null=True,
            random_state=0,
        )
        assert "null" in out or "null_distribution" in out


class TestISC:
    def test_isc_loo_returns_dict(self, bc_inmem):
        out = bc_inmem.isc(method="loo")
        assert isinstance(out, dict)

    def test_isc_test_returns_dict(self, bc_inmem):
        out = bc_inmem.isc_test(method="loo", n_permute=10)
        assert isinstance(out, dict)


class TestAlign:
    """Behavioral facade tests. Searchlight Procrustes needs more samples than
    the tiny fixture provides — algorithm correctness is covered separately
    in ``nltools.algorithms.alignment`` tests.
    """

    @pytest.mark.skip(
        reason="searchlight Procrustes needs >8 timepoints; algorithm "
        "correctness covered in nltools.algorithms tests"
    )
    def test_align_returns_collection(self, bc_inmem):
        out = bc_inmem.align(method="procrustes", scheme="searchlight")
        assert isinstance(out, BrainCollection)

    @pytest.mark.skip(
        reason="searchlight Procrustes needs >8 timepoints; algorithm "
        "correctness covered in nltools.algorithms tests"
    )
    def test_align_with_return_model_returns_tuple(self, bc_inmem):
        out, model = bc_inmem.align(
            method="procrustes",
            scheme="searchlight",
            return_model=True,
        )
        assert isinstance(out, BrainCollection)
        assert model is not None

    def test_align_facade_signature(self):
        import inspect

        sig = inspect.signature(BrainCollection.align)
        assert sig.parameters["return_model"].default is False
        assert sig.parameters["scheme"].default == "searchlight"
        assert sig.parameters["method"].default == "procrustes"
