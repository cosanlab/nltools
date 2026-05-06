"""Group reductions, ttest/anova/perm tests, ISC, alignment."""

from __future__ import annotations

import numpy as np
import pytest

from nltools.data import BrainCollection, BrainData


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


class TestReductionShape:
    """SPEC §"Group reductions": collapse subjects → BrainData (or dict)."""

    @XFAIL
    @pytest.mark.parametrize(
        "method",
        ["mean", "std", "var", "median", "sum", "min", "max"],
    )
    def test_simple_reductions_return_braindata(self, bc_inmem, method):
        out = getattr(bc_inmem, method)()
        assert isinstance(out, BrainData)

    @XFAIL
    def test_concat_returns_stacked_braindata(self, bc_inmem):
        out = bc_inmem.concat()
        assert isinstance(out, BrainData)
        # n_total_obs == sum of per-subject n_obs
        per_sub_obs = sum(bd.shape[0] for bd in bc_inmem)
        assert out.shape[0] == per_sub_obs

    @XFAIL
    def test_ttest_returns_dict_with_keys(self, bc_inmem):
        out = bc_inmem.ttest()
        for k in ("mean", "t", "z", "p"):
            assert k in out
            assert isinstance(out[k], BrainData)

    @XFAIL
    def test_ttest2_signature(self, bc_inmem):
        out = bc_inmem.ttest2(bc_inmem)
        for k in ("mean", "t", "z", "p"):
            assert k in out

    @XFAIL
    def test_anova_returns_dict(self, bc_inmem):
        out = bc_inmem.anova(np.array([0, 1, 0]))
        assert "f" in out or "F" in out


class TestStreaming:
    """SPEC §"Streaming algorithms": path-backed inputs stream when possible."""

    @XFAIL
    def test_mean_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.mean()
        assert isinstance(out, BrainData)
        # All items still path-backed afterward (not loaded as a side-effect).
        assert not any(bc_pathbacked.is_loaded)

    @XFAIL
    def test_ttest_streams_from_path_backed(self, bc_pathbacked):
        out = bc_pathbacked.ttest()
        assert "t" in out
        assert not any(bc_pathbacked.is_loaded)


class TestPermutationTest:
    @XFAIL
    def test_permutation_test_returns_dict(self, bc_inmem):
        out = bc_inmem.permutation_test(n_permute=20, random_state=0)
        assert isinstance(out, dict)

    @XFAIL
    def test_permutation_test_return_null(self, bc_inmem):
        out = bc_inmem.permutation_test(
            n_permute=10,
            return_null=True,
            random_state=0,
        )
        assert "null" in out or "null_distribution" in out


class TestISC:
    @XFAIL
    def test_isc_loo_returns_dict(self, bc_inmem):
        out = bc_inmem.isc(method="loo")
        assert isinstance(out, dict)

    @XFAIL
    def test_isc_test_returns_dict(self, bc_inmem):
        out = bc_inmem.isc_test(method="loo", n_permute=10)
        assert isinstance(out, dict)


class TestAlign:
    @XFAIL
    def test_align_returns_collection(self, bc_inmem):
        out = bc_inmem.align(method="procrustes", scheme="searchlight")
        assert isinstance(out, BrainCollection)

    @XFAIL
    def test_align_with_return_model_returns_tuple(self, bc_inmem):
        out, model = bc_inmem.align(
            method="procrustes",
            scheme="searchlight",
            return_model=True,
        )
        assert isinstance(out, BrainCollection)
        assert model is not None
