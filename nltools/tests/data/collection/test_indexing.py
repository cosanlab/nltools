"""Indexing, iteration, and filter behavior for BrainCollection."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nltools.data import BrainCollection, BrainData


XFAIL = pytest.mark.xfail(reason="not implemented", strict=True)


class TestGetItem:
    @XFAIL
    def test_int_returns_braindata(self, bc_inmem):
        bd = bc_inmem[0]
        assert isinstance(bd, BrainData)

    @XFAIL
    def test_slice_returns_collection(self, bc_inmem):
        sub = bc_inmem[0:2]
        assert isinstance(sub, BrainCollection)
        assert sub.n_subjects == 2

    @XFAIL
    def test_list_returns_collection(self, bc_inmem):
        sub = bc_inmem[[0, 2]]
        assert isinstance(sub, BrainCollection)
        assert sub.n_subjects == 2

    @XFAIL
    def test_bool_mask_returns_collection(self, bc_inmem):
        mask = np.array([True, False, True])
        sub = bc_inmem[mask]
        assert isinstance(sub, BrainCollection)
        assert sub.n_subjects == 2

    @XFAIL
    def test_subject_label_lookup(self, bc_inmem):
        """SPEC §"Indexing": ``bc['sub-01']`` looks up via metadata['subject']."""
        bd = bc_inmem["sub-01"]
        assert isinstance(bd, BrainData)

    @XFAIL
    def test_polars_expression_filter(self, bc_inmem):
        sub = bc_inmem[pl.col("subject") == "sub-01"]
        assert isinstance(sub, BrainCollection)

    @XFAIL
    def test_subset_shares_cache_root(self, bc_pathbacked):
        sub = bc_pathbacked[0:2]
        assert sub.cache_root == bc_pathbacked.cache_root

    @XFAIL
    def test_out_of_range_raises(self, bc_inmem):
        with pytest.raises(IndexError):
            _ = bc_inmem[99]


class TestIter:
    @XFAIL
    def test_iter_yields_braindata(self, bc_inmem):
        items = list(bc_inmem)
        assert len(items) == 3
        assert all(isinstance(x, BrainData) for x in items)

    @XFAIL
    def test_iter_pairs_yields_tuples(self, bc_inmem):
        for bd, dm in bc_inmem.iter_pairs():
            assert isinstance(bd, BrainData)
            # dm may be None when no design paired
            assert dm is None or hasattr(dm, "data")

    def test_len(self):
        """Works on uninitialized via __new__ (returns 0 if _items absent)."""
        bc = BrainCollection.__new__(BrainCollection)
        bc._items = []
        assert len(bc) == 0


class TestFilter:
    @XFAIL
    def test_filter_callable(self, bc_inmem):
        sub = bc_inmem.filter(lambda bd: bd.shape[0] >= 1)
        assert isinstance(sub, BrainCollection)

    @XFAIL
    def test_filter_bool_array(self, bc_inmem):
        sub = bc_inmem.filter(np.array([True, False, True]))
        assert sub.n_subjects == 2

    @XFAIL
    def test_filter_polars_series(self, bc_inmem):
        sub = bc_inmem.filter(pl.Series([True, True, False]))
        assert sub.n_subjects == 2

    @XFAIL
    def test_filter_length_mismatch_raises(self, bc_inmem):
        with pytest.raises(ValueError):
            bc_inmem.filter(np.array([True, False]))
