"""Warning propagation from BrainCollection workers to the parent process.

Warnings raised inside `_apply` workers (loky processes) previously died on
the worker's stderr: invisible to `warnings.catch_warnings`, `pytest.warns`,
and `filterwarnings("error")` in the parent, and usually invisible entirely
in notebook front-ends. `_wrap_worker` now captures them, `_apply` relays
them — deduplicated across subjects, categories preserved — through the
parent's warning machinery.
"""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from nltools.data import BrainCollection, BrainData, DesignMatrix
from nltools.data.braindata.modeling import (
    NearCollinearDesignWarning,
    RankDeficientDesignWarning,
)


@pytest.fixture(scope="function")
def bc_rank_deficient(tiny_mask):
    """Three-subject collection whose designs are all rank deficient (b = 2a)."""
    rng = np.random.default_rng(0)
    n_obs = 8
    brains = [
        BrainData(
            nib.Nifti1Image(
                rng.standard_normal(tiny_mask.shape + (n_obs,)).astype(np.float32),
                tiny_mask.affine,
            ),
            mask=tiny_mask,
        )
        for _ in range(3)
    ]
    t = np.linspace(0, 2 * np.pi, n_obs)
    designs = [
        DesignMatrix(
            pd.DataFrame({"a": np.sin(t) + i, "b": 2 * (np.sin(t) + i)}), TR=2.0
        )
        for i in range(3)
    ]
    return BrainCollection(
        brains, mask=tiny_mask, designs=designs, lazy=False, cache_dir=None
    )


@pytest.fixture(scope="function")
def bc_near_collinear(tiny_mask):
    """Three-subject collection whose designs are full rank but near-collinear.

    Each subject's b is built to correlate with a at exactly |r| = 0.97 — the
    heritage of the removed ``design_clean`` threshold — so the fit is fine
    for the rank check but must relay ``NearCollinearDesignWarning``.
    """
    rng = np.random.default_rng(7)
    n_obs = 24
    brains = [
        BrainData(
            nib.Nifti1Image(
                rng.standard_normal(tiny_mask.shape + (n_obs,)).astype(np.float32),
                tiny_mask.affine,
            ),
            mask=tiny_mask,
        )
        for _ in range(3)
    ]

    def correlated_pair(seed):
        local = np.random.default_rng(seed)
        a = local.standard_normal(n_obs)
        a = (a - a.mean()) / a.std()
        e = local.standard_normal(n_obs)
        e = e - e.mean()
        e = e - a * (e @ a) / (a @ a)
        e = e / e.std()
        return a, 0.97 * a + np.sqrt(1 - 0.97**2) * e

    designs = []
    for i in range(3):
        a, b = correlated_pair(seed=100 + i)
        designs.append(DesignMatrix(pd.DataFrame({"a": a, "b": b}), TR=2.0))
    return BrainCollection(
        brains, mask=tiny_mask, designs=designs, lazy=False, cache_dir=None
    )


class TestWorkerWarningRelay:
    def _fit_and_record(self, bc, n_jobs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc.fit(model="glm", n_jobs=n_jobs)
        return w

    def test_parallel_worker_warnings_reach_parent(self, bc_rank_deficient):
        """The core fix: loky-worker warnings are catchable in the parent."""
        w = self._fit_and_record(bc_rank_deficient, n_jobs=2)
        assert any(issubclass(x.category, RankDeficientDesignWarning) for x in w), (
            f"no RankDeficientDesignWarning relayed; got {[x.category for x in w]}"
        )

    def test_category_is_preserved(self, bc_rank_deficient):
        """Relayed warnings keep their real class, not a generic UserWarning."""
        w = self._fit_and_record(bc_rank_deficient, n_jobs=2)
        rank = [x for x in w if issubclass(x.category, RankDeficientDesignWarning)]
        assert rank and all(x.category is RankDeficientDesignWarning for x in rank)

    def test_deduplicated_across_subjects_with_annotation(self, bc_rank_deficient):
        """One warning per unique (category, message), annotated with counts."""
        w = self._fit_and_record(bc_rank_deficient, n_jobs=2)
        rank = [x for x in w if issubclass(x.category, RankDeficientDesignWarning)]
        # All three subjects share an identical message -> exactly one relay.
        assert len(rank) == 1
        assert "3/3 subjects" in str(rank[0].message)

    def test_serial_parallel_parity(self, bc_rank_deficient, tiny_mask):
        """n_jobs=1 and n_jobs=2 produce the same relayed warning set."""
        serial = self._fit_and_record(bc_rank_deficient, n_jobs=1)

        # Rebuild (fit consumes nothing, but keep runs independent).
        parallel = self._fit_and_record(bc_rank_deficient, n_jobs=2)

        def rank_msgs(records):
            return sorted(
                str(x.message)
                for x in records
                if issubclass(x.category, RankDeficientDesignWarning)
            )

        assert rank_msgs(serial) == rank_msgs(parallel)

    def test_parent_error_filter_promotes(self, bc_rank_deficient):
        """filterwarnings('error') in the parent turns relayed warnings into raises."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("error", category=RankDeficientDesignWarning)
            with pytest.raises(RankDeficientDesignWarning):
                bc_rank_deficient.fit(model="glm", n_jobs=2)

    def test_relayed_warning_is_attributed_to_the_caller(self, bc_rank_deficient):
        """The relay lands on the user's `bc.fit(...)` line, not on nltools.

        A worker's own stack ends in joblib/loky (parallel) or nltools
        (serial), so the recorded worker location can never be user code; the
        parent re-emits with `find_stack_level()` instead.
        """
        for n_jobs in (1, 2):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                bc_rank_deficient.fit(model="glm", n_jobs=n_jobs)
            rank = [x for x in w if issubclass(x.category, RankDeficientDesignWarning)]
            assert rank and rank[0].filename == __file__, (n_jobs, rank[0].filename)

    def test_clean_run_relays_nothing_extra(self, bc_with_designs):
        """A full-rank fit relays no RankDeficientDesignWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc_with_designs.fit(model="glm", n_jobs=2)
        assert not any(issubclass(x.category, RankDeficientDesignWarning) for x in w)


class TestNearCollinearRelay:
    def test_near_collinear_warning_relayed_from_workers(self, bc_near_collinear):
        """The relay carries NearCollinearDesignWarning with its real category."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc_near_collinear.fit(model="glm", n_jobs=2)
        near = [x for x in w if issubclass(x.category, NearCollinearDesignWarning)]
        assert near, (
            f"no NearCollinearDesignWarning relayed; got {[x.category for x in w]}"
        )
        assert all(x.category is NearCollinearDesignWarning for x in near)
        # Near-collinear is full rank: the exact-deficiency warning must not fire.
        assert not any(issubclass(x.category, RankDeficientDesignWarning) for x in w)


class TestRelayInternals:
    def test_unimportable_category_falls_back_to_userwarning(self):
        from nltools.data.collection.execution import (
            _WorkerWarning,
            _relay_worker_warnings,
        )

        record = _WorkerWarning(
            category_module="nonexistent_module_xyz",
            category_qualname="GhostWarning",
            message="spooky",
            idx=0,
            subject="sub-0001",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _relay_worker_warnings([record], n_subjects=2)
        assert len(w) == 1
        assert w[0].category is UserWarning
        msg = str(w[0].message)
        assert "spooky" in msg
        assert "GhostWarning" in msg  # original class name kept in the text

    def test_single_subject_annotation_names_subject(self):
        from nltools.data.collection.execution import (
            _WorkerWarning,
            _relay_worker_warnings,
        )

        record = _WorkerWarning(
            category_module="builtins",
            category_qualname="RuntimeWarning",
            message="overflow encountered",
            idx=4,
            subject="sub-0005",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _relay_worker_warnings([record], n_subjects=20)
        assert len(w) == 1
        assert w[0].category is RuntimeWarning
        assert "sub-0005" in str(w[0].message)
