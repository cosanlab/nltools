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
from nltools.data.braindata.modeling import RankDeficientDesignWarning


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

    def test_clean_run_relays_nothing_extra(self, bc_with_designs):
        """A full-rank fit relays no RankDeficientDesignWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc_with_designs.fit(model="glm", n_jobs=2)
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
            filename="ghost.py",
            lineno=1,
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
            filename="x.py",
            lineno=3,
            idx=4,
            subject="sub-0005",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _relay_worker_warnings([record], n_subjects=20)
        assert len(w) == 1
        assert w[0].category is RuntimeWarning
        assert "sub-0005" in str(w[0].message)
