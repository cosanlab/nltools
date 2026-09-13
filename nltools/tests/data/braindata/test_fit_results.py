"""Tests for the frozen `FitResult` record `BrainData.fit` leaves on `.model`."""

from dataclasses import FrozenInstanceError

import nibabel as nib
import numpy as np
import pytest

from nltools.data import BrainData, DesignMatrix, FitResult


N_SAMPLES, N_VOXELS = 20, 8


@pytest.fixture
def maps():
    """One BrainData per FitResult map field, all on the same tiny mask."""
    mask = nib.Nifti1Image(np.ones((N_VOXELS, 1, 1), dtype=np.uint8), np.eye(4))
    rng = np.random.default_rng(0)

    def _make(n_rows):
        return BrainData(rng.standard_normal((n_rows, N_VOXELS)), mask=mask)

    return {
        "betas": _make(2),
        "predicted": _make(N_SAMPLES),
        "residual": _make(N_SAMPLES),
        "r2": _make(1),
    }


@pytest.fixture
def record(maps):
    design = DesignMatrix(
        {"a": np.arange(N_SAMPLES, dtype=float), "b": np.ones(N_SAMPLES)},
        sampling_freq=0.5,
    )
    return FitResult(kind="glm", design=design, _estimator=object(), **maps)


def test_fields_cannot_be_rebound(record):
    """The record is frozen, like every other result type."""
    with pytest.raises(FrozenInstanceError):
        record.kind = "ridge"


def test_record_owns_its_maps(maps):
    """Mutating a map handed to the record never reaches the stored copy."""
    record = FitResult(kind="glm", design=None, **maps)
    maps["betas"].data[0, 0] = 1234.0
    assert record.betas.data[0, 0] != 1234.0


def test_repr_hides_the_estimator(record):
    """The fitted estimator is internal state, not something the repr advertises."""
    assert "_estimator" not in repr(record)
    assert "kind='glm'" in repr(record)


def test_estimator_is_held_as_given(maps):
    """The estimator is the facade's own object, not a copy of it."""
    estimator = object()
    record = FitResult(kind="glm", design=None, _estimator=estimator, **maps)
    assert record._estimator is estimator
