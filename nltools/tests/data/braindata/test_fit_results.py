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


@pytest.fixture
def fitted_glm(minimal_brain_data):
    n = len(minimal_brain_data)
    design = DesignMatrix(
        {"intercept": np.ones(n), "cond": np.random.default_rng(0).normal(size=n)}
    )
    return minimal_brain_data.fit(model="glm", X=design)


class TestFitResultWrite:
    """`FitResult.write` puts the maps, the design and a sidecar in one directory."""

    def test_writes_every_glm_map_the_design_and_a_sidecar(self, fitted_glm, tmp_path):
        import json

        written = fitted_glm.model.write(tmp_path / "out", prefix="sub-01")

        names = sorted(path.name for path in written)
        assert names == [
            "sub-01_betas.nii.gz",
            "sub-01_design.csv",
            "sub-01_fit.json",
            "sub-01_predicted.nii.gz",
            "sub-01_r2.nii.gz",
            "sub-01_residual.nii.gz",
        ]
        assert all(path.exists() for path in written)
        sidecar = json.loads((tmp_path / "out" / "sub-01_fit.json").read_text())
        assert sidecar["kind"] == "glm"
        assert sidecar["columns"] == ["intercept", "cond"]

    def test_ridge_also_writes_the_selected_alpha(self, minimal_brain_data, tmp_path):
        X = np.random.default_rng(1).normal(size=(len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        written = minimal_brain_data.model.write(tmp_path)

        assert (tmp_path / "alpha.nii.gz") in written
        assert (tmp_path / "design.csv").exists()

    def test_banded_ridge_writes_one_design_per_feature_space(
        self, minimal_brain_data, tmp_path
    ):
        rng = np.random.default_rng(2)
        spaces = {
            "a": rng.normal(size=(len(minimal_brain_data), 3)),
            "b": rng.normal(size=(len(minimal_brain_data), 2)),
        }
        minimal_brain_data.fit(
            model="ridge",
            X=spaces,
            ridge_alpha=[1.0, 10.0],
            ridge_cv=3,
            ridge_search_iterations=4,
            random_state=0,
        )

        minimal_brain_data.model.write(tmp_path)

        assert (tmp_path / "design-a.csv").exists()
        assert (tmp_path / "design-b.csv").exists()

    def test_the_written_map_reloads_as_the_same_numbers(self, fitted_glm, tmp_path):
        from nltools.data import BrainData

        fitted_glm.model.write(tmp_path)

        reloaded = BrainData(tmp_path / "betas.nii.gz", mask=fitted_glm.mask)
        np.testing.assert_allclose(
            reloaded.data, fitted_glm.model.betas.data, atol=1e-5
        )


class TestContrastResultWrite:
    """`ContrastResult.write` uses the same layout with contrast suffixes."""

    def test_writes_every_statistic_and_a_sidecar(self, fitted_glm, tmp_path):
        import json

        result = fitted_glm.compute_contrasts("intercept - cond", inference=True)

        written = result.write(tmp_path / "c", prefix="stim")

        assert sorted(path.name for path in written) == [
            "stim_contrast.json",
            "stim_effect.nii.gz",
            "stim_p.nii.gz",
            "stim_se.nii.gz",
            "stim_t.nii.gz",
            "stim_variance.nii.gz",
            "stim_z.nii.gz",
        ]
        sidecar = json.loads((tmp_path / "c" / "stim_contrast.json").read_text())
        assert sidecar["kind"] == "contrast"
        assert sidecar["degrees_of_freedom"] == pytest.approx(result.degrees_of_freedom)

    def test_a_non_brain_contrast_says_so(self):
        from nltools.models.results import ContrastResult

        result = ContrastResult(
            effect=np.zeros(3),
            variance=np.zeros(3),
            standard_error=np.zeros(3),
            statistic=np.zeros(3),
            z_score=np.zeros(3),
            p_value=np.zeros(3),
            degrees_of_freedom=7.0,
        )
        with pytest.raises(TypeError, match="brain"):
            result.write("nowhere")
