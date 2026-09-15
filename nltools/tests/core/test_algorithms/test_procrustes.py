"""Tests for nltools.algorithms.alignment.procrustes — data alignment and Procrustes."""

import nibabel as nib
import numpy as np
import pytest

from nltools.algorithms.alignment.procrustes import (
    align,
    procrustes,
    procrustes_distance,
    align_states,
)
from nltools.data import BrainData
from nltools.data.simulator import Simulator
from nltools.mask import create_sphere


# ==========================================================================
# Shared transformation-matrix orientation: `transformed = original @ T`
# ==========================================================================

N_IMAGES = 6
N_VOXELS = 4


def _seeded_brain(seed, n_voxels=N_VOXELS):
    """A tiny deterministic BrainData: `N_IMAGES` images over `n_voxels` voxels."""
    rng = np.random.default_rng(seed)
    spatial_shape = (2, 2, 2)
    mask_values = np.zeros(spatial_shape, dtype=np.float32)
    mask_values.flat[:n_voxels] = 1.0
    values = rng.standard_normal((N_IMAGES, n_voxels))
    volume = np.zeros(spatial_shape + (N_IMAGES,))
    for image in range(N_IMAGES):
        volume.reshape(-1, N_IMAGES)[:n_voxels, image] = values[image]
    affine = np.eye(4)
    return BrainData(
        nib.Nifti1Image(volume, affine),
        mask=nib.Nifti1Image(mask_values, affine),
    )


def _standardized(values):
    """Center each column and scale to unit Frobenius norm, as procrustes does."""
    centered = values - values.mean(axis=0)
    return centered / np.linalg.norm(centered)


class TestAlign:
    """Test hyperalignment algorithms (SRM, Procrustes)."""

    def test_mixed_types_raises(self):
        """A list mixing types must raise a clear ValueError (F137).

        The same-type guard previously used ``all(type(x) for x in data)``,
        which is always truthy and never triggered.
        """
        with pytest.raises(ValueError, match="same type"):
            align([np.zeros((10, 5)), [[1, 2], [3, 4]]])

    def test_unknown_keyword_raises_type_error(self):
        """3by0: an unknown keyword must raise TypeError, never be swallowed."""
        data = [np.random.randn(30, 5), np.random.randn(30, 5)]
        with pytest.raises(TypeError):
            align(data, method="deterministic_srm", bogus_kwarg=1)

    def test_default_n_features_is_the_smallest_subjects_voxel_count(self):
        """Ragged subjects resolve `n_features=None` to the smallest voxel count.

        The default read subject 0's voxel count, so a group whose first subject
        was not the narrowest asked for more features than another subject had.
        """
        rng = np.random.default_rng(0)
        data = [rng.standard_normal((20, n_voxels)) for n_voxels in (6, 4, 5)]

        out = align(data, method="deterministic_srm", n_features=None)

        assert out["common_model"].shape[1] == 4

    def test_isc_is_reported_for_a_single_aligned_unit(self):
        """One aligned unit is still a unit, not a scalar.

        The ISC stack used to be a single matrix whose mean collapsed to a
        scalar, and building the result dict from it raised.
        """
        rng = np.random.default_rng(0)
        data = [rng.standard_normal((20, 1)) for _ in range(3)]

        out = align(data, method="deterministic_srm")

        assert list(out["isc"]) == [0]

    def test_isc_is_the_mean_pairwise_correlation_of_each_unit(self):
        """ISC per voxel is the mean over the strict upper triangle."""
        rng = np.random.default_rng(1)
        data = [rng.standard_normal((20, 3)) for _ in range(3)]

        out = align(data, method="procrustes")

        for voxel, value in out["isc"].items():
            timecourses = [subject[voxel] for subject in out["transformed"]]
            pairs = [
                np.corrcoef(timecourses[i], timecourses[j])[0, 1]
                for i, j in [(0, 1), (0, 2), (1, 2)]
            ]
            np.testing.assert_allclose(value, np.mean(pairs))

    @pytest.fixture
    def simulated_brains(self):
        """Create simulated BrainData for alignment tests."""
        sim = Simulator()
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data([0, 1], 1, reps=10, output_dir=None).apply_mask(s1)
        d2 = sim.create_data([0, 1], 2, reps=10, output_dir=None).apply_mask(s1)
        d3 = sim.create_data([0, 1], 3, reps=10, output_dir=None).apply_mask(s1)
        return d1, d2, d3

    @pytest.mark.slow
    def test_deterministic_srm_numpy(self, simulated_brains):
        """Deterministic SRM on numpy arrays."""
        d1, d2, d3 = simulated_brains
        data = [d1.data, d2.data, d3.data]
        out = align(data, method="deterministic_srm")
        assert len(data) == len(out["transformed"])
        assert len(data) == len(out["transformation_matrix"])
        assert data[0].shape == out["common_model"].shape
        transformed = np.dot(data[0], out["transformation_matrix"][0])
        np.testing.assert_almost_equal(
            np.sum(out["transformed"][0] - transformed.T), 0, decimal=3
        )

    @pytest.mark.slow
    def test_braindata_outputs_drop_input_fit_state(self, simulated_brains):
        brains = list(simulated_brains)
        for brain in brains:
            X = np.arange(len(brain), dtype=float).reshape(-1, 1)
            brain.fit(model="ridge", X=X, ridge_alpha=1.0)

        out = align(brains, method="procrustes")

        for result in [*out["transformed"], *out["transformation_matrix"]]:
            assert result.model is None
        assert out["common_model"].model is None
        assert all(brain.model is not None for brain in brains)


class TestProcrustes:
    """Test Procrustes transformation directly."""

    def test_basic(self):
        """Procrustes on two similar matrices should yield small disparity."""
        np.random.seed(42)
        n = 20
        mat1 = np.random.randn(n, 5)
        mat2 = mat1 + np.random.randn(n, 5) * 0.1
        mtx1, mtx2, disparity, R, s = procrustes(mat1, mat2)
        assert disparity < 0.5  # Should be small since matrices are similar
        assert R.shape == (5, 5)  # Rotation matrix


class TestProcrustesDistance:
    """Test Procrustes distance with permutation testing."""

    def test_basic(self):
        """Procrustes distance with permutation test."""
        np.random.seed(42)
        mat1 = np.random.randn(20, 5)
        mat2 = mat1 + np.random.randn(20, 5) * 0.1
        result = procrustes_distance(mat1, mat2, n_permute=100, random_state=42)
        assert "similarity" in result
        assert "p" in result
        assert 0 <= result["p"] <= 1
        assert isinstance(result["similarity"], (float, np.floating))


class TestAlignStates:
    """Test state alignment using Hungarian algorithm."""

    def test_reorder_scrambled_states(self):
        """Align scrambled state columns back to reference."""
        import pandas as pd

        n = 20
        states = pd.DataFrame(
            {
                "State1": np.random.randint(1, 100, n),
                "State2": np.random.randint(1, 100, n),
                "State3": np.random.randint(1, 100, n),
            }
        )
        scramble_index = np.array([2, 0, 1])
        scrambled = states.iloc[:, scramble_index]

        assert np.array_equal(
            align_states(scrambled, states, return_index=True), scramble_index
        )
        assert np.array_equal(
            states.shape, align_states(scrambled, states, return_index=False).shape
        )

    def test_replacement_noise_survives_an_integer_state_map(self):
        """Uniform noise truncates to zero when written into an integer copy.

        The constant column then stays constant, correlation distance is still
        NaN, and the Hungarian solver refuses the matrix — so the option did
        nothing for exactly the inputs it exists to rescue.
        """
        reference = np.array([[0, 1], [0, 2], [0, 3]])
        target = np.array([[0, 1], [0, 2], [0, 3]])

        index = align_states(
            reference, target, return_index=True, replace_zero_variance=True
        )

        assert len(index) == 2
        assert reference.dtype == np.int64
        assert np.array_equal(reference, [[0, 1], [0, 2], [0, 3]])
        assert np.array_equal(target, [[0, 1], [0, 2], [0, 3]])


class TestTransformationMatrixOrientation:
    """Both alignment entry points return `T` with `transformed = original @ T`."""

    def test_braindata_procrustes_rejects_axis_one(self):
        """The axis=1 Procrustes transform has no voxel axis to come back on.

        It spans images on both of its axes, so it cannot be wrapped as a
        `BrainData` on the source's mask. The combination raises instead of
        returning a container whose width does not match its own mask.
        """
        brains = [_seeded_brain(0), _seeded_brain(1)]

        with pytest.raises(ValueError, match="axis=0 only"):
            align(brains, method="procrustes", axis=1)

    def test_braindata_procrustes_rejects_unequal_voxel_counts(self):
        """A padded result has no honest mask to come back on.

        `_hyperalign` zero-pads every subject's feature axis up to the widest
        subject, so a narrower subject's result is wider than its own mask.
        Wrapping it anyway produces an object that only fails later, in
        `to_nifti`.
        """
        narrow = _seeded_brain(0, n_voxels=2)
        wide = _seeded_brain(1, n_voxels=3)

        with pytest.raises(ValueError, match="mask supports 2 voxels"):
            align([narrow, wide], method="procrustes")

    def test_every_path_back_projects_with_the_same_transpose(self):
        """`transformed @ T.T / scale` recovers the standardized input everywhere.

        The two entry points fit different problems — a group template versus a
        pair — so their values cannot be compared to each other. The shared
        contract is the orientation, and this is the invariant that states it.
        """
        brains = [_seeded_brain(0), _seeded_brain(1)]
        arrays = [np.array(brain.data, copy=True) for brain in brains]

        # nltools.algorithms.align, numpy input: transformed is (voxels, images).
        numpy_out = align(arrays, method="procrustes")
        for values, transformed, matrix, scale in zip(
            arrays,
            numpy_out["transformed"],
            numpy_out["transformation_matrix"],
            numpy_out["scale"],
        ):
            np.testing.assert_allclose(
                transformed.T @ matrix.T / scale, _standardized(values), atol=1e-10
            )

        # nltools.algorithms.align, BrainData input: transformed is (images, voxels).
        brain_out = align([_seeded_brain(0), _seeded_brain(1)], method="procrustes")
        for values, transformed, matrix, scale in zip(
            arrays,
            brain_out["transformed"],
            brain_out["transformation_matrix"],
            brain_out["scale"],
        ):
            np.testing.assert_allclose(
                transformed.data @ matrix.data.T / scale,
                _standardized(values),
                atol=1e-10,
            )

        # BrainData.align.
        pair = brains[0].align(brains[1], method="procrustes")
        np.testing.assert_allclose(
            pair["transformed"].data
            @ pair["transformation_matrix"].data.T
            / pair["scale"],
            _standardized(arrays[0]),
            atol=1e-10,
        )
