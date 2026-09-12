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
from nltools.algorithms.alignment.srm import SRM, DetSRM
from nltools.data import BrainData
from nltools.data.simulator import Simulator
from nltools.mask import create_sphere


# ==========================================================================
# Shared transformation-matrix orientation: `transformed = original @ T`
# ==========================================================================
#
# Values pinned from the implementation on a seeded fixture. Every one of them
# is byte-identical to what nltools produced before the alignment package was
# pared back, except the transformation matrices, whose orientation this
# convention flips to `R.T`.

N_IMAGES = 6
N_VOXELS = 4


def _seeded_brain(seed):
    """A tiny deterministic BrainData: `N_IMAGES` images over `N_VOXELS` voxels."""
    rng = np.random.default_rng(seed)
    spatial_shape = (2, 2, 2)
    mask_values = np.zeros(spatial_shape, dtype=np.float32)
    mask_values.flat[:N_VOXELS] = 1.0
    values = rng.standard_normal((N_IMAGES, N_VOXELS))
    volume = np.zeros(spatial_shape + (N_IMAGES,))
    for image in range(N_IMAGES):
        volume.reshape(-1, N_IMAGES)[:N_VOXELS, image] = values[image]
    affine = np.eye(4)
    return BrainData(
        nib.Nifti1Image(volume, affine),
        mask=nib.Nifti1Image(mask_values, affine),
    )


def _standardized(values):
    """Center each column and scale to unit Frobenius norm, as procrustes does."""
    centered = values - values.mean(axis=0)
    return centered / np.linalg.norm(centered)


GROUP_TRANSFORMED_0 = np.array(
    [
        [
            0.202115440386422,
            -0.024426811367380598,
            0.16701571426537384,
            -0.046756959438512644,
        ],
        [
            0.0372789875037897,
            0.09861490805851418,
            0.33239492662574605,
            0.16313446556740202,
        ],
        [
            -0.004607008329685968,
            -0.3068761676084217,
            -0.1479275536641137,
            -0.06260114117374765,
        ],
        [
            -0.408672278428174,
            -0.046031216410223744,
            -0.3031033210329042,
            -0.2553989328057423,
        ],
        [
            0.035138253209019595,
            -0.07033264350332658,
            0.10999529747303279,
            0.1869184827990601,
        ],
        [
            0.13874660565862865,
            0.3490519308308384,
            -0.15837506366713466,
            0.014704085051540524,
        ],
    ]
)
GROUP_TRANSFORMED_1 = np.array(
    [
        [
            0.399295080179628,
            -0.21645559630463254,
            0.1957500673943978,
            -0.13037872450027133,
        ],
        [
            -0.12379111509757428,
            0.0801635862359363,
            0.240785113726937,
            0.1326630537605983,
        ],
        [
            -0.04349428699445653,
            0.11356989344460851,
            0.05302600138871258,
            0.009814968408371526,
        ],
        [
            -0.249330081438624,
            -0.09369943864438902,
            -0.18718478832211968,
            -0.09136444959770978,
        ],
        [
            -0.07523933419300649,
            -0.2635064386695052,
            -0.017113346334753057,
            0.11630597471752045,
        ],
        [
            0.09255973754403324,
            0.3799279939379819,
            -0.2852630478531747,
            -0.037040822788509176,
        ],
    ]
)
GROUP_TM_0 = np.array(
    [
        [
            1.0000000000000004,
            -2.0940299388870104e-16,
            5.779778459619224e-17,
            -1.752572733123854e-16,
        ],
        [
            4.433472630196136e-17,
            1.0000000000000002,
            -1.302285369067071e-16,
            6.819725312138955e-17,
        ],
        [
            -4.0434251063678984e-16,
            -3.4527837053483735e-17,
            1.0000000000000002,
            -6.35126452025028e-17,
        ],
        [
            -2.3398667233433565e-16,
            -8.106484350416212e-18,
            2.530600593271529e-16,
            1.0000000000000002,
        ],
    ]
)
GROUP_TM_1 = np.array(
    [
        [
            0.2622827706681766,
            0.32585428401300665,
            0.5834457514582717,
            0.6961449482013701,
        ],
        [
            0.01823238368294549,
            0.14214357318645882,
            0.7211433700042523,
            -0.6778016115988023,
        ],
        [
            0.6852961349189668,
            0.5899591654554207,
            -0.35566895505342766,
            -0.23625618504384518,
        ],
        [
            -0.6791498625936606,
            0.7249568078770625,
            -0.11420547016411109,
            0.012255669557654303,
        ],
    ]
)
GROUP_COMMON_MODEL = np.array(
    [
        [
            0.27960030068720815,
            -0.10664028509994222,
            0.17291944985862456,
            -0.08114332145431949,
        ],
        [
            -0.033942699191979966,
            0.08676095514785606,
            0.2796986238023959,
            0.14354742089503866,
        ],
        [
            -0.021269158725660833,
            -0.11275604452734803,
            -0.055098054576048605,
            -0.02878487607022655,
        ],
        [
            -0.32365263865542765,
            -0.06487405046075811,
            -0.24102585340075844,
            -0.17435024710223745,
        ],
        [
            -0.014043991310383804,
            -0.15124404614128217,
            0.05063469203872341,
            0.14901425374368843,
        ],
        [
            0.11330818719624407,
            0.3487534710814745,
            -0.2071288577229368,
            -0.008283230011943542,
        ],
    ]
)
GROUP_DISPARITY = [0.11442493638653858, 0.1844615078332206]
GROUP_SCALE = [0.9410499793387493, 0.9030716982426035]

PAIR_TRANSFORMED = np.array(
    [
        [
            0.08231588874849514,
            0.1140605320913914,
            0.056707469371972984,
            -0.1307376044091879,
        ],
        [
            0.2616002118119451,
            0.10768290890997564,
            -0.05468282346497337,
            0.007645255488618097,
        ],
        [
            -0.17301619947256253,
            -0.0808211231833835,
            -0.08744561899301914,
            -0.152145596946093,
        ],
        [
            -0.356994199973175,
            -0.04452094099981342,
            -0.10412311887539391,
            0.20638720468410507,
        ],
        [
            0.1352105746390726,
            -0.04247155272599868,
            -0.07538979877487069,
            -0.06373098710690941,
        ],
        [
            0.05088372424622478,
            -0.05392982409217144,
            0.2649338907362841,
            0.13258172828946713,
        ],
    ]
)
PAIR_TM = np.array(
    [
        [
            0.26228277066817673,
            0.018232383682945507,
            0.6852961349189666,
            -0.6791498625936604,
        ],
        [
            0.32585428401300665,
            0.14214357318645823,
            0.5899591654554206,
            0.7249568078770627,
        ],
        [
            0.5834457514582718,
            0.7211433700042524,
            -0.3556689550534281,
            -0.11420547016411049,
        ],
        [
            0.6961449482013696,
            -0.6778016115988025,
            -0.2362561850438452,
            0.012255669557654371,
        ],
    ]
)
PAIR_COMMON_MODEL = np.array(
    [
        [
            0.345584192064786,
            0.8216181435011584,
            0.33043707618338714,
            -1.303157231604361,
        ],
        [
            0.9053558666731177,
            0.4463745723640113,
            -0.5369532353602852,
            0.5811181041963531,
        ],
        [
            0.36457239618607573,
            0.294132496655526,
            0.02842224131579679,
            0.5467129866124469,
        ],
        [
            -0.7364540870016669,
            -0.16290994799305278,
            -0.48211931267997826,
            0.5988462126346276,
        ],
        [
            0.03972210748165899,
            -0.2924567509650886,
            -0.7819084623568421,
            -0.2571922406188707,
        ],
        [
            0.008142180518343508,
            -0.2756029052993704,
            1.2940638143982073,
            1.0067243153057943,
        ],
    ]
)
PAIR_DISPARITY = 0.5036048838382408
PAIR_SCALE = 0.7045531322489167


class TestAlign:
    """Test hyperalignment algorithms (SRM, Procrustes)."""

    def test_mixed_types_raises(self):
        """A list mixing types must raise a clear ValueError (F137).

        The same-type guard previously used ``all(type(x) for x in data)``,
        which is always truthy and never triggered.
        """
        with pytest.raises(ValueError, match="same type"):
            align([np.zeros((10, 5)), [[1, 2], [3, 4]]])

    def test_n_iter_and_random_state_reach_deterministic_srm(self, monkeypatch):
        """3by0: n_iter/random_state on align() must reach the constructed DetSRM."""
        captured = {}
        real_init = DetSRM.__init__

        def spy_init(self, **kwargs):
            captured.update(kwargs)
            real_init(self, **kwargs)

        monkeypatch.setattr(DetSRM, "__init__", spy_init)
        data = [np.random.randn(30, 5), np.random.randn(30, 5)]
        align(data, method="deterministic_srm", n_iter=3, random_state=11)
        assert captured["n_iter"] == 3
        assert captured["random_state"] == 11

    def test_n_iter_and_random_state_reach_probabilistic_srm(self, monkeypatch):
        """3by0: n_iter/random_state on align() must reach the constructed SRM."""
        captured = {}
        real_init = SRM.__init__

        def spy_init(self, **kwargs):
            captured.update(kwargs)
            real_init(self, **kwargs)

        monkeypatch.setattr(SRM, "__init__", spy_init)
        data = [np.random.randn(30, 5), np.random.randn(30, 5)]
        align(data, method="probabilistic_srm", n_iter=4, random_state=12)
        assert captured["n_iter"] == 4
        assert captured["random_state"] == 12

    def test_unknown_keyword_raises_type_error(self):
        """3by0: an unknown keyword must raise TypeError, never be swallowed."""
        data = [np.random.randn(30, 5), np.random.randn(30, 5)]
        with pytest.raises(TypeError):
            align(data, method="deterministic_srm", bogus_kwarg=1)

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
    def test_probabilistic_srm_numpy(self, simulated_brains):
        """Probabilistic SRM on numpy arrays."""
        d1, d2, d3 = simulated_brains
        data = [d1.data, d2.data, d3.data]
        out = align(data, method="probabilistic_srm")
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
            assert not hasattr(result, "model_")
            assert not hasattr(result, "ridge_weights")
        assert not hasattr(out["common_model"], "model_")
        assert all(hasattr(brain, "model_") for brain in brains)


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

    def test_near_identical_matrices_are_significant(self):
        """F136: near-identical matrices must yield a small p-value.

        The observed statistic and the permutation null must live on the same
        scale. Previously the observed disparity (~0 for similar matrices) was
        compared against a null of similarities (~1), so a near-identical pair
        got p ~ 1 instead of a small p.
        """
        np.random.seed(0)
        mat1 = np.random.randn(20, 5)
        mat2 = mat1 + np.random.randn(20, 5) * 0.01  # essentially identical
        result = procrustes_distance(mat1, mat2, n_permute=500, random_state=42)
        assert result["similarity"] > 0.5, (
            f"near-identical matrices should be highly similar, got "
            f"{result['similarity']}"
        )
        assert result["p"] < 0.05, (
            f"near-identical matrices should be significant, got p={result['p']}"
        )


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


class TestTransformationMatrixOrientation:
    """Both alignment entry points return `T` with `transformed = original @ T`."""

    def test_seeded_regression(self):
        """Pin the numeric output of both entry points on a seeded fixture."""
        group = align([_seeded_brain(0), _seeded_brain(1)], method="procrustes")

        np.testing.assert_allclose(
            group["transformed"][0].data, GROUP_TRANSFORMED_0, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            group["transformed"][1].data, GROUP_TRANSFORMED_1, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            group["transformation_matrix"][0].data, GROUP_TM_0, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            group["transformation_matrix"][1].data, GROUP_TM_1, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            group["common_model"].data, GROUP_COMMON_MODEL, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(group["disparity"], GROUP_DISPARITY, rtol=1e-10)
        np.testing.assert_allclose(group["scale"], GROUP_SCALE, rtol=1e-10)

        pair = _seeded_brain(0).align(_seeded_brain(1), method="procrustes")

        np.testing.assert_allclose(
            pair["transformed"].data, PAIR_TRANSFORMED, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            pair["transformation_matrix"].data, PAIR_TM, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            pair["common_model"].data, PAIR_COMMON_MODEL, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(pair["disparity"], PAIR_DISPARITY, rtol=1e-10)
        np.testing.assert_allclose(pair["scale"], PAIR_SCALE, rtol=1e-10)

    def test_braindata_procrustes_rejects_axis_one(self):
        """The axis=1 Procrustes transform has no voxel axis to come back on.

        It spans images on both of its axes, so it cannot be wrapped as a
        `BrainData` on the source's mask. The combination raises instead of
        returning a container whose width does not match its own mask.
        """
        brains = [_seeded_brain(0), _seeded_brain(1)]

        with pytest.raises(ValueError, match="axis=0 only"):
            align(brains, method="procrustes", axis=1)

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
