"""The consolidated `nltools.algorithms` public surface (v0.6.0).

`nltools.stats` is gone: every user-facing statistical function is importable
flat from `nltools.algorithms` (organized into submodules underneath). The
permutation-test entry points are the engine functions themselves — no wrapper
layer — so facade/engine drift is structurally impossible.
"""

import importlib

import pytest

import nltools
import nltools.algorithms as algorithms

# Everything the old nltools.stats exported, now expected flat on algorithms.
CONSOLIDATED_NAMES = [
    # corrections
    "fdr",
    "holm_bonf",
    "threshold",
    "multi_threshold",
    # outliers
    "zscore",
    "winsorize",
    "trim",
    "find_spikes",
    # signal
    "downsample",
    "upsample",
    "calc_bpm",
    "make_cosine_basis",
    # similarity
    "fisher_r_to_z",
    "fisher_z_to_r",
    "compute_similarity",
    "compute_multivariate_similarity",
    "transform_pairwise",
    # alignment (procrustes family)
    "align",
    "procrustes",
    "procrustes_distance",
    "align_states",
    # intersubject
    "isc",
    "isc_group",
    "isfc",
    "isps",
    # regression
    "regress",
    # permutation / inference
    "one_sample_permutation_test",
    "two_sample_permutation_test",
    "correlation_permutation_test",
    "timeseries_correlation_permutation_test",
    "circle_shift",
    "phase_randomize",
    "matrix_permutation_test",
    "double_center",
    "u_center",
    "distance_correlation",
]

# The permutation entry points must be the engine functions themselves.
ENGINE_IDENTITY = {
    "one_sample_permutation_test": "nltools.algorithms.inference.one_sample",
    "two_sample_permutation_test": "nltools.algorithms.inference.two_sample",
    "correlation_permutation_test": "nltools.algorithms.inference.correlation",
    "timeseries_correlation_permutation_test": "nltools.algorithms.inference.timeseries",
    "circle_shift": "nltools.algorithms.inference.timeseries",
    "phase_randomize": "nltools.algorithms.inference.timeseries",
    "matrix_permutation_test": "nltools.algorithms.inference.matrix",
    "double_center": "nltools.algorithms.inference.matrix",
    "u_center": "nltools.algorithms.inference.matrix",
    "distance_correlation": "nltools.algorithms.inference.matrix",
}


@pytest.mark.parametrize("name", CONSOLIDATED_NAMES)
def test_importable_flat_from_algorithms(name):
    assert hasattr(algorithms, name), f"nltools.algorithms lacks {name}"
    assert name in algorithms.__all__, f"{name} missing from algorithms.__all__"


def test_stats_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nltools.stats")
    assert "stats" not in nltools.__all__


def test_algorithms_bound_on_top_level_package():
    assert "algorithms" in nltools.__all__
    assert nltools.algorithms is algorithms


@pytest.mark.parametrize("name", sorted(ENGINE_IDENTITY))
def test_permutation_functions_are_engine_functions(name):
    engine_mod = importlib.import_module(ENGINE_IDENTITY[name])
    assert getattr(algorithms, name) is getattr(engine_mod, name), (
        f"algorithms.{name} is a wrapper, not the engine function — "
        "the v0.6.0 consolidation removed the wrapper layer"
    )
