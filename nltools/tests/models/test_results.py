"""Tests for the shared `ContrastResult` record and the removal of `BaseModel`."""

import dataclasses
import importlib

import numpy as np
import pytest

import nltools.models
from nltools.models import ContrastResult


class TestContrastResultShape:
    """Field names, order, and construction."""

    def test_field_names_and_order(self):
        """Fields match the spec's order exactly."""
        names = tuple(f.name for f in dataclasses.fields(ContrastResult))
        assert names == (
            "effect",
            "variance",
            "standard_error",
            "statistic",
            "z_score",
            "p_value",
            "degrees_of_freedom",
        )

    def test_array_payloads_are_stored_unchanged(self):
        """Array payloads round-trip without conversion."""
        effect = np.array([1.0, -2.0])
        result = ContrastResult(
            effect=effect,
            variance=np.array([1.0, 1.0]),
            standard_error=np.array([1.0, 1.0]),
            statistic=np.array([1.0, -2.0]),
            z_score=np.array([1.0, -2.0]),
            p_value=np.array([0.2, 0.8]),
            degrees_of_freedom=np.array([10.0, 10.0]),
        )

        assert np.array_equal(result.effect, effect)


class TestContrastResultIsFrozen:
    """Fields cannot be rebound."""

    @pytest.fixture()
    def result(self):
        return ContrastResult(0.5, 0.25, 0.5, 1.0, 1.0, 0.16, 12.0)

    def test_rebinding_a_field_raises(self, result):
        """Assigning to a field raises `FrozenInstanceError`."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.effect = 1.0


class TestContrastResultIsGeneric:
    """The payload type parameter is usable by callers and type checkers."""

    def test_subscripted_alias_constructs_instances(self):
        """A subscripted alias builds ordinary `ContrastResult` instances."""
        result = ContrastResult[float](0.5, 0.25, 0.5, 1.0, 1.0, 0.16, 12.0)

        assert isinstance(result, ContrastResult)


class TestModelsPackageSurface:
    """`ContrastResult` is reachable from `nltools.data` and `nltools.models`."""

    def test_exported_from_data_namespace(self):
        """`ContrastResult` is one of the result records `nltools.data` advertises."""
        from nltools.data import ContrastResult as DataContrastResult

        assert DataContrastResult is ContrastResult
        assert nltools.models.ContrastResult is ContrastResult

    def test_lives_in_the_results_module(self):
        """The class is defined in `nltools.models.results`."""
        results = importlib.import_module("nltools.models.results")

        assert results.ContrastResult is ContrastResult
