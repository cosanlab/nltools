"""Behavior required by the approved DesignMatrix specification."""

import numpy as np
import polars as pl
import pytest

from nltools.data import DesignMatrix


@pytest.mark.parametrize(
    "kwargs",
    [
        {"TR": 0},
        {"TR": -1},
        {"TR": float("inf")},
        {"sampling_freq": float("nan")},
        {"sampling_freq": -1},
        {"n_rows": True},
        {"n_rows": 1.5},
    ],
)
def test_invalid_constructor_controls(kwargs):
    with pytest.raises(ValueError):
        DesignMatrix(**kwargs)


def test_array_grammar_and_ownership():
    array = np.arange(6).reshape(3, 2)
    dm = DesignMatrix(array, columns=["a", "b"])
    array[0, 0] = 99
    assert dm.item(0, "a") == 0
    assert DesignMatrix(np.empty((4, 0))).shape == (4, 0)
    for array in [np.array(1), np.empty((1, 2, 3))]:
        with pytest.raises(ValueError):
            DesignMatrix(array)


def test_annotations_are_valid_and_detached():
    with pytest.raises(ValueError, match="column"):
        DesignMatrix({"a": [1]}, convolved=["missing"])
    dm = DesignMatrix({"a": [1]}, convolved=["a"], confounds=["a"])
    dm.convolved.clear()
    dm.confounds.clear()
    assert dm.convolved == dm.confounds == ["a"]


def test_direct_polars_results_and_metadata():
    dm = DesignMatrix(
        {"a": [1, 2], "b": [3, 4]}, sampling_freq=2, convolved=["a"], confounds=["b"]
    )
    dm.multi = True
    selected = dm.select("a")
    assert isinstance(selected, DesignMatrix)
    assert selected.convolved == ["a"] and selected.multi
    for result in [dm.select(pl.col("a").mean()), dm.describe(), dm.reverse()]:
        assert isinstance(result, DesignMatrix)
        assert result.sampling_freq is None
        assert result.convolved == result.confounds == []
        assert result.multi is False
    assert isinstance(dm.get_column("a"), pl.Series)
    assert isinstance(dm.lazy(), pl.LazyFrame)
    assert isinstance(dm.group_by("a").agg(pl.col("b").sum()), pl.DataFrame)
    assert dm.head(1).sampling_freq == 2
    renamed = dm.rename({"a": "task", "b": "noise"})
    assert renamed.convolved == ["task"] and renamed.confounds == ["noise"]
    dm.columns = ["task", "noise"]
    assert dm.convolved == ["task"] and dm.confounds == ["noise"]
    replaced = dm.with_columns(pl.col("task") * 2)
    assert replaced.convolved == [] and replaced.confounds == ["noise"]
    dm["task"] = 5
    assert dm.convolved == []


def test_zero_column_rows_selection_assignment_equality():
    dm = DesignMatrix(n_rows=5, sampling_freq=2)
    assert dm.select([]).shape == (5, 0)
    assert dm.head(0).shape == (0, 0)
    assert dm.slice(1, 2).shape == (2, 0)
    assert dm.filter(pl.Series([True, False, True, False, False])).shape == (2, 0)
    assert dm != DesignMatrix(n_rows=3)
    dm["a"] = 2
    assert dm.to_numpy().tolist() == [[2]] * 5


def test_forwarded_mutation_updates_annotations():
    dm = DesignMatrix({"a": [1, 2], "b": [2, 3]}, convolved=["a"], confounds=["b"])
    result = dm.replace_column(0, pl.Series("a", [3, 4]))
    assert result is dm
    assert dm.convolved == [] and dm.confounds == ["b"]
    removed = dm.drop_in_place("b")
    assert isinstance(removed, pl.Series)
    assert dm.confounds == []


def test_numpy_and_series_exports_detached():
    dm = DesignMatrix({"a": [1, 2]})
    array = dm.to_numpy()
    array[0, 0] = 9
    series = dm["a"]
    series[0] = 8
    assert dm.item(0, "a") == 1


def test_pandas_export_and_direct_append_removed():
    import pandas as pd

    dm = DesignMatrix({"a": [1, 2]})
    assert "to_pandas" not in DesignMatrix.__dict__
    with pytest.raises(TypeError):
        dm.append(pd.DataFrame({"b": [3, 4]}), axis=1)


def test_zero_column_text_export_rejected(tmp_path):
    with pytest.raises(ValueError, match="column"):
        DesignMatrix(n_rows=4).write(tmp_path / "empty.tsv")


def test_append_recorded_empty_rows_and_run_numbers():
    empty = DesignMatrix(n_rows=3, sampling_freq=2)
    dm = DesignMatrix({"a": [1, 2]}, sampling_freq=2).add_poly(0)
    with pytest.raises(ValueError, match="rows"):
        dm.append(empty, axis=1)
    combined = empty.append(dm)
    assert combined.shape == (5, 2)
    assert combined.confounds == [".nl_r1_poly_0"]
    assert combined["a"].to_list() == [0, 0, 0, 1, 2]
    assert combined.append(dm).confounds == [".nl_r1_poly_0", ".nl_r2_poly_0"]
    assert empty.append(empty).shape == (6, 0)
    for axis in (0, 1):
        assert DesignMatrix().append(dm, axis=axis) == dm
        assert dm.append(DesignMatrix(), axis=axis) == dm
    zero = DesignMatrix(n_rows=0, sampling_freq=2)
    assert zero.append(dm).confounds == [".nl_r0_poly_0"]


@pytest.mark.parametrize(
    "a,b,fill,duplicate",
    [
        ([2**53], [2**53 + 1], None, False),
        ([float("nan"), 1.0], [float("nan"), 1.0], None, True),
        ([None, 1.0], [0.0, 1.0], 0, True),
    ],
)
def test_append_exact_final_values(a, b, fill, duplicate):
    left = DesignMatrix({"a": a})
    right = DesignMatrix({"b": b})
    if duplicate:
        with pytest.raises(ValueError, match="duplicates"):
            left.append(right, axis=1, fill_na=fill)
    else:
        assert left.append(right, axis=1, fill_na=fill).columns == ["a", "b"]


def test_h5_unicode_and_empty_run_identity(tmp_path):
    dm = DesignMatrix({"刺激": [1, 2]}, sampling_freq=1, convolved=["刺激"])
    path = tmp_path / "unicode.h5"
    dm.write(path)
    restored = DesignMatrix(path)
    assert restored.convolved == ["刺激"]
    empty = DesignMatrix(n_rows=3, sampling_freq=1)
    empty.append(empty).write(path)
    combined = DesignMatrix(path).append(dm.add_poly(0))
    assert combined.confounds == [".nl_r2_poly_0"]


def test_private_pandas_boundary_preserves_columns_and_row_count():
    """`_to_pandas` is the one conversion out of polars, for pandas-only callers."""
    from nltools.data.designmatrix.io import _to_pandas

    dm = DesignMatrix({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    frame = _to_pandas(dm)
    assert list(frame.columns) == ["a", "b"]
    assert frame["a"].tolist() == [1.0, 2.0]
    assert list(frame.index) == [0, 1]


def test_plot_matrix_uses_arrays_with_labels(monkeypatch):
    import matplotlib.pyplot as plt
    import seaborn as sns

    dm = DesignMatrix({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    original = sns.heatmap

    def heatmap(data, **kwargs):
        assert isinstance(data, np.ndarray)
        return original(data, **kwargs)

    monkeypatch.setattr(sns, "heatmap", heatmap)
    fig = dm.plot(columns=["b", "a"])
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == ["b", "a"]
    plt.close(fig)


def test_column_expression_selection_retains_provenance():
    dm = DesignMatrix({"a": [1, 2]}, sampling_freq=1, convolved=["a"])
    selected = dm.select(pl.col("a"))
    assert selected.convolved == ["a"] and selected.sampling_freq == 1
    assert DesignMatrix(n_rows=4).select(pl.all()).shape == (4, 0)


def test_mutators_accept_expressions_and_preserve_untouched_columns():
    dm = DesignMatrix({"a": [1, 2]}, sampling_freq=1, convolved=["a"])
    assert dm.insert_column(1, (pl.col("a") * 2).alias("b")) is dm
    assert dm.convolved == ["a"]
    assert dm.hstack([pl.Series("c", [4, 5])], in_place=True) is dm
    assert dm.convolved == ["a"] and dm.sampling_freq == 1


def test_replace_data_invalidates_reused_convolved_name():
    dm = DesignMatrix({"task": [1, 2]}, convolved=["task"], sampling_freq=1)
    assert dm.replace_data(np.array([[3], [4]]), ["task"]).convolved == []


def test_zero_length_multi_does_not_contribute_run():
    dm = DesignMatrix({"task": [1, 2]}, sampling_freq=1).add_poly(0)
    empty = dm.append(dm).head(0)
    result = empty.append(dm)
    assert result.confounds == [".nl_r0_poly_0"]
    assert result.shape == (2, 3)
    assert ".nl_r1_poly_0" in result.columns
    assert result.append(dm).confounds == [".nl_r0_poly_0", ".nl_r1_poly_0"]


def test_unsized_column_initialization_uses_input_length():
    assert DesignMatrix().with_columns(a=[1, 2, 3]).shape == (3, 1)
    assert DesignMatrix().with_columns(a=1).shape == (1, 1)
    assert DesignMatrix(n_rows=0).with_columns(a=1).shape == (0, 1)


@pytest.mark.parametrize(
    "operation",
    [
        lambda dm: dm.insert_column(0, pl.Series("a", [1, 2])),
        lambda dm: dm.hstack([pl.Series("a", [1, 2])], in_place=True),
        lambda dm: dm.hstack([pl.Series("a", [1, 2])]),
        lambda dm: dm.with_columns(a=[1, 2]),
    ],
)
def test_sized_columnless_addition_validates_height_without_mutation(operation):
    dm = DesignMatrix(n_rows=5)
    with pytest.raises((ValueError, pl.exceptions.ShapeError)):
        operation(dm)
    assert dm.shape == (5, 0)


def test_empty_vertical_inputs_preserve_schema_and_validate_dtypes():
    empty = DesignMatrix(pl.DataFrame(schema={"a": pl.Int64}))
    other = DesignMatrix({"a": [1.0]})
    with pytest.raises(ValueError, match="dtype"):
        empty.append(other)
    combined = empty.append(DesignMatrix(pl.DataFrame(schema={"b": pl.Int64})))
    assert combined.shape == (0, 2)
    assert combined.schema == {"a": pl.Int64, "b": pl.Int64}


def test_rename_callback_matches_native_invocation_count():
    dm = DesignMatrix({"a": [1], "b": [2]}, convolved=["a"], confounds=["b"])
    calls = []
    native_calls = []

    def rename(name, log):
        log.append(name)
        return f"{name}_{len(log)}"

    expected = dm.data.rename(lambda name: rename(name, native_calls))
    result = dm.rename(lambda name: rename(name, calls))
    assert calls == native_calls
    assert result.columns == expected.columns
    assert result.convolved == [expected.columns[0]]
    assert result.confounds == [expected.columns[1]]


def test_all_null_columns_duplicate_without_fill():
    with pytest.raises(ValueError, match="duplicates"):
        DesignMatrix({"a": [None, None]}).append(
            DesignMatrix({"b": [None, None]}), axis=1, fill_na=None
        )
