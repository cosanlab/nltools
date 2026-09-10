"""Approved Adjacency shape, metadata, and ownership contracts."""

from copy import copy, deepcopy

import numpy as np
import polars as pl
import pytest

from nltools.data import Adjacency


@pytest.mark.parametrize("data", [None, [], np.empty((0, 0))])
def test_empty_constructor(data):
    adj = Adjacency(data)
    assert (adj.shape, adj.data.shape, len(adj), adj.is_empty) == (
        (0, 0),
        (0,),
        0,
        True,
    )


@pytest.mark.parametrize(
    "kind,edges", [("distance", 1), ("similarity", 1), ("directed", 4)]
)
def test_selection_preserves_matrix_rank_and_schema(kind, edges):
    adj = Adjacency(
        np.arange(3 * edges).reshape(3, edges),
        matrix_type=kind + "_flat",
        labels=["a", "b"],
        Y=pl.DataFrame({"id": [1, 2, 3]}),
    )
    for index in [slice(1, 2), [1], [False, True, False], (1,)]:
        selected = adj[index]
        assert selected.shape == (1, 2, 2)
        assert selected.data.shape == (1, edges)
        assert selected.labels == ["a", "b"]
        assert selected.Y["id"].to_list() == [2]
    assert adj[1].shape == (2, 2)
    assert adj[1][0].shape == (2, 2)
    assert len(list(adj[1])) == 1
    empty = adj[[]]
    assert empty.shape == (0, 2, 2)
    assert empty.data.shape == (0, edges)
    assert empty.is_empty


def test_one_node_and_singleton_list_are_not_empty():
    single = Adjacency(np.array([]))
    assert single.shape == (1, 1)
    assert len(single) == 1 and not single.is_empty
    stack = Adjacency([np.zeros((1, 1))])
    assert stack.shape == (1, 1, 1)
    assert stack.data.shape == (1, 0)
    assert stack.squareform()[0].shape == (1, 1)


@pytest.mark.parametrize(
    "data,kind",
    [
        (np.arange(2), "distance_flat"),
        (np.arange(3), "directed_flat"),
        (np.ones((2, 3)), None),
        (np.ones((1, 2, 2)), None),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "distance"),
        (np.array([[1.0, 1.0], [2.0, 1.0]]), "similarity"),
        (np.diag([-1.0, 1.0]), None),
        (np.diag([0.0, 2.0]), None),
    ],
)
def test_malformed_or_ambiguous_input_is_rejected(data, kind):
    with pytest.raises(ValueError):
        Adjacency(data, matrix_type=kind)


def test_nan_symmetry_and_zero_diagonal_policy():
    matrix = np.array([[1.0, np.nan, 2.0], [np.nan, 1.0, 3.0], [2.0, 3.0, 1.0]])
    adj = Adjacency(matrix)
    assert adj.matrix_type == "similarity"
    assert np.isnan(adj.data[0])
    np.testing.assert_array_equal(np.diag(adj.squareform()), 0)
    matrix[1, 0] = 0
    with pytest.raises(ValueError):
        Adjacency(matrix, matrix_type="similarity")


def test_labels_are_structural_and_y_rows_are_validated():
    values = np.ones((2, 1))
    shared = Adjacency(values, matrix_type="distance_flat", labels=["a", "b"])
    assert shared.labels == ["a", "b"]
    nested = Adjacency(
        values, matrix_type="distance_flat", labels=[["a", "b"], ["c", "d"]]
    )
    assert nested[1].labels == ["c", "d"]
    assert nested[[1]].labels == [["c", "d"]]
    for labels in [["a"], [["a"], ["b"]]]:
        with pytest.raises(ValueError):
            Adjacency(values, matrix_type="distance_flat", labels=labels)
    with pytest.raises(ValueError):
        Adjacency(np.ones(1), Y=pl.DataFrame({"id": [1, 2]}))
    with pytest.raises(ValueError):
        shared.Y = pl.DataFrame({"id": [1]})


def test_owned_construction_copy_selection_and_directed_export():
    values = np.arange(4.0)
    cell = {"items": []}
    labels = ["a", "b"]
    original = Adjacency(
        values,
        matrix_type="directed_flat",
        labels=labels,
        Y=pl.DataFrame({"info": pl.Series([cell], dtype=pl.Object)}),
    )
    values[0] = 99
    labels[0] = "changed"
    cell["items"].append(1)
    assert original.data[0] == 0 and original.labels[0] == "a"
    assert original.Y["info"][0]["items"] == []
    original.alias = original.Y["info"][0]
    original.cycle = original
    for result in [
        copy(original),
        deepcopy(original),
        original.copy(),
        Adjacency(original),
        original[0],
        original + 1,
    ]:
        assert result.cycle is result
        assert result.alias is result.Y["info"][0]
        assert result.alias is not original.alias
        result.data[0] = 42
        assert original.data[0] == 0
    square = original.squareform()
    square[0, 0] = 33
    assert original.data[0] == 0


def test_append_compatibility_labels_and_y_union():
    a = Adjacency(np.ones(1), labels=["a", "b"], Y=pl.DataFrame({"left": [1]}))
    b = Adjacency(np.ones(1), labels=["c", "d"], Y=pl.DataFrame({"right": [2]}))
    out = a.append(b)
    assert out.labels == [["a", "b"], ["c", "d"]]
    assert out.Y.to_dict(as_series=False) == {"left": [1, None], "right": [None, 2]}
    assert a.append(Adjacency()).shape == a.shape
    assert a.append(a[[]]).shape == a.shape
    with pytest.raises(ValueError):
        a.append(Adjacency(np.ones(1), matrix_type="similarity_flat", labels=a.labels))
    with pytest.raises(ValueError):
        a.append(Adjacency(np.ones(1)))
    with pytest.raises(ValueError):
        a + b
    with pytest.raises(ValueError):
        a + a[[0]]


def test_reductions_and_transforms_metadata():
    a = Adjacency(
        np.arange(6.0).reshape(2, 3),
        matrix_type="distance_flat",
        labels=["a", "b", "c"],
        Y=pl.DataFrame({"id": [1, 2]}),
    )
    for method in ["mean", "median", "std", "sum"]:
        reduced = getattr(a, method)()
        assert reduced.shape == (3, 3) and reduced.labels == a.labels
        assert reduced.Y.shape == (0, 0)
        with pytest.raises(ValueError):
            getattr(a[0], method)(axis=2)
    for metric in ["correlation", "euclidean"]:
        transformed = a.distance_to_similarity(metric=metric)
        assert transformed.shape == a.shape and transformed.labels == a.labels
        assert transformed.Y.equals(a.Y) and transformed.matrix_type == "similarity"
    assert a.distance(metric="euclidean").labels == []
    assert a[0].distance(metric="euclidean").shape == (1, 1)


@pytest.mark.parametrize("kind,edges", [("distance", 3), ("directed", 9)])
@pytest.mark.parametrize("selection", [0, [0], []])
def test_hdf_roundtrip_shape_labels_and_y(tmp_path, kind, edges, selection):
    adj = Adjacency(
        np.arange(2 * edges).reshape(2, edges),
        matrix_type=kind + "_flat",
        labels=[["a", "b", "c"], ["d", "e", "f"]],
        Y=pl.DataFrame({"id": [1, 2]}),
    )[selection]
    path = tmp_path / "matrix.h5"
    adj.write(path)
    loaded = Adjacency(path)
    assert loaded.shape == adj.shape
    assert loaded.labels == adj.labels
    assert loaded.matrix_type == adj.matrix_type
    assert loaded.Y.equals(adj.Y)
    np.testing.assert_array_equal(loaded.data, adj.data)


def test_empty_and_numeric_labels_hdf_roundtrip(tmp_path):
    for adj in [
        Adjacency(),
        Adjacency(np.ones(3), labels=[1, 2, 3]),
        Adjacency(np.array([])),
    ]:
        path = tmp_path / "empty.h5"
        adj.write(path)
        loaded = Adjacency(path)
        assert (loaded.shape, loaded.labels) == (adj.shape, adj.labels)


@pytest.mark.parametrize("predictors", [1, 2])
@pytest.mark.parametrize("tail", [1, 2])
def test_regression_axes_and_rss_reference(predictors, tail):
    from scipy.stats import t as t_dist
    from nltools.data import DesignMatrix

    rng = np.random.default_rng(13)
    design = np.column_stack([np.ones(8), np.arange(8)])[:, :predictors]
    values = rng.normal(size=(8, 6))
    adj = Adjacency(
        values,
        matrix_type="distance_flat",
        labels=list("abcd"),
        Y=pl.DataFrame({"id": range(8)}),
    )
    output = adj.regress(DesignMatrix(design), tail=tail)
    beta = np.linalg.pinv(design) @ values
    residual = values - design @ beta
    df = 8 - predictors
    se = np.sqrt(np.diag(np.linalg.pinv(design.T @ design)))[:, None] * np.sqrt(
        np.sum(residual**2, axis=0) / df
    )
    t = beta / se
    p = 1 - t_dist.cdf(t, df) if tail == 1 else 2 * (1 - t_dist.cdf(np.abs(t), df))
    assert output["df"] == df
    for key, expected in [("beta", beta), ("sigma", se), ("t", t), ("p", p)]:
        result = output[key]
        assert result.shape == ((4, 4) if predictors == 1 else (predictors, 4, 4))
        np.testing.assert_allclose(
            result.data, expected[0] if predictors == 1 else expected
        )
        assert result.labels == adj.labels and result.Y.shape == (0, 0)
        assert np.shape(result.squareform()) == result.shape
    assert output["residual"].shape == adj.shape
    np.testing.assert_allclose(output["residual"].data, residual)
    assert output["residual"].Y.equals(adj.Y)


@pytest.mark.parametrize("predictors", [1, 2])
def test_edge_regression_returns_native_predictor_values(predictors):
    rng = np.random.default_rng(12)
    design = rng.normal(size=(6, predictors))
    values = rng.normal(size=6)
    adj = Adjacency(values, labels=list("abcd"))
    X = Adjacency(design.T, matrix_type="distance_flat", labels=adj.labels)
    if predictors == 1:
        X = X[0]
    output = adj.regress(X)
    expected = np.linalg.pinv(design) @ values
    np.testing.assert_allclose(output["beta"], expected.squeeze())
    for key in ["beta", "sigma", "t", "p"]:
        assert not isinstance(output[key], Adjacency)
        assert np.shape(output[key]) == (() if predictors == 1 else (predictors,))
    assert output["df"] == 6 - predictors
    assert output["residual"].shape == adj.shape
    np.testing.assert_allclose(output["residual"].data, values - design @ expected)
    with pytest.raises(ValueError, match="single response"):
        adj[[0]].regress(X)


@pytest.mark.filterwarnings("ignore:n_samples=:UserWarning")
@pytest.mark.filterwarnings("ignore:Only .* samples available:UserWarning")
def test_bootstrap_maps_use_single_matrix_metadata():
    adj = Adjacency(
        np.arange(24.0).reshape(8, 3),
        matrix_type="distance_flat",
        labels=["a", "b", "c"],
        Y=pl.DataFrame({"id": range(8)}),
    )
    output = adj.bootstrap("mean", n_samples=20, n_jobs=1, random_state=4)
    for field in ("estimate", "standard_error", "ci_lower", "ci_upper"):
        result = getattr(output, field)
        assert result.shape == (3, 3) and result.data.shape == (3,)
        assert result.labels == adj.labels and result.Y.shape == (0, 0)


@pytest.mark.parametrize(
    "values", [np.array(["1", "2", "3"]), np.array([{}, {}, {}], dtype=object)]
)
def test_nonnumeric_storage_rejected_at_ingress(values):
    with pytest.raises(ValueError, match="numeric"):
        Adjacency(values, matrix_type="distance_flat")


def test_nullable_numeric_pandas_and_boolean_input():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "a": pd.Series([0, 1, pd.NA], dtype="Float64"),
            "b": pd.Series([1, 0, 2], dtype="Float64"),
            "c": pd.Series([pd.NA, 2, 0], dtype="Float64"),
        }
    )
    result = Adjacency(frame, matrix_type="distance")
    np.testing.assert_equal(result.data, [1, np.nan, 2])
    assert Adjacency(np.eye(2, dtype=bool)).matrix_type == "similarity"


def test_constructor_and_y_setter_detach_numpy_backed_frame():
    values = np.array([1, 2])
    frame = pl.DataFrame({"id": values})
    adj = Adjacency(np.ones((2, 3)), matrix_type="distance_flat", Y=frame)
    second = Adjacency(adj)
    second.Y = frame
    values[0] = 99
    assert adj.Y["id"].to_list() == [1, 2]
    assert second.Y["id"].to_list() == [1, 2]


def test_list_of_hdf_paths_not_supported(tmp_path):
    path = tmp_path / "matrix.h5"
    Adjacency(np.ones(3)).write(path)
    with pytest.raises(ValueError, match="HDF"):
        Adjacency([path])


def test_nested_label_rows_preserve_aliases_with_y_object_cells():
    row = ["a", "b"]
    labels = [row, row]
    Y = pl.DataFrame({"info": pl.Series([row, row], dtype=pl.Object)})
    adj = Adjacency(np.ones((2, 1)), matrix_type="distance_flat", labels=labels, Y=Y)
    assert adj.labels[0] is adj.labels[1]
    assert adj.labels[0] is adj.Y["info"][0]
    assert adj.labels[0] is not row
    adj.labels[0][0] = "changed"
    assert row[0] == "a"
    assert adj.Y["info"][1][0] == "changed"


def test_copy_matrix_type_confirms_case_normalized_kind():
    adj = Adjacency(np.ones(3))
    assert Adjacency(adj, matrix_type="DISTANCE_FLAT").matrix_type == "distance"
    with pytest.raises(ValueError, match="reinterpret"):
        Adjacency(adj, matrix_type="similarity")


def test_symmetry_allows_only_numerical_roundoff():
    matrix = np.array([[1.0, 0.2], [0.2 + 1e-16, 1.0]])
    adj = Adjacency(matrix, matrix_type="similarity")
    np.testing.assert_allclose(adj.data, [0.2])
    matrix[1, 0] += 1e-5
    with pytest.raises(ValueError, match="symmetric"):
        Adjacency(matrix, matrix_type="similarity")


def test_integer_symmetry_is_exact_even_for_large_weights():
    matrix = np.array([[0, 2**60], [2**60 + 1, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="symmetric"):
        Adjacency(matrix, matrix_type="distance")


def test_integer_dataframe_preserves_exact_weights():
    import pandas as pd

    matrix = np.array([[0, 2**60 + 1], [2**60 + 1, 0]], dtype=np.int64)
    adj = Adjacency(pd.DataFrame(matrix), matrix_type="distance")
    assert adj.data.dtype == np.int64
    assert adj.data[0] == 2**60 + 1
