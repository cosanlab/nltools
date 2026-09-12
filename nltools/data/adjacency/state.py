"""Normalize relation matrices and construct owned, metadata-consistent results."""

from copy import deepcopy
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import squareform

from nltools.data.ownership import _copy_graph, _copy_object_frames, copy_frame
from nltools.data.validation import validate_frame


@dataclass(frozen=True)
class MatrixState:
    """Validated vector storage and its interpretation."""

    data: np.ndarray
    matrix_type: str
    n_nodes: int
    single: bool


def normalize_matrix(data, matrix_type=None):
    """Parse square or explicitly flat values without guessing stack axes."""
    kind = matrix_type.lower() if isinstance(matrix_type, str) else matrix_type
    valid = {"distance", "similarity", "directed"}
    if kind is not None and kind not in valid | {x + "_flat" for x in valid}:
        raise ValueError("Invalid matrix_type.")
    if data is None or isinstance(data, list) and not data:
        return MatrixState(np.array([]), "empty", 0, False)
    from_file = isinstance(data, (str, Path))
    if from_file:
        data = pl.read_csv(data).to_numpy()
    if isinstance(data, pl.DataFrame):
        data = data.to_numpy()
    # Nullable pandas numerics use Object arrays unless explicitly converted.
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
            raise ValueError("Adjacency data must be numeric or boolean.")
        values = data.to_numpy()
        data = (
            data.to_numpy(dtype=float, na_value=np.nan)
            if values.dtype.kind == "O"
            else values
        )
    values = np.asarray(data)
    if values.dtype.kind not in "biufc":
        raise ValueError("Adjacency data must be numeric or boolean.")
    if values.ndim not in (1, 2):
        raise ValueError("Data must be a square matrix or a 1-D/2-D flat array.")
    flat = kind is not None and kind.endswith("_flat")
    if from_file and values.shape[1] == 1 and flat:
        values = values[:, 0]
    if flat or values.ndim == 1 and kind is None:
        kind = kind.removesuffix("_flat") if kind else "distance"
        edges = values.shape[-1]
        if kind == "directed":
            nodes = isqrt(edges)
            valid_length = nodes * nodes == edges
        else:
            nodes = (1 + isqrt(1 + 8 * edges)) // 2
            valid_length = nodes * (nodes - 1) // 2 == edges
        if not valid_length:
            raise ValueError(
                "Flat data length must be triangular or a perfect square for directed matrices."
            )
        if kind == "directed" and nodes == 0 and values.ndim == 1:
            return MatrixState(np.array([]), "empty", 0, False)
        return MatrixState(values.copy(), kind, nodes, values.ndim == 1)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            "Data must be square; rectangular stacks require an explicit flat matrix_type."
        )
    nodes = values.shape[0]
    if nodes == 0:
        return MatrixState(np.array([]), "empty", 0, False)
    # Correlation producers can differ across triangles by floating-point roundoff.
    symmetric = (
        np.allclose(values, values.T, rtol=1e-12, atol=1e-12, equal_nan=True)
        if values.dtype.kind in "fc"
        else np.array_equal(values, values.T)
    )
    if kind is None:
        if not symmetric:
            kind = "directed"
        elif np.all(np.diag(values) == 0):
            kind = "distance"
        elif np.all(np.diag(values) == 1):
            kind = "similarity"
        else:
            raise ValueError(
                "Symmetric matrices with other diagonals require an explicit matrix_type."
            )
    if kind != "directed" and not symmetric:
        raise ValueError(
            "Distance and similarity matrices must be symmetric, including NaN positions."
        )
    vector = values.ravel() if kind == "directed" else values[np.triu_indices(nodes, 1)]
    return MatrixState(vector.copy(), kind, nodes, True)


def labels_are_nested(labels):
    """Recognize a matrix-by-node label grid by its structure."""
    return bool(labels) and isinstance(labels[0], (list, tuple, np.ndarray))


def validate_labels(labels, *, n_nodes, n_matrices, single):
    """Accept shared node labels or a stack's explicit label grid."""
    if labels is None:
        return []
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("labels must be a list or numpy array.")
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
    if not labels:
        return labels
    if labels_are_nested(labels):
        if (
            single
            or len(labels) != n_matrices
            or any(len(row) != n_nodes for row in labels)
        ):
            raise ValueError("Nested labels must have shape (n_matrices, n_nodes).")
        return labels
    if len(labels) != n_nodes:
        raise ValueError("Node labels must match n_nodes.")
    return labels


def owned_frame(value, n_matrices):
    """Validate and detach matrix metadata, including Python Object cells."""
    frame = validate_frame(value, frame_type="Y")
    if frame.width and frame.height != n_matrices:
        raise ValueError(
            f"Y rows ({frame.height}) do not match matrices ({n_matrices})."
        )
    return copy_frame(frame)


def initialize(adj, data, *, matrix_type, labels, Y):
    """Initialize one facade from normalized values or an owned source graph."""
    from . import Adjacency
    from nltools.io import is_h5_path

    if isinstance(data, (str, Path)) and is_h5_path(data):
        from .io import read_h5

        data = read_h5(data)
        matrix_type = None
    if isinstance(data, list) and data:
        if any(isinstance(item, (str, Path)) and is_h5_path(item) for item in data):
            raise ValueError(
                "Lists of HDF paths are not supported; load each Adjacency first."
            )
        members = [
            item
            if isinstance(item, Adjacency)
            else Adjacency(item, matrix_type=matrix_type)
            for item in data
        ]
        data = members[0].copy()
        for member in members[1:]:
            data = append(data, member)
        if data.is_single_matrix:
            data = select(data, [0])
    if isinstance(data, Adjacency):
        if (
            matrix_type is not None
            and matrix_type.lower().removesuffix("_flat") != data.matrix_type
        ):
            raise ValueError("Copy construction cannot reinterpret matrix_type.")
        values = dict(data.__dict__)
        if labels is not None:
            values["labels"] = validate_labels(
                labels,
                n_nodes=data.n_nodes,
                n_matrices=len(data),
                single=data.is_single_matrix,
            )
        if Y is not None:
            values["_Y"] = validate_frame(Y, frame_type="Y")
            if values["_Y"].width and values["_Y"].height != len(data):
                raise ValueError("Y rows must match the number of matrices.")
        memo = {id(data): adj}
        copy_frame(values["_Y"], memo)
        _copy_object_frames(values, memo)
        adj.__dict__.update(deepcopy(values, memo))
        return
    state = normalize_matrix(data, matrix_type)
    adj.data = state.data
    adj.matrix_type = state.matrix_type
    adj._n_nodes = state.n_nodes
    adj.is_single_matrix = state.single
    adj.issymmetric = state.matrix_type in ("distance", "similarity")
    metadata = {
        "labels": validate_labels(
            labels, n_nodes=state.n_nodes, n_matrices=len(adj), single=state.single
        ),
        "_Y": validate_frame(Y, frame_type="Y"),
    }
    if metadata["_Y"].width and metadata["_Y"].height != len(adj):
        raise ValueError("Y rows must match the number of matrices.")
    memo = {}
    copy_frame(metadata["_Y"], memo)
    _copy_object_frames(metadata, memo)
    adj.__dict__.update(deepcopy(metadata, memo))


def result(adj, values, *, labels, Y, matrix_type=None):
    """Build an owned result while retaining aliases in the retained metadata graph."""
    state = normalize_matrix(values, (matrix_type or adj.matrix_type) + "_flat")
    return _copy_graph(
        adj,
        replacements={
            "data": state.data,
            "matrix_type": state.matrix_type,
            "_n_nodes": state.n_nodes,
            "is_single_matrix": state.single,
            "issymmetric": state.matrix_type in ("distance", "similarity"),
            "labels": labels,
            "_Y": Y,
        },
    )


def select(adj, index):
    """Select matrices with stable scalar versus sequence rank."""
    if isinstance(index, tuple):
        index = list(index)
    positions = np.arange(len(adj))[index]
    single = np.ndim(positions) == 0
    rows = np.atleast_1d(positions).tolist()
    values = (
        adj.data.reshape(len(adj), -1)
        if len(adj) and adj.data.size
        else np.empty((len(adj), adj.data.shape[-1]), dtype=adj.data.dtype)
    )
    values = values[positions] if single else values[rows]
    labels = adj.labels
    if labels_are_nested(labels):
        labels = labels[int(positions)] if single else [labels[i] for i in rows]
    frame = adj.Y[rows] if adj.Y.width else adj.Y
    if adj.matrix_type == "empty":
        return adj.copy()
    return result(adj, values, labels=labels, Y=frame)


def common_labels(adj):
    """Return labels shared by all matrices, or no labels if they disagree."""
    if not labels_are_nested(adj.labels):
        return adj.labels
    first = adj.labels[0]
    return first if all(row == first for row in adj.labels) else []


def validate_compatible(left, right, *, labels=False):
    """Require the same relation schema and optionally the same node ordering."""
    if left.n_nodes != right.n_nodes or left.matrix_type != right.matrix_type:
        raise ValueError(
            "Adjacency instances must have matching node counts and matrix types."
        )
    if labels:

        def rows(adj):
            return (
                adj.labels if labels_are_nested(adj.labels) else [adj.labels] * len(adj)
            )

        if rows(left) != rows(right):
            raise ValueError("Adjacency node labels and ordering must match.")


def append(left, right):
    """Stack compatible matrices and merge matrix metadata by column name."""
    from . import Adjacency

    if not isinstance(right, Adjacency):
        raise ValueError("Data must be an Adjacency instance.")
    if left.matrix_type == "empty":
        return right.copy()
    if right.matrix_type == "empty":
        return left.copy()
    validate_compatible(left, right)
    if left.is_empty:
        return right.copy()
    if right.is_empty:
        return left.copy()
    if bool(left.labels) != bool(right.labels):
        raise ValueError(
            "Both inputs must supply node labels or neither may supply them."
        )
    labels = left.labels
    if left.labels != right.labels or labels_are_nested(left.labels):
        labels = []
        for adj in (left, right):
            labels.extend(
                adj.labels if labels_are_nested(adj.labels) else [adj.labels] * len(adj)
            )
    frames = []
    columns = set(left.Y.columns) | set(right.Y.columns)
    for adj in (left, right):
        frames.append(
            adj.Y
            if adj.Y.width or not columns
            else pl.DataFrame({name: [None] * len(adj) for name in sorted(columns)})
        )
    frame = pl.concat(frames, how="diagonal_relaxed")
    return result(left, np.vstack([left.data, right.data]), labels=labels, Y=frame)


def to_square(adj):
    """Export detached square matrices with a zero symmetric diagonal."""
    if adj.matrix_type == "empty":
        return np.empty((0, 0))

    def expand(row):
        return (
            squareform(row)
            if adj.issymmetric
            else row.reshape(adj.n_nodes, adj.n_nodes).copy()
        )

    return (
        expand(adj.data) if adj.is_single_matrix else [expand(row) for row in adj.data]
    )


def distance_to_similarity(adj, metric, beta):
    """Apply the established distance conversion independently per matrix."""
    if adj.matrix_type != "distance":
        raise ValueError("Matrix is not a distance matrix.")
    if metric == "correlation":
        values = 1 - adj.data
    elif metric == "euclidean":
        scales = np.array([np.std(squareform(row)) for row in np.atleast_2d(adj.data)])
        values = np.exp(
            -beta * adj.data / (scales[0] if adj.is_single_matrix else scales[:, None])
        )
    else:
        raise ValueError('metric can only be ["correlation","euclidean"]')
    return result(adj, values, labels=adj.labels, Y=adj.Y, matrix_type="similarity")
