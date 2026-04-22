"""I/O functions for Adjacency objects."""

from pathlib import Path

import numpy as np
import polars as pl


def write(adj, file_name, method="long"):
    """Write out Adjacency object to csv file.

    Args:
        adj: Adjacency object to write
        file_name (str):  name of file name to write
        method (str):     method to write out data ['long','square']

    """
    from nltools.io import is_h5_path, to_h5

    if method not in ["long", "square"]:
        raise ValueError('Make sure method is ["long","square"].')

    if isinstance(file_name, Path):
        file_name = str(file_name)

    if is_h5_path(file_name):
        if method == "square":
            raise NotImplementedError('Saving as hdf5 does not support method="square"')
        to_h5(adj, file_name, obj_type="adjacency")
    else:
        if method == "long":
            _write_2d_csv(adj.data, file_name)
        elif adj.is_single_matrix and method == "square":
            _write_2d_csv(np.asarray(adj.squareform()), file_name)
        elif not adj.is_single_matrix and method == "square":
            raise NotImplementedError(
                "Need to decide how we should write out multiple matrices. As separate files?"
            )


def _write_2d_csv(arr: np.ndarray, file_name: str) -> None:
    """Write an array to CSV via polars with numeric-index column names.

    Matches the pandas ``pd.DataFrame(arr).to_csv(index=None)`` layout:
    1-D arrays become a single-column frame with ``len(arr)`` rows.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    schema = [str(i) for i in range(arr.shape[1])]
    pl.DataFrame(arr, schema=schema).write_csv(file_name)


def to_graph(adj):
    """Convert Adjacency into networkx graph.

    Only works on single matrices for now.

    Args:
        adj (Adjacency): Adjacency instance (must be a single matrix).

    Returns:
        networkx.Graph or networkx.DiGraph: Graph representation of the
            adjacency matrix. Uses DiGraph for directed matrices.
    """

    import networkx as nx

    if adj.is_single_matrix:
        if adj.matrix_type == "directed":
            G = nx.DiGraph(adj.squareform())
        else:
            G = nx.Graph(adj.squareform())
        if adj.labels:
            labels = dict(zip(G.nodes, adj.labels))
            nx.relabel_nodes(G, labels, copy=False)
        return G
    raise NotImplementedError("This function currently only works on single matrices.")
