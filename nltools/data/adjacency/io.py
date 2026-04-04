"""I/O functions for Adjacency objects."""

import pandas as pd
from pathlib import Path


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
            pd.DataFrame(adj.data).to_csv(file_name, index=None)
        elif adj.is_single_matrix and method == "square":
            pd.DataFrame(adj.squareform()).to_csv(file_name, index=None)
        elif not adj.is_single_matrix and method == "square":
            raise NotImplementedError(
                "Need to decide how we should write out multiple matrices. As separate files?"
            )


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
            labels = {x: y for x, y in zip(G.nodes, adj.labels)}
            nx.relabel_nodes(G, labels, copy=False)
        return G
    else:
        raise NotImplementedError(
            "This function currently only works on single matrices."
        )
