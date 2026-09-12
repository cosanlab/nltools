"""Stacking data objects of one class into a single object."""


def concatenate(data):
    """Concatenate a list of `BrainData` or `Adjacency` objects.

    Args:
        data (list[BrainData] | list[Adjacency]): Objects to concatenate; all must
            be of the same class.

    Returns:
        BrainData | Adjacency: A single object holding every input in order.

    Raises:
        ValueError: If `data` is not a list or mixes classes.
    """

    if not isinstance(data, list):
        raise ValueError("Make sure you are passing a list of objects.")

    if all(isinstance(x, data[0].__class__) for x in data):
        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError("Make sure all objects in the list are the same type.")
    return out
