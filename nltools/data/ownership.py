"""Copying that leaves the clone owning its own buffers.

A data class holds polars frames whose storage may be a view onto a NumPy
array the user still holds, and frames whose `pl.Object` cells are Python
objects shared with the original. Copying either naively hands the clone a
buffer or a cell somebody else can mutate. `copy_frame` detaches one frame;
`_copy_graph` walks a whole object's `__dict__` and detaches every frame it
finds, preserving the aliases inside that graph through a shared memo.

The two frame copiers are deliberately not the same function: `copy_frame`
re-`gather`s every non-Object series so a `DesignMatrix` clone owns its
numeric buffers outright, while `_copy_object_frames` clones the frame and
rewrites only its `pl.Object` columns, which is what `BrainData` and
`Adjacency` metadata need.
"""

from copy import deepcopy

import polars as pl


def copy_frame(frame: pl.DataFrame, memo: dict | None = None) -> pl.DataFrame:
    """Detach frame storage and Python Object cells with a shared copy memo."""
    if memo is None:
        memo = {}
    if id(frame) in memo:
        return memo[id(frame)]
    frames = []
    visited = set()

    def discover(value):
        if id(value) in visited or id(value) in memo:
            return
        visited.add(id(value))
        if isinstance(value, pl.DataFrame):
            frames.append(value)
            for series in value:
                if series.dtype == pl.Object:
                    for cell in series:
                        discover(cell)
        elif isinstance(value, dict):
            for key, item in value.items():
                discover(key)
                discover(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                discover(item)

    discover(frame)
    for source in frames:
        memo[id(source)] = source.clone()
    for source in frames:
        for index, series in enumerate(source):
            if series.dtype == pl.Object:
                detached = pl.Series(
                    series.name,
                    [deepcopy(cell, memo) for cell in series],
                    dtype=pl.Object,
                )
            else:
                # Gather owns buffers even when the frame wraps a NumPy view.
                detached = series.gather(pl.int_range(0, len(series), eager=True))
            memo[id(source)].replace_column(index, detached)
    return memo[id(frame)]


def _copy_graph(source, *, memo=None, exclude=(), replacements=None):
    """Copy one retained object graph, preserving its internal aliases."""
    if memo is None:
        memo = {}
    if id(source) in memo:
        return memo[id(source)]
    new = type(source).__new__(type(source))
    memo[id(source)] = new
    values = {
        key: value for key, value in source.__dict__.items() if key not in exclude
    }
    if replacements is not None:
        values.update(replacements)
    _copy_object_frames(values, memo)
    for key, value in values.items():
        setattr(new, key, deepcopy(value, memo))
    return new


def _copy_object_frames(values, memo):
    """Prepare Polars Object cells for deepcopy without sharing Python objects."""
    frames = []
    seen = set()

    def discover(value):
        if id(value) in seen or id(value) in memo:
            return
        seen.add(id(value))
        if isinstance(value, pl.DataFrame):
            frames.append(value)
            for series in value:
                if series.dtype == pl.Object:
                    for cell in series:
                        discover(cell)
        elif isinstance(value, dict):
            for key, item in value.items():
                discover(key)
                discover(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                discover(item)

    discover(values)
    # Register all frames first, including frames referred to by Object cells.
    # The common memo preserves cycles and cell aliases across metadata frames.
    for frame in frames:
        memo[id(frame)] = frame.clone()
    for frame in frames:
        for index, series in enumerate(frame):
            if series.dtype == pl.Object:
                memo[id(frame)].replace_column(
                    index,
                    pl.Series(
                        series.name,
                        [deepcopy(cell, memo) for cell in series],
                        dtype=pl.Object,
                    ),
                )


def _copy_complete(source, memo=None):
    """Return a complete independently owned snapshot."""
    return _copy_graph(source, memo=memo)
