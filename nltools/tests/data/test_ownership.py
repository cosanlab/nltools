"""Tests for the shared copy helpers in nltools.data.ownership."""

import numpy as np
import polars as pl

from nltools.data.ownership import _copy_object_frames, _copy_frame


def test_copy_frame_owns_buffers_that_object_frame_copying_leaves_shared():
    """`_copy_frame` detaches numeric buffers; `_copy_object_frames` does not.

    Polars wraps a NumPy array zero-copy, so a plain clone still reads the
    caller's memory. The two copiers answer that differently and both answers
    are load-bearing, so they stay separate functions.
    """
    values = np.arange(5, dtype=np.float64)
    frame = pl.DataFrame({"a": values})

    detached = _copy_frame(frame)
    memo = {}
    _copy_object_frames({"frame": frame}, memo)
    shared = memo[id(frame)]

    values[0] = 99.0

    assert detached["a"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert shared["a"].to_list() == [99.0, 1.0, 2.0, 3.0, 4.0]
