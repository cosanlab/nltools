"""nltools I/O utilities.

`events_to_dm` turns a BIDS events table into boxcar regressors. HDF5
serialization for the data classes lives in `nltools.io.h5`; users reach it
through `BrainData.write`/`Adjacency.write` and the constructors.
"""

from .events import events_to_dm

__all__ = ["events_to_dm"]
