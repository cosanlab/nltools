"""
nltools I/O utilities.

HDF5 serialization and file reading for neuroimaging data types.
"""

from .h5 import is_h5_path, load_brain_data_h5, to_h5


def __getattr__(name):
    # Lazy import to avoid circular dependency (file_reader imports nltools.data)
    if name == "onsets_to_dm":
        from .file_reader import onsets_to_dm

        return onsets_to_dm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "is_h5_path",
    "load_brain_data_h5",
    "onsets_to_dm",
    "to_h5",
]
