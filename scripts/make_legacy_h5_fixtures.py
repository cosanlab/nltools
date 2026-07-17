"""Generate tiny, faithful pre-0.6 (deepdish/PyTables) HDF5 fixtures.

nltools <= 0.5.1 saved ``BrainData``/``Adjacency`` via deepdish, producing an
h5 layout that 0.6's ``nltools.io.h5`` still reads for backward compatibility
(see ``_load_legacy_brain_data_h5`` / ``load_legacy_adjacency_h5``). Real files
run to several MB; this script writes byte-tiny fixtures with the *same* node
structure so the legacy-read tests always run without committing research data.

The layout mirrors what deepdish emitted (validated against real 0.5.1 files):
- ``data``                    flat Dataset (2D for BrainData, 1D vector for Adjacency)
- ``X``/``Y``                 Dataset of frame values (empty frames -> deepdish
                              stub Dataset; populated -> 2D values array)
- ``X_columns``/``Y_columns`` empty deepdish list Group (``TITLE='list:0'``) when
                              the frame is empty, else a string Dataset of names
- ``mask_data``/``mask_affine``  BrainData only; no ``mask_file_name`` (old files
                              frequently omitted it -> mask rebuilt from array)

Run: ``uv run python scripts/make_legacy_h5_fixtures.py``
"""

from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "nltools/tests/io_tests/legacy_fixtures"


def _empty_list_group(f: h5py.File, name: str) -> None:
    """Write a deepdish empty-list stub: a childless Group titled ``list:0``."""
    g = f.create_group(name)
    g.attrs["TITLE"] = np.bytes_("list:0")


def _string_dataset(f: h5py.File, name: str, values: list[str]) -> None:
    """Write column names the way deepdish stored them: fixed-width bytes."""
    f.create_dataset(name, data=np.array(values, dtype="S32"))


def make_braindata(path: Path) -> None:
    """Tiny legacy BrainData: 2D data, populated X and Y, array-rebuilt mask."""
    rng = np.random.default_rng(0)

    # Small mask: 8^3 grid with 20 in-brain voxels -> data is (5 images, 20 voxels).
    mask_data = np.zeros((8, 8, 8), dtype=np.float64)
    flat_idx = rng.choice(mask_data.size, size=20, replace=False)
    mask_data.flat[flat_idx] = 1.0
    n_vox = int(mask_data.sum())
    affine = np.eye(4, dtype=np.float64) * 3.0
    affine[3, 3] = 1.0

    data = rng.standard_normal((5, n_vox)).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)

        # Populated X (design-matrix-like) and Y (metadata) frames.
        f.create_dataset("X", data=np.column_stack([np.ones(5), np.arange(5.0)]))
        _string_dataset(f, "X_columns", ["intercept", "regressor"])
        f.create_dataset("Y", data=np.arange(5.0).reshape(5, 1))
        _string_dataset(f, "Y_columns", ["condition"])

        f.create_dataset("mask_data", data=mask_data)
        f.create_dataset("mask_affine", data=affine)


def make_adjacency(path: Path) -> None:
    """Tiny legacy Adjacency: 300-element long-form vector -> 25 nodes, distance."""
    rng = np.random.default_rng(1)
    n_nodes = 25
    n_pairs = n_nodes * (n_nodes - 1) // 2  # 300
    data = rng.random(n_pairs).astype(np.float64)

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        # Empty Y frame, but Y_columns node present so is_legacy_adjacency_h5 fires.
        f.create_dataset("Y", data=np.array([0], dtype=np.int64))
        _empty_list_group(f, "Y_columns")
        _empty_list_group(f, "Y_index")
        # No matrix_type node -> loader warns and defaults to distance. No labels.


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_braindata(OUT / "legacy_braindata.h5")
    make_adjacency(OUT / "legacy_adjacency.h5")
    for p in sorted(OUT.glob("*.h5")):
        print(f"wrote {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
