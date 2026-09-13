"""Write a fit or a contrast to a directory of NIfTI maps, designs and a sidecar.

One layout serves both records: every brain map becomes a compressed NIfTI named
`<prefix>_<suffix>.nii.gz`, every design becomes a CSV, and one small JSON
sidecar names the kind of result and whatever describes its columns. Nothing
here is BIDS: the files are the ones a user would otherwise write by hand after
a fit.

`ContrastResult` lives in `nltools/models`, the functional core, and delegates
here, so nothing on the contrast path may import a facade class. It duck-types
on the payload instead. `DesignMatrix` is reached only from `_write_fit`, which
is called from the facade side.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _output_path(directory: Path, prefix: str | None, name: str) -> Path:
    """Join `directory` and `name`, prefixed with `<prefix>_` when there is one."""
    return directory / (f"{prefix}_{name}" if prefix else name)


def _prepared_directory(directory, prefix: str | None) -> Path:
    """Validate the prefix and make the destination, returning it as a `Path`.

    A prefix naming a subdirectory would need that subdirectory to exist, and
    the failure surfaces from nibabel as a bare `FileNotFoundError`, so it is
    refused here instead.
    """
    if prefix is not None and (os.sep in prefix or "/" in prefix):
        raise ValueError(
            f"prefix={prefix!r} names a path, but it is a filename prefix: it "
            f"is prepended to each file's name inside `directory`. Put the "
            f"subdirectory in `directory` instead."
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_maps(directory: Path, prefix: str | None, maps: dict) -> list[Path]:
    """Write each named `BrainData` as a compressed NIfTI, skipping the `None`s."""
    written = []
    for name, brain_map in maps.items():
        if brain_map is None:
            continue
        path = _output_path(directory, prefix, f"{name}.nii.gz")
        brain_map.write(path)
        written.append(path)
    return written


def _write_design(directory: Path, prefix: str | None, design) -> list[Path]:
    """Write the design as CSV — one file per feature space for a banded ridge.

    A `DesignMatrix` writes itself; a plain feature matrix goes through one so
    the file has a header and reads back the same way.
    """
    import numpy as np

    from nltools.data.designmatrix import DesignMatrix

    if design is None:
        return []
    if isinstance(design, dict):
        written = []
        for space, matrix in design.items():
            path = _output_path(directory, prefix, f"design-{space}.csv")
            DesignMatrix(np.asarray(matrix)).write(str(path))
            written.append(path)
        return written

    path = _output_path(directory, prefix, "design.csv")
    frame = design if isinstance(design, DesignMatrix) else DesignMatrix(design)
    frame.write(str(path))
    return [path]


def _write_sidecar(
    directory: Path, prefix: str | None, name: str, contents: dict
) -> Path:
    """Write one JSON sidecar describing what the other files hold."""
    path = _output_path(directory, prefix, f"{name}.json")
    path.write_text(json.dumps(contents, indent=2) + "\n")
    return path


def _design_sidecar(design) -> dict:
    """Describe the design's shape for the sidecar, in the CSVs' own terms.

    A banded design's spaces are named under `spaces`, one per CSV, because
    they are a different kind of thing from a single design's columns. Every
    other design reports `columns` holding exactly the header its CSV carries,
    auto-generated names for a plain feature matrix included.
    """
    from nltools.data.designmatrix import DesignMatrix

    if design is None:
        return {"columns": None}
    if isinstance(design, dict):
        return {"spaces": [str(space) for space in design]}
    if isinstance(design, DesignMatrix):
        return {"columns": list(design.columns)}
    return {"columns": list(DesignMatrix(design).columns)}


def _write_fit(fit, directory, prefix: str | None) -> list[Path]:
    """Write one `FitResult`: its maps, its design, and a `_fit.json` sidecar.

    Args:
        fit (FitResult): The record to write.
        directory (str | Path): Destination, created when it does not exist.
        prefix (str | None): Prepended to every filename as `<prefix>_`.

    Returns:
        list[Path]: Every file written, in the order it was written.
    """
    directory = _prepared_directory(directory, prefix)

    written = _write_maps(
        directory,
        prefix,
        {
            "betas": fit.betas,
            "predicted": fit.predicted,
            "residual": fit.residual,
            "r2": fit.r2,
            "alpha": fit.alpha,
        },
    )
    written += _write_design(directory, prefix, fit.design)
    written.append(
        _write_sidecar(
            directory,
            prefix,
            "fit",
            {
                "kind": fit.kind,
                **_design_sidecar(fit.design),
                "files": [path.name for path in written],
            },
        )
    )
    return written


#: The contrast statistic each field is written under. `t`, `z` and `p` are the
#: conventional neuroimaging filenames for the three inferential maps.
_CONTRAST_SUFFIXES = {
    "effect": "effect",
    "variance": "variance",
    "standard_error": "se",
    "statistic": "t",
    "z_score": "z",
    "p_value": "p",
}


def _write_contrast(result, directory, prefix: str | None) -> list[Path]:
    """Write one `ContrastResult`: its maps and a `_contrast.json` sidecar.

    Args:
        result (ContrastResult): The record to write, whose payloads must be
            `BrainData` maps.
        directory (str | Path): Destination, created when it does not exist.
        prefix (str | None): Prepended to every filename as `<prefix>_`.

    Returns:
        list[Path]: Every file written, in the order it was written.

    Raises:
        TypeError: If the payloads are bare arrays rather than brain maps.
    """
    import numpy as np

    # A payload that can write itself is a brain map; asking for the method
    # rather than the class keeps this module — which `nltools.models` reaches
    # for `ContrastResult.write` — from importing a facade package.
    if not hasattr(result.effect, "write"):
        raise TypeError(
            f"write() saves brain maps as NIfTI, but this contrast holds "
            f"{type(result.effect).__name__} payloads. Contrasts computed "
            f"through BrainData.compute_contrasts carry brain maps."
        )

    directory = _prepared_directory(directory, prefix)

    written = _write_maps(
        directory,
        prefix,
        {suffix: getattr(result, name) for name, suffix in _CONTRAST_SUFFIXES.items()},
    )
    degrees_of_freedom = result.degrees_of_freedom
    written.append(
        _write_sidecar(
            directory,
            prefix,
            "contrast",
            {
                "kind": "contrast",
                "degrees_of_freedom": np.asarray(degrees_of_freedom).tolist(),
                "files": [path.name for path in written],
            },
        )
    )
    return written
