"""Lazy fetcher for files hosted in the ``nltools/niftis`` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(``~/.cache/huggingface/hub`` by default); subsequent calls return the
cached path without touching the network.
"""

import sys

REPO_ID = "nltools/niftis"
REVISION = "main"


def fetch_resource(relpath: str) -> str:
    """Return a local path to a file from the ``nltools/niftis`` HF dataset.

    Args:
        relpath: Path within the dataset repo, e.g.
            ``'default/2mm-MNI152-2009fsl-mask.nii.gz'`` or
            ``'masks/k88_parcel_names.csv'``.

    Returns:
        Absolute path to the cached file on disk.
    """
    if "pyodide" in sys.modules:
        return _fetch_pyodide(relpath)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=relpath,
        repo_type="dataset",
        revision=REVISION,
    )


def _fetch_pyodide(relpath: str) -> str:
    raise NotImplementedError(
        "Pyodide async fetch path is not yet wired up. "
        "Pre-seed MEMFS at app boot or call an async fetcher from JS."
    )
