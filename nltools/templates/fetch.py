"""Lazy fetcher for files hosted in the ``nltools/niftis`` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(``~/.cache/huggingface/hub`` by default); subsequent calls return the
cached path without touching the network.

In Pyodide the synchronous HF cache is unavailable (``huggingface_hub``
isn't installed and sync HTTP from Python is not viable). Consumers must
``await seed_resources([...])`` once at app boot to pre-download the
files they need; subsequent sync ``fetch_resource()`` calls then hit a
small MEMFS cache populated by the seed.
"""

import sys
from pathlib import Path

REPO_ID = "nltools/niftis"
REVISION = "main"

_PYODIDE_CACHE_ROOT = Path.home() / ".cache" / "nltools-niftis" / REVISION


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


async def seed_resources(relpaths: list[str]) -> None:
    """Pre-download dataset files in Pyodide so sync fetches resolve from cache.

    No-op outside Pyodide — :func:`fetch_resource` does its own lazy download
    via ``huggingface_hub`` there. In Pyodide this must be called (and
    awaited) before any code path that calls :func:`fetch_resource`,
    :func:`resolve_paths`, or :func:`resolve_template_name` synchronously.

    Args:
        relpaths: Paths within the dataset repo to pre-fetch.
    """
    if "pyodide" not in sys.modules:
        return

    from pyodide.http import pyfetch

    for relpath in relpaths:
        target = _PYODIDE_CACHE_ROOT / relpath
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{relpath}"
        resp = await pyfetch(url)
        target.write_bytes(await resp.bytes())


def _fetch_pyodide(relpath: str) -> str:
    cache = _PYODIDE_CACHE_ROOT / relpath
    if cache.exists():
        return str(cache)
    raise RuntimeError(
        f"Resource {relpath!r} not in the Pyodide cache. Call "
        f"`await nltools.templates.seed_resources([{relpath!r}])` "
        f"before any sync template access."
    )
