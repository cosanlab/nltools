"""Lazy fetcher for files hosted in the ``nltools/niftis`` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(``~/.cache/huggingface/hub`` by default); subsequent calls return the
cached path without touching the network.

In Pyodide the synchronous HF cache is unavailable (``huggingface_hub``
isn't installed and sync HTTP from Python is not viable). Consumers must
``await seed_resources([...])`` once at app boot to pre-download the
files they need; subsequent sync ``fetch_resource()`` calls then hit
the IDBFS-backed cache populated by the seed. The cache persists across
page reloads via IndexedDB, so seeding only does network work once per
browser per dataset revision.
"""

import sys
from pathlib import Path

REPO_ID = "nltools/niftis"
REVISION = "main"

# In Pyodide we mount IDBFS at this top-level path so writes survive page
# reloads. Outside Pyodide the constant is unused (huggingface_hub manages
# its own cache under ~/.cache/huggingface).
_PYODIDE_CACHE_ROOT = Path("/nltools_cache") / REVISION

# Per-session flag — IDBFS can only be mounted once per Pyodide instance.
_idbfs_mounted = False


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

    The cache is backed by IndexedDB, so files persist across page reloads.
    The first call per session mounts IDBFS and pulls any prior data;
    subsequent calls only download files not already cached.

    Args:
        relpaths: Paths within the dataset repo to pre-fetch.
    """
    if "pyodide" not in sys.modules:
        return

    await _ensure_idbfs_mounted()

    from pyodide.http import pyfetch

    new_files = False
    for relpath in relpaths:
        target = _PYODIDE_CACHE_ROOT / relpath
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{relpath}"
        resp = await pyfetch(url)
        target.write_bytes(await resp.bytes())
        new_files = True

    if new_files:
        await _flush_idbfs()


def _fetch_pyodide(relpath: str) -> str:
    cache = _PYODIDE_CACHE_ROOT / relpath
    if cache.exists():
        return str(cache)
    raise RuntimeError(
        f"Resource {relpath!r} not in the Pyodide cache. Call "
        f"`await nltools.templates.seed_resources([{relpath!r}])` "
        f"before any sync template access."
    )


async def _ensure_idbfs_mounted() -> None:
    """Mount IDBFS at the cache root and load any prior data. Idempotent."""
    global _idbfs_mounted
    if _idbfs_mounted:
        return

    import pyodide_js

    mount_point = str(_PYODIDE_CACHE_ROOT.parent)
    Path(mount_point).mkdir(parents=True, exist_ok=True)

    fs = pyodide_js.FS
    fs.mount(fs.filesystems.IDBFS, {}, mount_point)
    await _syncfs(populate=True)
    _idbfs_mounted = True


async def _flush_idbfs() -> None:
    """Push MEMFS writes back to IndexedDB."""
    await _syncfs(populate=False)


async def _syncfs(*, populate: bool) -> None:
    """Wrap Emscripten FS.syncfs callback in an awaitable."""
    import asyncio

    import pyodide_js
    from pyodide.ffi import create_proxy

    fut: asyncio.Future = asyncio.get_event_loop().create_future()

    def callback(err):
        if err is not None:
            fut.set_exception(RuntimeError(f"FS.syncfs failed: {err}"))
        else:
            fut.set_result(None)

    proxy = create_proxy(callback)
    try:
        pyodide_js.FS.syncfs(populate, proxy)
        await fut
    finally:
        proxy.destroy()
