"""Lazy fetcher for files hosted in the `nltools/niftis` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(`~/.cache/huggingface/hub` by default); subsequent calls return the
cached path without touching the network.

In Pyodide `hf_hub_download` is unusable: it HEAD-probes the resolve URL for
metadata, and both Pyodide HTTP backends dereference the body of that bodiless
response and crash. The browser path therefore skips the client and does a
direct GET (`pyodide.http.pyfetch`) into an IDBFS-backed cache directory, which
survives a page reload. Where the browser has no JavaScript Promise Integration
to bridge that async fetch back to a synchronous call, a `requests` GET stands
in and the cache is session-only — a page reload downloads again. Same repo,
same revision, same memoization either way; only the transport differs.
"""

import functools
import sys
from pathlib import Path

REPO_ID = "nltools/niftis"
REVISION = "main"

# Mount point for the Pyodide cache. IDBFS is mounted at the parent so the
# whole tree lands in one IndexedDB store. Outside Pyodide this is unused;
# huggingface_hub manages its own cache under ~/.cache/huggingface.
_PYODIDE_CACHE_ROOT = Path("/nltools_cache") / REVISION

# IDBFS can only be mounted once per Pyodide instance.
_idbfs_mounted = False


@functools.cache
def fetch_resource(relpath: str) -> str:
    """Return a local path to a file from the `nltools/niftis` HF dataset.

    Args:
        relpath (str): Path within the dataset repo, e.g.
            `'default/2mm-MNI152-2009fsl-mask.nii.gz'` or
            `'masks/k88_parcel_names.csv'`. Use `list_resources`
            to enumerate what's available.

    Returns:
        str: Absolute path to the cached file on disk. The returned path drops
            straight into anything that takes a NIfTI path — nilearn plotting
            and masking helpers, `nibabel.load`, and `BrainData(path)`.

    Note:
        Resolution is memoized per `relpath` for the session — repeated
        calls (e.g. every default-mask `BrainData` construction) return the
        cached path with no work. A file already in the HF cache is resolved
        offline, so only a genuine cache miss touches the network. Under
        Pyodide the file comes from an IndexedDB-backed cache instead, which
        persists across page reloads.
    """
    if "pyodide" in sys.modules:
        return _fetch_pyodide(relpath)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=relpath,
            repo_type="dataset",
            revision=REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=relpath,
            repo_type="dataset",
            revision=REVISION,
        )


@functools.lru_cache(maxsize=8)
def _list_repo_files_cached(repo_id: str, revision: str) -> tuple[str, ...]:
    """Single HF API hit per (repo, revision) for the session."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=revision
    )
    return tuple(sorted(files))


def list_resources(prefix: str | None = None) -> list[str]:
    """List files available in the `nltools/niftis` HF dataset.

    Companion to `fetch_resource` — surfaces what's downloadable
    without forcing users to remember relpath strings or visit the HF
    web UI.

    Args:
        prefix (str, optional): Path prefix to filter by (e.g. `'masks/'`,
            `'default/'`, `'fmriprep/'`). Matches with `str.startswith`.

    Returns:
        list[str]: Sorted relative paths usable with `fetch_resource`.

    Note:
        Hits the HF API once per session (cached). Under Pyodide this needs
        `httpcore` installed — `huggingface_hub`'s client imports it lazily and
        nothing else in the browser environment pulls it in, so run
        `await micropip.install("httpcore")` first.
    """
    files = _list_repo_files_cached(REPO_ID, REVISION)
    if prefix:
        return [f for f in files if f.startswith(prefix)]
    return list(files)


def _fetch_pyodide(relpath: str) -> str:
    """Return a cached path in the browser, downloading on a cache miss.

    Two transports, chosen by what the browser supports. With JavaScript
    Promise Integration, `pyodide.ffi.run_sync` drives an async `pyfetch` and
    the cache is IDBFS-backed, so it survives a page reload. Without it — an
    older browser, or a synchronous entry point that cannot stack-switch — a
    `requests` GET does the same job into a plain in-memory cache. Either way
    `fetch_resource` keeps the signature it has everywhere else.
    """
    from pyodide.ffi import can_run_sync, run_sync

    if can_run_sync():
        return run_sync(_download_pyodide(relpath))
    return _download_pyodide_sync(relpath)


async def _download_pyodide(relpath: str) -> str:
    """Fetch one dataset file into the IDBFS cache with a direct GET."""
    await _ensure_idbfs_mounted()

    target = _PYODIDE_CACHE_ROOT / relpath
    if target.exists():
        return str(target)

    from pyodide.http import pyfetch

    url = _resource_url(relpath)
    response = await pyfetch(url)
    _check_status(response.status, relpath, url)
    path = _write_cached(target, await response.bytes())
    await _flush_idbfs()
    return path


def _download_pyodide_sync(relpath: str) -> str:
    """Fetch one dataset file with `requests`, into a session-only cache.

    The fallback for a browser without Promise Integration. Mounting IDBFS
    needs an await of its own, so this path stays on the in-memory filesystem
    and a page reload downloads again.
    """
    target = _PYODIDE_CACHE_ROOT / relpath
    if target.exists():
        return str(target)

    try:
        import requests
    except ImportError as error:
        raise RuntimeError(
            "Downloading nltools data in this browser needs either JavaScript "
            "Promise Integration (call from an async notebook cell) or "
            "`requests` — run `await micropip.install('requests')`."
        ) from error

    url = _resource_url(relpath)
    response = requests.get(url, timeout=60)
    _check_status(response.status_code, relpath, url)
    return _write_cached(target, response.content)


def _resource_url(relpath: str) -> str:
    """Build the HF resolve URL both browser transports GET."""
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{relpath}"


def _check_status(status: int, relpath: str, url: str) -> None:
    """Raise unless the download returned 200."""
    if status != 200:
        raise RuntimeError(
            f"Could not download {relpath!r} from {REPO_ID}: HTTP {status} for {url}"
        )


def _write_cached(target: Path, payload: bytes) -> str:
    """Write `payload` to `target` atomically and return the path.

    A write that dies partway would otherwise leave a short file that every
    later call serves as a cache hit — and that `_flush_idbfs` may already have
    pushed into IndexedDB, where only wiping site data clears it. Renaming a
    fully written sibling into place is atomic on Emscripten's filesystem.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        partial.write_bytes(payload)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return str(target)


async def _ensure_idbfs_mounted() -> None:
    """Mount IDBFS at the cache root and load any prior data. Idempotent."""
    global _idbfs_mounted
    if _idbfs_mounted:
        return

    import js
    import pyodide_js

    mount_point = str(_PYODIDE_CACHE_ROOT.parent)
    Path(mount_point).mkdir(parents=True, exist_ok=True)

    fs = pyodide_js.FS
    # `FS.mount` wants a JS object for its options; a Python dict arrives as a
    # proxy the Emscripten filesystem cannot read.
    fs.mount(fs.filesystems.IDBFS, js.Object.new(), mount_point)
    # The mount is what must not be repeated: a second mount of a live point
    # raises a bare Emscripten error. A failed populate costs the prior cache,
    # not the mount, so the flag flips here rather than after the sync.
    _idbfs_mounted = True
    await _syncfs(populate=True)


async def _flush_idbfs() -> None:
    """Push MEMFS writes back to IndexedDB."""
    await _syncfs(populate=False)


async def _syncfs(*, populate: bool) -> None:
    """Wrap the Emscripten `FS.syncfs` callback in an awaitable."""
    import asyncio

    import pyodide_js
    from pyodide.ffi import create_proxy

    done: asyncio.Future = asyncio.get_running_loop().create_future()

    def callback(err):
        # Success passes JS `null`, which Pyodide surfaces as `JsNull` rather
        # than `None` — falsy either way, and an error object is truthy.
        if err:
            done.set_exception(RuntimeError(f"FS.syncfs failed: {err}"))
        else:
            done.set_result(None)

    proxy = create_proxy(callback)
    try:
        pyodide_js.FS.syncfs(populate, proxy)
        await done
    finally:
        proxy.destroy()
