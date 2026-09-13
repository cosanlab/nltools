"""The browser fetch path: a direct GET, never `hf_hub_download`.

`hf_hub_download` HEAD-probes the resolve URL, and both Pyodide HTTP backends
crash on the bodiless response, so under Pyodide `fetch_resource` must go
straight to `pyodide.http.pyfetch` instead. The whole Pyodide runtime is faked
here — a real one needs a browser or node.
"""

import asyncio
import importlib
import sys
import types

import pytest


class _FakeProxy:
    """Stand-in for `pyodide.ffi.create_proxy`'s JS callback wrapper."""

    def __init__(self, function):
        self._function = function

    def __call__(self, *args):
        return self._function(*args)

    def destroy(self):
        pass


class _FakeResponse:
    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    async def bytes(self):
        return self._payload


@pytest.fixture
def pyodide_env(tmp_path, monkeypatch):
    """Install a fake Pyodide runtime and hand back the reloaded fetch module."""
    calls = {"urls": [], "syncfs": [], "mounts": []}

    async def pyfetch(url):
        calls["urls"].append(url)
        return _FakeResponse(b"fake-nifti-bytes")

    def syncfs(populate, callback):
        calls["syncfs"].append(populate)
        callback(None)

    filesystems = types.SimpleNamespace(IDBFS=object())
    fs = types.SimpleNamespace(
        filesystems=filesystems,
        mount=lambda fs_type, options, point: calls["mounts"].append(point),
        syncfs=syncfs,
    )

    pyodide = types.ModuleType("pyodide")
    pyodide_http = types.ModuleType("pyodide.http")
    pyodide_http.pyfetch = pyfetch
    pyodide_ffi = types.ModuleType("pyodide.ffi")
    pyodide_ffi.create_proxy = _FakeProxy
    pyodide_ffi.run_sync = asyncio.run
    pyodide_js = types.ModuleType("pyodide_js")
    pyodide_js.FS = fs
    js = types.ModuleType("js")
    js.Object = types.SimpleNamespace(new=dict)

    def exploding_download(*args, **kwargs):
        raise AssertionError("hf_hub_download must not run under Pyodide")

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = exploding_download

    for name, module in {
        "pyodide": pyodide,
        "pyodide.http": pyodide_http,
        "pyodide.ffi": pyodide_ffi,
        "pyodide_js": pyodide_js,
        "js": js,
        "huggingface_hub": hub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    fetch = importlib.reload(importlib.import_module("nltools.templates.fetch"))
    monkeypatch.setattr(fetch, "_PYODIDE_CACHE_ROOT", tmp_path / "cache")
    yield fetch, calls

    # Other tests import the module without a fake Pyodide in sys.modules.
    monkeypatch.undo()
    importlib.reload(fetch)


def test_fetch_resource_downloads_into_the_idbfs_cache(pyodide_env):
    fetch, calls = pyodide_env

    path = fetch.fetch_resource("masks/k88_parcel_names.csv")

    assert path.endswith("masks/k88_parcel_names.csv")
    assert calls["urls"] == [
        "https://huggingface.co/datasets/nltools/niftis/resolve/main/"
        "masks/k88_parcel_names.csv"
    ]
    assert calls["mounts"] == [str(fetch._PYODIDE_CACHE_ROOT.parent)]
    # Mounted with populate=True, flushed with populate=False after the write.
    assert calls["syncfs"] == [True, False]


def test_second_fetch_serves_the_cache_without_a_download(pyodide_env):
    fetch, calls = pyodide_env

    first = fetch.fetch_resource("masks/k88_parcel_names.csv")
    fetch.fetch_resource.cache_clear()
    second = fetch.fetch_resource("masks/k88_parcel_names.csv")

    assert first == second
    assert len(calls["urls"]) == 1


def test_list_resources_is_unavailable_under_pyodide(pyodide_env):
    fetch, _ = pyodide_env

    with pytest.raises(RuntimeError, match="fetch_resource"):
        fetch.list_resources()
