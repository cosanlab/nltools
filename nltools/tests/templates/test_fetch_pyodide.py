"""The browser fetch path: a direct GET, never `hf_hub_download`.

`hf_hub_download` HEAD-probes the resolve URL, and both Pyodide HTTP backends
crash on the bodiless response, so under Pyodide `fetch_resource` must go
straight to a GET instead. The whole Pyodide runtime is faked here — a real one
needs a browser or node.
"""

import asyncio
import sys
import types

import pytest

from nltools.templates import fetch

RESOURCE = "masks/k88_parcel_names.csv"
EXPECTED_URL = (
    "https://huggingface.co/datasets/nltools/niftis/resolve/main/"
    "masks/k88_parcel_names.csv"
)


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
    """Install a fake Pyodide runtime around the real fetch module.

    Every Pyodide import in the module is lazy, and the `"pyodide" in
    sys.modules` check runs per call, so faking `sys.modules` is enough — the
    module never needs reloading.
    """
    calls = {"urls": [], "syncfs": [], "mounts": []}

    async def pyfetch(url):
        calls["urls"].append(url)
        return _FakeResponse(b"fake-nifti-bytes")

    def syncfs(populate, callback):
        calls["syncfs"].append(populate)
        callback(None)

    fs = types.SimpleNamespace(
        filesystems=types.SimpleNamespace(IDBFS=object()),
        mount=lambda fs_type, options, point: calls["mounts"].append(point),
        syncfs=syncfs,
    )

    pyodide = types.ModuleType("pyodide")
    pyodide_http = types.ModuleType("pyodide.http")
    pyodide_http.pyfetch = pyfetch
    pyodide_ffi = types.ModuleType("pyodide.ffi")
    pyodide_ffi.create_proxy = _FakeProxy
    pyodide_ffi.run_sync = asyncio.run
    pyodide_ffi.can_run_sync = lambda: True
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

    monkeypatch.setattr(fetch, "_PYODIDE_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(fetch, "_idbfs_mounted", False)
    fetch.fetch_resource.cache_clear()
    yield fetch, calls, pyodide_ffi
    fetch.fetch_resource.cache_clear()


def test_fetch_resource_downloads_into_the_idbfs_cache(pyodide_env):
    fetch, calls, _ = pyodide_env

    path = fetch.fetch_resource(RESOURCE)

    assert path.endswith(RESOURCE)
    assert calls["urls"] == [EXPECTED_URL]
    assert calls["mounts"] == [str(fetch._PYODIDE_CACHE_ROOT.parent)]
    # Mounted with populate=True, flushed with populate=False after the write.
    assert calls["syncfs"] == [True, False]


def test_second_fetch_serves_the_cache_without_a_download(pyodide_env):
    fetch, calls, _ = pyodide_env

    first = fetch.fetch_resource(RESOURCE)
    fetch.fetch_resource.cache_clear()
    second = fetch.fetch_resource(RESOURCE)

    assert first == second
    assert len(calls["urls"]) == 1


def test_no_partial_file_is_left_behind_when_the_write_fails(pyodide_env):
    """A short cache file would be served as a hit forever, and persisted."""
    fetch, _, _ = pyodide_env
    target = fetch._PYODIDE_CACHE_ROOT / RESOURCE

    def exploding_write(self, payload):
        raise OSError("no space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("pathlib.Path.write_bytes", exploding_write)
        with pytest.raises(OSError, match="no space"):
            fetch.fetch_resource(RESOURCE)

    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_fetch_falls_back_to_requests_without_promise_integration(
    pyodide_env, monkeypatch
):
    """No JSPI means no stack switch, so the async transport is unreachable."""
    fetch, calls, pyodide_ffi = pyodide_env
    monkeypatch.setattr(pyodide_ffi, "can_run_sync", lambda: False)

    requested = []

    def get(url, timeout):
        requested.append(url)
        return types.SimpleNamespace(status_code=200, content=b"fake-nifti-bytes")

    requests = types.ModuleType("requests")
    requests.get = get
    monkeypatch.setitem(sys.modules, "requests", requests)

    path = fetch.fetch_resource(RESOURCE)

    assert path.endswith(RESOURCE)
    assert requested == [EXPECTED_URL]
    # No IDBFS on this path: the cache is session-only.
    assert calls["mounts"] == []
    assert calls["urls"] == []
