/**
 * Pyodide/WASM smoke test runner for nltools.
 *
 * Checks the things that only break in the browser: that the wheel's own
 * dependency specifiers resolve under Pyodide, that it imports, that the data
 * classes construct, and that the direct-GET template fetch path works against
 * a real IndexedDB-backed cache.
 *
 * Needs node and network access, so it is deliberately outside `poe ok` and CI.
 *
 * Usage: uv run poe test-pyodide
 *        (or: uv build --wheel && npm install && node test_runner.mjs)
 *
 * Exit code 0 = all tests passed, 1 = some tests failed.
 */

// Pyodide's IDBFS needs IndexedDB, which node does not have. Must be imported
// before loadPyodide().
import "fake-indexeddb/auto";

import { loadPyodide } from "pyodide";
import { readFileSync, readdirSync, statSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Project root is three levels up from nltools/tests/pyodide/.
const projectRoot = join(__dirname, "..", "..", "..");

function findWheel() {
    const distDir = join(projectRoot, "dist");
    // `uv build --wheel` does not clear dist/, so pick the freshest build
    // rather than the last name alphabetically.
    const wheels = readdirSync(distDir)
        .filter((f) => f.endsWith(".whl") && f.startsWith("nltools"))
        .map((f) => join(distDir, f))
        .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    if (wheels.length === 0) {
        console.error("ERROR: no nltools wheel in dist/. Run: uv build --wheel");
        process.exit(1);
    }
    return wheels[0];
}

async function main() {
    console.log("=".repeat(60));
    console.log("nltools Pyodide/WASM smoke test runner");
    console.log("=".repeat(60));

    const pyodide = await loadPyodide();
    console.log(`\nPyodide ${pyodide.version}`);
    await pyodide.loadPackage("micropip");

    const wheelPath = findWheel();
    const wheelName = wheelPath.split("/").pop();
    pyodide.FS.mkdirTree("/wheel");
    pyodide.FS.writeFile(`/wheel/${wheelName}`, readFileSync(wheelPath));
    console.log(`Wheel: ${wheelName}`);

    // Deps ON, nothing pinned: micropip resolves the wheel's own declared
    // specifiers against the Pyodide lockfile and PyPI, which is what the
    // dependency floors exist for. `keep_going` reports every unresolvable
    // requirement instead of stopping at the first.
    console.log("\nResolving and installing the wheel with its dependencies...");
    try {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/wheel/${wheelName}", keep_going=True)
`);
    } catch (err) {
        console.error("\nFAIL: the wheel's declared dependencies do not resolve");
        console.error(String(err).split("\n").slice(-10).join("\n"));
        process.exit(1);
    }
    console.log("Installed");

    console.log("\n" + "=".repeat(60));
    console.log("Running smoke tests");
    console.log("=".repeat(60));

    const testCode = `
import sys
import numpy as np

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  OK   {name}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

print(f"\\nPlatform: {sys.platform}")

# ------------------------------------------------------------------
# What the resolve above actually chose
# ------------------------------------------------------------------
print("\\n[dependencies]")

import micropip

def test_bundled_versions_were_used():
    """polars and h5py are compiled: only the Pyodide lockfile can supply them."""
    installed = {p.name.replace("_", "-"): p.version for p in micropip.list().values()}
    print(f"       polars {installed['polars']}, h5py {installed['h5py']}")
    assert installed["polars"].startswith("1.33."), installed["polars"]
    assert installed["h5py"].startswith("3.13."), installed["h5py"]
test("compiled dependencies came from the Pyodide lockfile", test_bundled_versions_were_used)

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
print("\\n[imports]")

def test_import_nltools():
    import nltools
    assert hasattr(nltools, "__version__")
    print(f"       nltools {nltools.__version__}")
test("import nltools", test_import_nltools)

def test_import_submodules():
    import nltools.algorithms, nltools.data, nltools.io, nltools.mask
    import nltools.plotting, nltools.utils
test("import submodules", test_import_submodules)

# ------------------------------------------------------------------
# Data classes from in-memory arrays
# ------------------------------------------------------------------
print("\\n[data classes]")

_bd = None

def test_braindata_constructs():
    global _bd
    import nibabel as nib
    from nltools.data import BrainData
    np.random.seed(0)
    affine = np.eye(4)
    img = nib.Nifti1Image(np.random.randn(3, 2, 1, 5).astype(np.float32), affine)
    mask = nib.Nifti1Image(np.ones((3, 2, 1), dtype=np.float32), affine)
    _bd = BrainData(img, mask=mask)
    assert _bd.shape == (5, 6), f"got {_bd.shape}"
test("BrainData from a Nifti1Image", test_braindata_constructs)

def test_braindata_mean():
    assert _bd.mean().shape[-1] == 6
test("BrainData.mean()", test_braindata_mean)

def test_adjacency():
    from nltools.data import Adjacency
    mat = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]])
    assert Adjacency(mat, matrix_type="similarity").n_nodes == 3
test("Adjacency(similarity)", test_adjacency)

def test_designmatrix():
    from nltools.data import DesignMatrix
    dm = DesignMatrix(np.random.randint(0, 2, size=(20, 3)),
                      columns=["a", "b", "c"], sampling_freq=0.5)
    assert dm.shape == (20, 3), f"got {dm.shape}"
test("DesignMatrix(numpy)", test_designmatrix)

def test_horizontal_append():
    """The polars floor: horizontal concat has two names across 1.33-1.44."""
    from nltools.data import DesignMatrix
    a = DesignMatrix({"a": [1.0, 2.0]}, sampling_freq=1.0)
    b = DesignMatrix({"b": [3.0, 4.0]}, sampling_freq=1.0)
    assert a.append(b, axis=1).shape == (2, 2)
test("DesignMatrix.append(axis=1)", test_horizontal_append)

def test_joblib_parallel():
    """n_jobs=-1 degrades to sequential under wasm rather than raising."""
    from joblib import Parallel, delayed
    assert Parallel(n_jobs=-1)(delayed(lambda i: i * i)(i) for i in range(4)) == [0, 1, 4, 9]
test("joblib Parallel(n_jobs=-1)", test_joblib_parallel)

def test_himalaya_numpy_backend():
    from himalaya.backend import set_backend
    set_backend("numpy")
test("himalaya numpy backend", test_himalaya_numpy_backend)

# ------------------------------------------------------------------
# The template fetch path: direct GET into the IDBFS cache
# ------------------------------------------------------------------
print("\\n[fetch_resource]")

import nltools.templates.fetch as _fetch
from nltools.templates import fetch_resource, list_resources

_relpath = "masks/k88_parcel_names.csv"

def test_jspi_available():
    from pyodide.ffi import can_run_sync
    assert can_run_sync(), "no Promise Integration: fetch_resource uses the requests fallback"
test("Promise Integration available", test_jspi_available)

def test_fetch_resource_downloads():
    import os
    path = fetch_resource(_relpath)
    assert os.path.exists(path), f"missing: {path}"
    assert os.path.getsize(path) > 0, f"empty: {path}"
    print(f"       {path} ({os.path.getsize(path)} bytes)")
test("fetch_resource downloads via pyfetch", test_fetch_resource_downloads)

def test_missing_resource_raises_cleanly():
    import os
    try:
        fetch_resource("masks/definitely-not-a-file.nii.gz")
    except RuntimeError as e:
        assert "404" in str(e), f"unhelpful message: {e}"
        assert not os.path.exists(str(_fetch._PYODIDE_CACHE_ROOT / "masks/definitely-not-a-file.nii.gz"))
        return
    raise AssertionError("expected RuntimeError for a missing file")
test("a missing file raises and leaves no cache entry", test_missing_resource_raises_cleanly)

# Simulate a page reload: unmount IDBFS, drop the mounted flag, remount. The
# file must reappear from IndexedDB without any network call.
import pyodide_js as _pyodide_js

def test_cache_survives_remount():
    import os
    cached = fetch_resource(_relpath)
    _pyodide_js.FS.unmount(str(_fetch._PYODIDE_CACHE_ROOT.parent))
    _fetch._idbfs_mounted = False
    _fetch.fetch_resource.cache_clear()
    assert not os.path.exists(cached), f"unmount left an in-memory copy: {cached}"
    path = fetch_resource(_relpath)
    assert os.path.exists(path) and os.path.getsize(path) > 0
test("cache survives a remount (IndexedDB)", test_cache_survives_remount)
`;

    const listResourcesCode = `
# list_resources goes through huggingface_hub, whose client imports httpcore
# lazily. Nothing in the browser environment pulls it in, which is what the
# docstring's Note tells users.
def test_list_resources_needs_httpcore():
    try:
        list_resources(prefix="masks/")
    except ModuleNotFoundError as e:
        assert "httpcore" in str(e), f"unexpected missing module: {e}"
        return
    raise AssertionError("expected httpcore to be missing before it is installed")
test("list_resources reports the missing httpcore", test_list_resources_needs_httpcore)

import micropip
await micropip.install("httpcore")

def test_list_resources_works_with_httpcore():
    files = list_resources(prefix="masks/")
    assert _relpath in files, f"{_relpath} not in {len(files)} listed files"
    print(f"       {len(files)} files under masks/")
test("list_resources works once httpcore is installed", test_list_resources_works_with_httpcore)
`;

    const summaryCode = `
passed = sum(1 for _, ok, _ in results if ok)
print()
print("=" * 60)
print(f"Results: {passed}/{len(results)} tests passed")
print("=" * 60)

failed = [(n, e) for n, ok, e in results if not ok]
if failed:
    print("\\nFailed tests:")
    for name, err in failed:
        print(f"  {name}: {err}")
    print()

print("ALL TESTS PASSED" if not failed else "SOME TESTS FAILED")
not failed
`;

    await pyodide.runPythonAsync(testCode);
    await pyodide.runPythonAsync(listResourcesCode);
    const allPassed = await pyodide.runPythonAsync(summaryCode);
    process.exit(allPassed ? 0 : 1);
}

main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
});
