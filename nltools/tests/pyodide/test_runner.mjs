/**
 * Pyodide/WASM smoke test runner for nltools.
 *
 * Checks the things that only break in the browser: that every declared
 * dependency resolves under Pyodide, that the wheel imports, that the data
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
import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Project root is three levels up from nltools/tests/pyodide/.
const projectRoot = join(__dirname, "..", "..", "..");

// nltools' declared runtime dependencies, unpinned: micropip takes each from
// the Pyodide lockfile when it is bundled and from PyPI otherwise. Pinning any
// of them would hide exactly the resolution failures this runner exists to
// catch.
const DEPENDENCIES = [
    "anywidget",
    "h5py",
    "himalaya",
    "huggingface_hub",
    "joblib",
    "matplotlib",
    "nibabel",
    "nilearn",
    "numpy",
    "pandas",
    "polars",
    "pynv",
    "scikit-learn",
    "scipy",
    "seaborn",
    "tqdm",
];

function findWheel() {
    const distDir = join(projectRoot, "dist");
    const wheels = readdirSync(distDir).filter(
        (f) => f.endsWith(".whl") && f.startsWith("nltools")
    );
    if (wheels.length === 0) {
        console.error("ERROR: no nltools wheel in dist/. Run: uv build --wheel");
        process.exit(1);
    }
    wheels.sort().reverse();
    return join(distDir, wheels[0]);
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

    console.log("\nInstalling declared dependencies via micropip...");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(DEPENDENCIES)})
`);
    console.log("Dependencies installed");

    console.log("Installing the nltools wheel...");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/wheel/${wheelName}", deps=False)
`);
    console.log("nltools installed");

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
# Dependency resolution — the thing the floors exist for
# ------------------------------------------------------------------
print("\\n[dependencies]")

import micropip

def test_declared_requirements_resolve():
    installed = {p.name.replace("_", "-"): p.version for p in micropip.list().values()}
    for name in ["polars", "h5py", "nilearn", "nibabel", "himalaya", "anywidget"]:
        assert name in installed, f"{name} did not install"
    print(f"       polars {installed['polars']}, h5py {installed['h5py']}")
test("every declared dependency installed", test_declared_requirements_resolve)

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

import nibabel as nib
from nltools.data import Adjacency, BrainData, DesignMatrix

_spatial = (3, 2, 1)
_n_samples = 5
_n_vox = int(np.prod(_spatial))
np.random.seed(0)
_affine = np.eye(4)
_img = nib.Nifti1Image(np.random.randn(*_spatial, _n_samples).astype(np.float32), _affine)
_mask = nib.Nifti1Image(np.ones(_spatial, dtype=np.float32), _affine)
_bd = BrainData(_img, mask=_mask)

def test_braindata_shape():
    assert _bd.shape == (_n_samples, _n_vox), f"got {_bd.shape}"
test("BrainData.shape", test_braindata_shape)

def test_braindata_mean():
    assert _bd.mean().shape[-1] == _n_vox
test("BrainData.mean()", test_braindata_mean)

def test_adjacency():
    mat = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.0]])
    assert Adjacency(mat, matrix_type="similarity").n_nodes == 3
test("Adjacency(similarity)", test_adjacency)

def test_designmatrix():
    arr = np.random.randint(0, 2, size=(20, 3))
    dm = DesignMatrix(arr, columns=["a", "b", "c"], sampling_freq=0.5)
    assert dm.shape == (20, 3), f"got {dm.shape}"
test("DesignMatrix(numpy)", test_designmatrix)

def test_horizontal_append():
    """The polars floor: horizontal concat has two names across 1.33-1.44."""
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

def test_fetch_resource_downloads():
    import os
    path = fetch_resource(_relpath)
    assert os.path.exists(path), f"missing: {path}"
    assert os.path.getsize(path) > 0, f"empty: {path}"
    print(f"       {path} ({os.path.getsize(path)} bytes)")
test("fetch_resource downloads via pyfetch", test_fetch_resource_downloads)

def test_list_resources_raises():
    try:
        list_resources()
    except RuntimeError as e:
        assert "fetch_resource" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("expected RuntimeError under Pyodide")
test("list_resources raises under Pyodide", test_list_resources_raises)

# Simulate a page reload: unmount IDBFS, drop the mounted flag, remount. The
# file must reappear from IndexedDB without any network call.
import pyodide_js as _pyodide_js

_cached_path = fetch_resource(_relpath)
_pyodide_js.FS.unmount(str(_fetch._PYODIDE_CACHE_ROOT.parent))
_fetch._idbfs_mounted = False
_fetch.fetch_resource.cache_clear()

def test_unmount_clears_memfs():
    import os
    assert not os.path.exists(_cached_path), f"unmount left a copy: {_cached_path}"
test("unmount drops the in-memory copy", test_unmount_clears_memfs)

def test_cache_survives_remount():
    import os
    path = fetch_resource(_relpath)
    assert os.path.exists(path) and os.path.getsize(path) > 0
test("cache survives a remount (IndexedDB)", test_cache_survives_remount)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
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

    const allPassed = await pyodide.runPythonAsync(testCode);
    process.exit(allPassed ? 0 : 1);
}

main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
});
