/**
 * Pyodide/WASM smoke test runner for nltools.
 *
 * Verifies that the core (non-optional) install works in a browser/Pyodide
 * environment: imports succeed, the main data classes construct from in-memory
 * numpy arrays, and optional-extras gating behaves.
 *
 * Exit code 0 = all tests passed, 1 = some tests failed.
 *
 * Usage: node nltools/tests/pyodide/test_runner.mjs
 * Prereq: a wheel built in dist/ (run `uv build --wheel` first).
 */

// Polyfill IndexedDB so Pyodide's IDBFS works in node — required for the
// persistent-cache tests below. Must happen before loadPyodide().
import "fake-indexeddb/auto";

import { loadPyodide } from "pyodide";
import { readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Project root is 3 levels up from nltools/tests/pyodide/
const projectRoot = join(__dirname, "..", "..", "..");

async function main() {
    console.log("=".repeat(60));
    console.log("nltools Pyodide/WASM smoke test runner");
    console.log("=".repeat(60));
    console.log();

    console.log("Loading Pyodide...");
    const pyodide = await loadPyodide();
    console.log("Pyodide loaded\n");

    console.log("Installing dependencies via micropip...");
    await pyodide.loadPackage("micropip");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install([
    "nibabel",
    "nilearn",
    "polars",
    "pynv",
    "seaborn",
])
print("Dependencies installed")
`);

    console.log("\nLoading nltools wheel...");
    const distDir = join(projectRoot, "dist");
    const wheels = readdirSync(distDir).filter(
        f => f.endsWith(".whl") && f.startsWith("nltools")
    );

    if (wheels.length === 0) {
        console.error("ERROR: No nltools wheel found in dist/");
        console.error("Run: uv build --wheel");
        process.exit(1);
    }

    // Prefer the newest wheel if multiple exist.
    wheels.sort().reverse();
    const wheelPath = join(distDir, wheels[0]);
    console.log(`Loading: ${wheels[0]}`);

    const wheelBuffer = readFileSync(wheelPath);
    const wheelData = new Uint8Array(wheelBuffer);
    await pyodide.unpackArchive(wheelData, "wheel");
    console.log("nltools installed\n");

    console.log("=".repeat(60));
    console.log("Running smoke tests");
    console.log("=".repeat(60));
    console.log();

    const testCode = `
import sys
import numpy as np
print(f"Platform: {sys.platform}")

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  OK  {name}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# 1. Top-level imports
# ------------------------------------------------------------------
print("\\n[imports]")

def test_import_nltools():
    import nltools
    assert hasattr(nltools, "__version__")
test("import nltools", test_import_nltools)

def test_import_submodules():
    import nltools.data, nltools.stats, nltools.utils, nltools.mask
    import nltools.plotting, nltools.algorithms, nltools.io
test("import submodules", test_import_submodules)

def test_public_api_names():
    from nltools import BrainData, Adjacency, DesignMatrix
    assert all(x is not None for x in (BrainData, Adjacency, DesignMatrix))
test("public API names importable", test_public_api_names)

# ------------------------------------------------------------------
# 2. BrainData from an in-memory Nifti1Image
# ------------------------------------------------------------------
print("\\n[BrainData]")

import nibabel as nib
from nltools.data import BrainData

# Small 4D volume with an all-true mask — no file I/O.
np.random.seed(0)
_spatial = (3, 2, 1)
_n_samples = 5
_affine = np.eye(4)
_img = nib.Nifti1Image(np.random.randn(*_spatial, _n_samples).astype(np.float32), _affine)
_mask = nib.Nifti1Image(np.ones(_spatial, dtype=np.float32), _affine)
_bd = BrainData(_img, mask=_mask)

_n_vox = int(np.prod(_spatial))

def test_braindata_shape():
    assert _bd.shape == (_n_samples, _n_vox), f"got {_bd.shape}"
test("BrainData.shape", test_braindata_shape)

def test_braindata_mean():
    m = _bd.mean()
    assert m.shape[-1] == _n_vox, f"got {m.shape}"
test("BrainData.mean()", test_braindata_mean)

def test_braindata_std():
    s = _bd.std()
    assert s.shape[-1] == _n_vox, f"got {s.shape}"
test("BrainData.std()", test_braindata_std)

# ------------------------------------------------------------------
# 3. Adjacency from a numpy matrix
# ------------------------------------------------------------------
print("\\n[Adjacency]")

from nltools.data import Adjacency

def test_adjacency_similarity():
    mat = np.array([[1.0, 0.5, 0.1],
                    [0.5, 1.0, 0.3],
                    [0.1, 0.3, 1.0]])
    adj = Adjacency(mat, matrix_type="similarity")
    assert adj.n_nodes == 3, f"got n_nodes={adj.n_nodes}"
test("Adjacency(similarity)", test_adjacency_similarity)

def test_adjacency_distance():
    mat = np.array([[0.0, 1.0, 2.0],
                    [1.0, 0.0, 1.5],
                    [2.0, 1.5, 0.0]])
    adj = Adjacency(mat, matrix_type="distance")
    assert adj.n_nodes == 3, f"got n_nodes={adj.n_nodes}"
test("Adjacency(distance)", test_adjacency_distance)

# ------------------------------------------------------------------
# 4. DesignMatrix from numpy
# ------------------------------------------------------------------
print("\\n[DesignMatrix]")

from nltools.data import DesignMatrix

def test_designmatrix_numpy():
    arr = np.random.randint(0, 2, size=(20, 3))
    dm = DesignMatrix(arr, columns=["a", "b", "c"], sampling_freq=0.5)
    assert dm.shape == (20, 3), f"got {dm.shape}"
test("DesignMatrix(numpy)", test_designmatrix_numpy)

# ------------------------------------------------------------------
# 5. HF dataset fetcher (pyodide async seed → sync fetch)
# ------------------------------------------------------------------
print("\\n[fetch_resource]")

from nltools.templates import fetch_resource, seed_resources, resolve_paths

_relpath = "default/2mm-MNI152-2009fsl-mask.nii.gz"

# Sync access without seeding should raise a clear error.
def test_fetch_unseeded_raises():
    try:
        fetch_resource(_relpath)
    except RuntimeError as e:
        assert "seed_resources" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("expected RuntimeError when cache is empty")
test("fetch_resource raises before seed", test_fetch_unseeded_raises)

# Seed and verify the file lands in the cache.
await seed_resources([_relpath])

def test_fetch_after_seed():
    import os
    path = fetch_resource(_relpath)
    assert os.path.exists(path), f"missing: {path}"
    assert os.path.getsize(path) > 0, f"empty: {path}"
test("fetch_resource after seed", test_fetch_after_seed)

# resolve_paths(default, 2) needs all three of mask/brain/T1.
await seed_resources([
    "default/2mm-MNI152-2009fsl-mask.nii.gz",
    "default/2mm-MNI152-2009fsl-brain.nii.gz",
    "default/2mm-MNI152-2009fsl-T1.nii.gz",
])

def test_resolve_paths_after_seed():
    import os
    paths = resolve_paths("default", 2)
    assert set(paths) == {"mask", "brain", "plot"}, f"keys={set(paths)}"
    for k, p in paths.items():
        assert os.path.exists(p), f"{k} missing: {p}"
test("resolve_paths after seed", test_resolve_paths_after_seed)

# Persistence: simulate a page reload by unmounting IDBFS and resetting
# state. After remount + syncfs(true), the file should reappear without
# any network call — proving the cache survives in IndexedDB.
import nltools.templates.fetch as _fetch
import pyodide_js as _pyodide_js

_seeded_path = fetch_resource(_relpath)
_mount_point = str(_fetch._PYODIDE_CACHE_ROOT.parent)
_pyodide_js.FS.unmount(_mount_point)
_fetch._idbfs_mounted = False

def test_unmount_clears_memfs():
    import os
    assert not os.path.exists(_seeded_path), (
        f"unmount should drop in-memory copy: {_seeded_path}"
    )
test("unmount clears MEMFS copy", test_unmount_clears_memfs)

await _fetch._ensure_idbfs_mounted()

def test_idbfs_persistence():
    import os
    assert os.path.exists(_seeded_path), (
        f"file did not survive remount via IDB: {_seeded_path}"
    )
    assert os.path.getsize(_seeded_path) > 0
test("file restored from IndexedDB after remount", test_idbfs_persistence)

# ------------------------------------------------------------------
# 6. Optional-extras gating
# ------------------------------------------------------------------
# These should be NOT installed in the default Pyodide env — the split in
# pyproject.toml moved them out of the core deps. We only verify import
# fails cleanly; any code path that needs them should raise ImportError.
print("\\n[optional extras gating]")

def test_h5py_not_installed():
    try:
        import h5py  # noqa: F401
    except ImportError:
        return
    raise AssertionError("h5py should NOT be installed in the core Pyodide env")
test("h5py absent from core env", test_h5py_not_installed)

def test_networkx_not_installed():
    try:
        import networkx  # noqa: F401
    except ImportError:
        return
    raise AssertionError("networkx should NOT be installed in the core Pyodide env")
test("networkx absent from core env", test_networkx_not_installed)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"Results: {passed}/{total} tests passed")
print("=" * 60)

failed = [(n, e) for n, ok, e in results if not ok]
if failed:
    print("\\nFailed tests:")
    for name, err in failed:
        print(f"  {name}: {err}")
    print()

all_passed = all(ok for _, ok, _ in results)
print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
all_passed
`;

    const allPassed = await pyodide.runPythonAsync(testCode);

    process.exit(allPassed ? 0 : 1);
}

main().catch(err => {
    console.error("Fatal error:", err);
    process.exit(1);
});
