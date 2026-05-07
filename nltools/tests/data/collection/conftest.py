"""BrainCollection-specific test fixtures.

Lightweight fixtures kept tiny so signature/contract tests are fast. The
``fake_bids_root`` fixture leans on ``nilearn._utils.data_gen`` for a real
BIDS-shaped tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

# BrainCollection is being rewritten on the `collection-imp` branch; skip the
# whole test directory here so this branch's suite isn't tied to scaffold-era
# contracts that will change. Remove after collection-imp merges back.
collect_ignore_glob = ["test_*.py"]


# ---------------------------------------------------------------------------
# Mask + data builders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tiny_mask() -> nib.Nifti1Image:
    """3×3×3 mask with 27 voxels (all True)."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    data = np.ones((3, 3, 3), dtype=np.int8)
    return nib.Nifti1Image(data, affine)


@pytest.fixture(scope="function")
def tiny_mask_path(tmp_path: Path, tiny_mask: nib.Nifti1Image) -> Path:
    p = tmp_path / "mask.nii.gz"
    nib.save(tiny_mask, p)
    return p


@pytest.fixture(scope="function")
def tiny_brain_factory(tiny_mask):
    """Factory: ``tiny_brain_factory(n_obs=10, seed=0)`` → ``BrainData``."""
    from nltools.data import BrainData

    def _make(n_obs: int = 10, seed: int = 0) -> BrainData:
        rng = np.random.default_rng(seed)
        spatial = tiny_mask.shape  # (3, 3, 3)
        vol = rng.standard_normal(spatial + (n_obs,)).astype(np.float32)
        img = nib.Nifti1Image(vol, tiny_mask.affine)
        return BrainData(img, mask=tiny_mask)

    return _make


@pytest.fixture(scope="function")
def tiny_nifti_paths(tmp_path: Path, tiny_mask, tiny_brain_factory) -> list[Path]:
    """Three small NIfTI files on disk (one per 'subject')."""
    paths: list[Path] = []
    for i in range(3):
        rng = np.random.default_rng(i)
        vol = rng.standard_normal(tiny_mask.shape + (8,)).astype(np.float32)
        img = nib.Nifti1Image(vol, tiny_mask.affine)
        p = tmp_path / f"sub-{i + 1:02d}.nii.gz"
        nib.save(img, p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# BIDS scaffolding (only used by from_bids tests; gated on nilearn)
# ---------------------------------------------------------------------------


def _has_nilearn_fake_bids() -> bool:
    try:
        from nilearn._utils.data_gen import create_fake_bids_dataset  # noqa: F401

        return True
    except Exception:
        return False


HAS_FAKE_BIDS = _has_nilearn_fake_bids()


@pytest.fixture(scope="function")
def fake_bids_root(tmp_path: Path) -> Path:
    """Build a minimal fake BIDS dataset via nilearn. Returns the dataset root."""
    if not HAS_FAKE_BIDS:
        pytest.skip("nilearn fake-BIDS helper not available")
    from nilearn._utils.data_gen import create_fake_bids_dataset

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        name = create_fake_bids_dataset(
            base_dir=str(tmp_path),
            n_sub=2,
            n_ses=1,
            tasks=["task01"],
            n_runs=[1],
            with_derivatives=True,
            with_confounds=True,
            n_voxels=4,
        )
    finally:
        os.chdir(cwd)
    return tmp_path / name


# ---------------------------------------------------------------------------
# Pre-built collections (in-memory + path-backed) — for tests that don't
# need to exercise construction itself.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def bc_inmem(tiny_mask, tiny_brain_factory):
    """Three-subject in-memory collection.

    Returns a ``BrainCollection`` already constructed with loaded ``BrainData``.
    Skipped if ``__init__`` is still a stub.
    """
    from nltools.data import BrainCollection

    brains = [tiny_brain_factory(n_obs=8, seed=i) for i in range(3)]
    try:
        return BrainCollection(brains, mask=tiny_mask, lazy=False, cache_dir=None)
    except NotImplementedError:
        pytest.skip("BrainCollection.__init__ not yet implemented")


@pytest.fixture(scope="function")
def bc_pathbacked(tiny_mask, tiny_nifti_paths, tmp_path):
    """Three-subject collection constructed from on-disk paths."""
    from nltools.data import BrainCollection

    try:
        return BrainCollection.from_paths(
            tiny_nifti_paths, mask=tiny_mask, cache_dir=tmp_path / "cache"
        )
    except NotImplementedError:
        pytest.skip("BrainCollection.from_paths not yet implemented")


@pytest.fixture(scope="function")
def tiny_design_factory():
    """Factory: ``tiny_design_factory(n_obs=8, seed=0)`` → 2-col DesignMatrix."""
    import pandas as pd

    from nltools.data import DesignMatrix

    def _make(n_obs: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Two non-collinear regressors so design_clean keeps both columns.
        t = np.linspace(0, 2 * np.pi, n_obs)
        return DesignMatrix(
            pd.DataFrame(
                {
                    "a": np.sin(t) + 0.1 * rng.standard_normal(n_obs),
                    "b": np.cos(t) + 0.1 * rng.standard_normal(n_obs),
                }
            ),
            TR=2.0,
        )

    return _make


@pytest.fixture(scope="function")
def bc_with_designs(tiny_mask, tiny_brain_factory, tiny_design_factory, tmp_path):
    """Three-subject in-memory collection with paired DesignMatrix per item."""
    from nltools.data import BrainCollection

    brains = [tiny_brain_factory(n_obs=8, seed=i) for i in range(3)]
    designs = [tiny_design_factory(n_obs=8, seed=10 + i) for i in range(3)]
    try:
        return BrainCollection(
            brains,
            mask=tiny_mask,
            designs=designs,
            lazy=False,
            cache_dir=tmp_path / "cache",
        )
    except NotImplementedError:
        pytest.skip("BrainCollection.__init__ not yet implemented")
