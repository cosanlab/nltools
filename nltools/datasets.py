"""
NeuroLearn datasets
===================

Functions to help download datasets from Neurovault and other sources.

"""

__all__ = [
    "download_nifti",
    "fetch_emotion_ratings",
    "fetch_neurovault_collection",
    "fetch_pain",
    "load_haxby_example",
]

from pathlib import Path
from nltools.data import BrainData
from nltools.io import onsets_to_dm

# Core dependencies
from nilearn.datasets import fetch_neurovault_ids

# Optional dependencies
try:
    import requests
except ImportError:
    requests = None


def download_nifti(url, data_dir=None):
    """Download an image from a URL to a nifti file.

    Args:
        url (str): URL of the image to download
        data_dir (str, optional): Directory to save the file. If None, uses current directory.

    Returns:
        str: Path to the downloaded file

    Raises:
        ImportError: If requests is not available
        ValueError: If URL is invalid
    """
    if requests is None:
        raise ImportError("requests package is required for downloading files")

    if not url:
        raise ValueError("URL cannot be empty")
    if isinstance(url, Path):
        url = str(url)

    local_filename = url.split("/")[-1]
    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        local_filename = data_dir / local_filename

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    except requests.RequestException as e:
        raise ValueError(f"Failed to download {url}: {e}")

    return str(local_filename)


def fetch_neurovault_collection(collection_id, data_dir=None, verbose=1):
    """Download images and metadata from a Neurovault collection.

    This function uses the modern nilearn API to download collections from Neurovault.

    Args:
        collection_id (int): Neurovault collection ID
        data_dir (str, optional): Directory to store downloaded data.
            If None, uses nilearn's default data directory.
        verbose (int, optional): Verbosity level. Default: 1

    Returns:
        tuple: (metadata polars.DataFrame, list of image file paths)

    Raises:
        ValueError: If collection_id is invalid
        RuntimeError: If download fails
    """
    import polars as pl

    if not isinstance(collection_id, int) or collection_id <= 0:
        raise ValueError("collection_id must be a positive integer")

    try:
        nv_data = fetch_neurovault_ids(
            collection_ids=[collection_id], data_dir=data_dir, verbose=verbose
        )

        files = nv_data["images"]
        metadata = pl.DataFrame(nv_data["images_meta"])

        return metadata, files

    except Exception as e:
        raise RuntimeError(f"Failed to download collection {collection_id}: {e}")


def fetch_pain(data_dir=None, verbose=1):
    """Download and load pain dataset from Neurovault.

    This downloads the Chang et al. (2015) pain dataset from Neurovault collection 504.

    Args:
        data_dir (str, optional): Path of the data directory. Used to force data
            storage in a specified location. Default: None
        verbose (int, optional): Verbosity level. Default: 1

    Returns:
        BrainData: BrainData object with downloaded data. X=metadata

    References:
        Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
        A sensitive and specific neural signature for picture-induced negative affect.
        PLoS biology, 13(6), e1002180.
    """
    collection_id = 504

    try:
        metadata, files = fetch_neurovault_collection(
            collection_id=collection_id, data_dir=data_dir, verbose=verbose
        )
        return BrainData(data=files, X=metadata, verbose=0)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch pain dataset: {e}")


def fetch_emotion_ratings(data_dir=None, verbose=1):
    """Download and load emotion rating dataset from Neurovault.

    This downloads the Chang et al. (2015) emotion ratings dataset from
    Neurovault collection 1964.

    Args:
        data_dir (str, optional): Path of the data directory. Used to force data
            storage in a specified location. Default: None
        verbose (int, optional): Verbosity level. Default: 1

    Returns:
        BrainData: BrainData object with downloaded data. X=metadata

    References:
        Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
        A sensitive and specific neural signature for picture-induced negative affect.
        PLoS biology, 13(6), e1002180.
    """
    collection_id = 1964

    try:
        metadata, files = fetch_neurovault_collection(
            collection_id=collection_id, data_dir=data_dir, verbose=verbose
        )
        return BrainData(data=files, X=metadata, verbose=0)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch emotion ratings dataset: {e}")


_HAXBY_CONDITIONS = (
    "face",
    "house",
    "cat",
    "bottle",
    "scissors",
    "shoe",
    "chair",
    "scrambledpix",
)


def load_haxby_example(n_runs=1, random_state=42):
    """Load a small synthetic Haxby-like dataset, entirely in-memory.

    Returns paired lists of `BrainData` and `DesignMatrix`, one entry per
    run, generated from a tiny synthetic volume (10 x 10 x 5 = 500 voxels)
    with condition-specific signal injected into disjoint voxel clusters.
    No network I/O, no disk I/O, no nilearn fetcher dependency. Runs in
    well under a second.

    Intended for tutorials, documentation examples, and Pyodide / in-browser
    environments where downloading a real fMRI dataset is impractical. The
    eight conditions match the real Haxby 2001 object-recognition experiment
    (face, house, cat, bottle, scissors, shoe, chair, scrambledpix), arranged
    in a randomized 9-TR block design with TR=2.5s.

    Args:
        n_runs (int): Number of runs to generate. Default 1.
        random_state (int | None): Seed for reproducible output. Default 42.

    Returns:
        tuple: `(list[BrainData], list[DesignMatrix])`, each of length n_runs.
            The DesignMatrix columns are the eight condition names plus a
            "constant" column.

    Examples:
        >>> from nltools.datasets import load_haxby_example
        >>> brain_data, design_matrices = load_haxby_example()
        >>> data, dm = brain_data[0], design_matrices[0]
        >>> data.shape
        (72, 500)
        >>> "face" in dm.columns
        True
    """
    import numpy as np
    import pandas as pd
    import nibabel as nib

    rng = np.random.default_rng(random_state)

    TR = 2.5
    block_tr = 9  # 22.5s blocks, same order of magnitude as real Haxby
    n_conditions = len(_HAXBY_CONDITIONS)
    n_timepoints = block_tr * n_conditions  # 72 TRs per run
    spatial_shape = (10, 10, 5)
    n_voxels = int(np.prod(spatial_shape))
    affine = np.diag([3.0, 3.0, 3.0, 1.0]).astype(np.float32)
    mask_img = nib.Nifti1Image(np.ones(spatial_shape, dtype=np.float32), affine)

    brain_data_list = []
    design_matrix_list = []

    for _ in range(n_runs):
        order = list(rng.permutation(_HAXBY_CONDITIONS))
        events_df = pd.DataFrame(
            [
                {
                    "onset": i * block_tr * TR,
                    "duration": block_tr * TR,
                    "trial_type": cond,
                }
                for i, cond in enumerate(order)
            ]
        )
        dm = onsets_to_dm(
            timings=events_df,
            run_length=n_timepoints,
            TR=TR,
            hrf_model="glover",
        )

        # Positive BOLD-like baseline intensity + voxelwise Gaussian noise,
        # plus signal injected into a disjoint voxel cluster per condition
        # so contrasts produce real spatial patterns. The positive baseline
        # is required so percent-signal-change scaling inside GLM fits
        # (which divides by the voxel mean) stays numerically sane.
        baseline = 100.0
        noise_sd = 1.0
        signal_strength = 5.0  # BOLD units → clearly visible pattern in plots
        data_flat = (
            baseline + noise_sd * rng.standard_normal((n_voxels, n_timepoints))
        ).astype(np.float32)
        voxels_per_cond = n_voxels // n_conditions
        voxel_order = rng.permutation(n_voxels)
        for c_idx, cond in enumerate(_HAXBY_CONDITIONS):
            if cond not in dm.columns:
                continue
            regressor = np.asarray(dm[cond], dtype=np.float32)
            cluster = voxel_order[
                c_idx * voxels_per_cond : (c_idx + 1) * voxels_per_cond
            ]
            data_flat[cluster, :] += signal_strength * regressor

        data_4d = data_flat.reshape((*spatial_shape, n_timepoints))
        img = nib.Nifti1Image(data_4d, affine)
        brain_data_list.append(BrainData(img, mask=mask_img, verbose=0))
        design_matrix_list.append(dm)

    return brain_data_list, design_matrix_list
