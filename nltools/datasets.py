"""Dataset download and example-data utilities.

Functions to help download example datasets. The curated example datasets
(`fetch_pain`, `fetch_emotion_ratings`) are hosted on the ``nltools/niftis``
Hugging Face dataset and resolve through the same `fetch_resource` /
`seed_resources` machinery as the MNI templates and atlases, so they work both
on a normal Python kernel and in Pyodide / JupyterLite (pre-seed with
`seed_resources` there). Arbitrary Neurovault collections are still available
via `fetch_neurovault_collection`.

"""

__all__ = [
    "EMOTION_METADATA",
    "PAIN_RESOURCES",
    "download_nifti",
    "emotion_resources",
    "fetch_emotion_ratings",
    "fetch_neurovault_collection",
    "fetch_pain",
    "load_haxby_example",
]

from pathlib import Path
from nltools.data import BrainData, DesignMatrix
from nltools.data.designmatrix.io import events_to_dm
from nltools.templates import fetch_resource

# Core dependencies
from nilearn.datasets import fetch_neurovault_ids

# Optional dependencies
try:
    import requests
except ImportError:
    requests = None

# Curated pain dataset hosted on the ``nltools/niftis`` HF dataset. The 84
# images form a complete 28-subject x 3-pain-level grid; filenames are
# deterministic so the full resource list can be enumerated without a
# network round-trip (needed to pre-seed the Pyodide cache).
_PAIN_DIR = "datasets/pain"
_PAIN_LEVELS = ("low", "medium", "high")
PAIN_RESOURCES: list[str] = [f"{_PAIN_DIR}/metadata.csv"] + [
    f"{_PAIN_DIR}/sub-{sub:02d}_pain-{level}.nii.gz"
    for sub in range(1, 29)
    for level in _PAIN_LEVELS
]
"""Every `fetch_resource` relpath the pain dataset needs (metadata + 84 images).

Pass to `nltools.templates.seed_resources` before calling `fetch_pain` in
Pyodide / JupyterLite, where synchronous downloads are unavailable:
``await seed_resources(PAIN_RESOURCES)``.
"""

# Curated emotion-rating dataset hosted on the ``nltools/niftis`` HF dataset.
# Unlike pain, the 679 images are keyed by Neurovault id rather than a
# generable subject x condition grid, so the filename manifest is read from
# ``metadata.csv`` (see `emotion_resources`) instead of being a static list.
_EMOTION_DIR = "datasets/emotion_ratings"
EMOTION_METADATA = f"{_EMOTION_DIR}/metadata.csv"
"""Relpath of the emotion dataset's metadata table (its filename manifest)."""


def emotion_resources() -> list[str]:
    """List every `fetch_resource` relpath the emotion dataset needs.

    The emotion image filenames are keyed by Neurovault id (not a generable
    grid like `PAIN_RESOURCES`), so this reads `EMOTION_METADATA` to enumerate
    them. To pre-seed the Pyodide / JupyterLite cache, seed the metadata file
    first (it is read here), then seed the images:

    ```python
    await seed_resources([EMOTION_METADATA])
    await seed_resources(emotion_resources())
    ```

    Returns:
        list[str]: `[EMOTION_METADATA, ...679 image relpaths]`.
    """
    import polars as pl

    meta = pl.read_csv(fetch_resource(EMOTION_METADATA))
    return [EMOTION_METADATA] + [
        f"{_EMOTION_DIR}/{fn}" for fn in meta["filename"].to_list()
    ]


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
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
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


def fetch_pain(verbose=0):
    """Download and load the pain dataset from the nltools HF dataset.

    Loads the Chang et al. (2015) pain-perception study: 28 subjects x 3
    stimulus-intensity conditions = 84 whole-brain contrast images, with a
    curated metadata table (`SubjectID`, `PainLevel`, `PainIntensity`, `Age`,
    `Sex`, provenance `neurovault_id` / `name`).

    Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
    locally on first use, so this works on a normal Python kernel with no extra
    setup. In Pyodide / JupyterLite, pre-seed the cache first:
    ``await seed_resources(PAIN_RESOURCES)``.

    Args:
        verbose (int, optional): Verbosity passed to `BrainData` while loading.
            Default: 0

    Returns:
        BrainData: `BrainData` with the 84 images; `X` holds the metadata table.

    References:
        Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
        A sensitive and specific neural signature for picture-induced negative affect.
        PLoS biology, 13(6), e1002180.
    """
    import polars as pl

    try:
        metadata = pl.read_csv(fetch_resource(f"{_PAIN_DIR}/metadata.csv"))
        files = [fetch_resource(f"{_PAIN_DIR}/{fn}") for fn in metadata["filename"]]
        return BrainData(data=files, X=metadata, verbose=verbose)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch pain dataset: {e}")


def fetch_emotion_ratings(verbose=0):
    """Download and load the emotion-rating dataset from the nltools HF dataset.

    Loads the Chang et al. (2015) IAPS emotion-rating study: 679 whole-brain
    contrast images across 150 subjects, each rating images 1-5, with a
    built-in train/test holdout split. `X` carries the full portable Neurovault
    metadata (key columns: `SubjectID`, `Rating`, `Holdout`, `AGE`, `SEX`).

    Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
    locally on first use, so this works on a normal Python kernel with no extra
    setup. In Pyodide / JupyterLite, pre-seed the cache first (see
    `emotion_resources`).

    Args:
        verbose (int, optional): Verbosity passed to `BrainData` while loading.
            Default: 0

    Returns:
        BrainData: `BrainData` with the 679 images; `X` holds the metadata table.

    References:
        Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
        A sensitive and specific neural signature for picture-induced negative affect.
        PLoS biology, 13(6), e1002180.
    """
    import polars as pl

    try:
        metadata = pl.read_csv(fetch_resource(EMOTION_METADATA))
        files = [
            fetch_resource(f"{_EMOTION_DIR}/{fn}")
            for fn in metadata["filename"].to_list()
        ]
        return BrainData(data=files, X=metadata, verbose=verbose)

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
            The DesignMatrix columns are the eight condition names suffixed
            with ``_c0`` (HRF-convolved boxcars).

    Examples:
        >>> from nltools.datasets import load_haxby_example
        >>> brain_data, design_matrices = load_haxby_example()
        >>> data, dm = brain_data[0], design_matrices[0]
        >>> data.shape
        (72, 500)
        >>> "face_c0" in dm.columns
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
        dm_data = events_to_dm(
            events_df,
            run_length=n_timepoints,
            sampling_freq=1.0 / TR,
        )
        dm = DesignMatrix(dm_data, sampling_freq=1.0 / TR).convolve()

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
            cond_col = f"{cond}_c0"
            if cond_col not in dm.columns:
                continue
            regressor = np.asarray(dm[cond_col], dtype=np.float32)
            cluster = voxel_order[
                c_idx * voxels_per_cond : (c_idx + 1) * voxels_per_cond
            ]
            data_flat[cluster, :] += signal_strength * regressor

        data_4d = data_flat.reshape((*spatial_shape, n_timepoints))
        img = nib.Nifti1Image(data_4d, affine)
        brain_data_list.append(BrainData(img, mask=mask_img, verbose=0))
        design_matrix_list.append(dm)

    return brain_data_list, design_matrix_list
