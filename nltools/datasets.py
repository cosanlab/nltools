"""
NeuroLearn datasets
===================

Functions to help download datasets from Neurovault and other sources.

"""

__all__ = [
    "download_nifti",
    "fetch_emotion_ratings",
    "fetch_haxby",
    "fetch_neurovault_collection",
    "fetch_pain",
    "load_haxby_example",
]

import os
import sys
import logging
from pathlib import Path
from nltools.data import BrainData
from nltools.io import onsets_to_dm

# Core dependencies
from nilearn.datasets import fetch_neurovault_ids

try:
    from nilearn.datasets import fetch_haxby as nilearn_fetch_haxby
except ImportError:
    nilearn_fetch_haxby = None

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


def fetch_haxby(
    n_subjects=1,
    data_dir=None,
    verbose=1,
    mask="haxby_mask",
    resample=False,
):
    """Download and load Haxby2001 dataset from nilearn.

    Args:
        n_subjects (int, None, or 'all'): Which subject to load (1-6), or None/'all' for all subjects.
            Default: 1.
            - `n_subjects=1`: Returns all runs for subject 1
            - `n_subjects=2`: Returns all runs for subject 2
            - `n_subjects=None` or `'all'`: Returns all runs for all subjects (nested lists)
        data_dir (str, optional): Directory to store downloaded data. Default: None
        verbose (int, optional): Verbosity level. Default: 1
        mask (str, nibabel.Nifti1Image, or None, default="haxby_mask"): Brain mask to use.
            - `"haxby_mask"`: Use the default mask provided with the Haxby dataset (default)
            - `None`: Use default MNI template mask
            - Other: Passed directly to BrainData (file path, nibabel object, etc.)
        resample (bool, default=False): Whether to automatically resample data to mask space.
            See BrainData.__init__() for details.

    Returns:
        tuple:
            - If n_subjects is int: (list of BrainData, list of DesignMatrix) - all runs for that subject
            - If n_subjects is None or 'all': (list of lists of BrainData, list of lists of DesignMatrix)
              First level: subjects, second level: runs per subject

    Examples:
        >>> # Load all runs for subject 1
        >>> brain_data, design_matrix = fetch_haxby(n_subjects=1)
        >>> len(brain_data)  # Number of runs
        >>>
        >>> # Load all runs for subject 2
        >>> brain_data, design_matrix = fetch_haxby(n_subjects=2)
        >>>
        >>> # Load all runs for all subjects
        >>> brain_data_nested, design_matrix_nested = fetch_haxby(n_subjects='all')
        >>> len(brain_data_nested)  # Number of subjects
        >>> len(brain_data_nested[0])  # Number of runs for first subject
    """
    import pandas as pd

    if nilearn_fetch_haxby is None:
        raise ImportError("nilearn package is required for fetch_haxby")

    # Validate n_subjects parameter
    if isinstance(n_subjects, int):
        if n_subjects < 1 or n_subjects > 6:
            raise ValueError(f"n_subjects must be between 1 and 6 (got {n_subjects})")
        subject_list = [n_subjects]
    elif n_subjects in (None, "all"):
        subject_list = list(range(1, 7))  # Subjects 1-6
    else:
        raise ValueError(
            f"n_subjects must be an int (1-6), None, or 'all' (got {type(n_subjects).__name__})"
        )

    # Haxby dataset TR is 2.5 seconds
    TR = 2.5

    # Process each subject
    all_subjects_brain_data = []
    all_subjects_design_matrix = []

    for subject_id in subject_list:
        # Download data for this subject
        # Suppress nilearn's "[fetch_haxby] Dataset found..." message
        # This message is printed even when verbose=0, so we suppress all output
        # Save original stdout, stderr, and logging level
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        nilearn_logger = logging.getLogger("nilearn")
        old_log_level = nilearn_logger.level if nilearn_logger.level else logging.NOTSET

        try:
            # Redirect both stdout and stderr to devnull
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115
            sys.stderr = open(os.devnull, "w")  # noqa: SIM115
            # Suppress nilearn logging
            nilearn_logger.setLevel(logging.CRITICAL)

            nilearn_data = nilearn_fetch_haxby(
                data_dir=data_dir,
                subjects=(subject_id,),
                fetch_stimuli=False,
                verbose=0,  # Force verbose=0 to minimize output
            )
        finally:
            # Restore stdout, stderr, and logging level
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            nilearn_logger.setLevel(old_log_level)

        # Process each run
        subject_brain_data = []
        subject_design_matrix = []

        # Haxby dataset structure: one func file per subject, but labels are split by chunks (runs)
        # We need to split the func file by chunks or use the labels to create separate runs

        # For now, let's assume we create one BrainData per chunk (run)
        # Load the full func file
        func_file = nilearn_data.func[0]
        mask_file = nilearn_data.mask if mask == "haxby_mask" else mask

        # Load full BrainData
        full_brain_data = BrainData(
            func_file, mask=mask_file, verbose=0, resample=resample
        )

        # Parse labels to get events per chunk
        labels_file = nilearn_data.session_target[0]

        # Read labels to determine number of chunks and TRs per chunk
        with open(labels_file) as f:
            lines = f.readlines()

        chunks = {}
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) >= 2:
                chunk_num = int(parts[1])
                if chunk_num not in chunks:
                    chunks[chunk_num] = []
                chunks[chunk_num].append(parts[0])

        # Process each chunk as a separate run
        for chunk_num in sorted(chunks.keys()):
            chunk_labels = chunks[chunk_num]
            run_length = len(chunk_labels)

            # Get the TR range for this chunk
            # We need to extract the corresponding timepoints from full_brain_data
            # For now, let's create events for this chunk and use onsets_to_dm
            # Then we'll match the BrainData accordingly

            # Build events for this chunk
            events_list = []
            current_label = None
            start_idx = None

            for tr_idx, label in enumerate(chunk_labels):
                if label != current_label:
                    if current_label is not None and current_label != "rest":
                        events_list.append(
                            {
                                "onset": start_idx * TR,
                                "duration": (tr_idx - start_idx) * TR,
                                "trial_type": current_label,
                            }
                        )
                    current_label = label
                    start_idx = tr_idx

            # Handle last event
            if current_label is not None and current_label != "rest":
                events_list.append(
                    {
                        "onset": start_idx * TR,
                        "duration": (run_length - start_idx) * TR,
                        "trial_type": current_label,
                    }
                )

            if events_list:
                events_df = pd.DataFrame(events_list)

                # Create DesignMatrix using onsets_to_dm
                design_matrix = onsets_to_dm(
                    timings=events_df, run_length=run_length, TR=TR, hrf_model="glover"
                )

                # Extract corresponding timepoints from BrainData
                # We need to find the start index for this chunk
                # Chunks are sequential in the labels file
                chunk_start_idx = sum(
                    len(chunks[c]) for c in sorted(chunks.keys()) if c < chunk_num
                )
                chunk_end_idx = chunk_start_idx + run_length

                # Extract BrainData for this chunk
                chunk_brain_data = full_brain_data[chunk_start_idx:chunk_end_idx]

                subject_brain_data.append(chunk_brain_data)
                subject_design_matrix.append(design_matrix)

        all_subjects_brain_data.append(subject_brain_data)
        all_subjects_design_matrix.append(subject_design_matrix)

    # Return format based on number of subjects
    if len(subject_list) == 1:
        # Single subject: return flat lists
        return all_subjects_brain_data[0], all_subjects_design_matrix[0]
    # Multiple subjects: return nested lists
    return all_subjects_brain_data, all_subjects_design_matrix


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

    Matches the return shape of `fetch_haxby(n_subjects=1)` — paired lists of
    `BrainData` and `DesignMatrix`, one entry per run — but generates a tiny
    synthetic volume (10 x 10 x 5 = 500 voxels) with condition-specific signal
    injected into disjoint voxel clusters. No network I/O, no disk I/O, no
    nilearn fetcher dependency. Runs in well under a second.

    Intended for tutorials, documentation examples, and Pyodide / in-browser
    environments where `fetch_haxby` cannot download the real dataset. The
    eight conditions match the real Haxby object-recognition experiment
    (face, house, cat, bottle, scissors, shoe, chair, scrambledpix), arranged
    in a randomized 9-TR block design with TR=2.5s.

    Args:
        n_runs (int): Number of runs to generate. Default 1.
        random_state (int | None): Seed for reproducible output. Default 42.

    Returns:
        tuple: `(list[BrainData], list[DesignMatrix])`, each of length n_runs.
            The DesignMatrix columns are the eight condition names plus a
            "constant" column, matching `fetch_haxby`'s output shape.

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
