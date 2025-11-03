"""
NeuroLearn datasets
===================

Functions to help download datasets from Neurovault and other sources.

"""

__all__ = [
    "download_nifti",
    "fetch_neurovault_collection",
    "fetch_emotion_ratings",
    "fetch_pain",
    "fetch_haxby",
    # Deprecated functions - kept for backward compatibility
    "get_collection_image_metadata",
    "download_collection",
]

import pandas as pd
import warnings
import os
import sys
import logging
from pathlib import Path
from nltools.data import BrainData
from nltools.file_reader import onsets_to_dm

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
        tuple: (metadata DataFrame, list of image file paths)

    Raises:
        ValueError: If collection_id is invalid
        RuntimeError: If download fails
    """
    if not isinstance(collection_id, int) or collection_id <= 0:
        raise ValueError("collection_id must be a positive integer")

    try:
        nv_data = fetch_neurovault_ids(
            collection_ids=[collection_id], data_dir=data_dir, verbose=verbose
        )

        files = nv_data["images"]
        metadata = pd.DataFrame(nv_data["images_meta"])

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

    # Helper function to convert Haxby labels to events format
    def _haxby_labels_to_events(labels_file, run_length):
        """Convert Haxby labels file to events DataFrame."""
        events_list = []

        # Read labels file
        with open(labels_file, "r") as f:
            lines = f.readlines()

        # Parse labels per chunk (run)
        chunks = {}
        for line in lines[1:]:  # Skip header 'labels chunks'
            parts = line.strip().split()
            if len(parts) >= 2:
                label = parts[0]
                chunk_num = int(parts[1])
                if chunk_num not in chunks:
                    chunks[chunk_num] = []
                chunks[chunk_num].append(label)

        # Convert each chunk to events (group consecutive TRs with same label)
        for chunk_num in sorted(chunks.keys()):
            labels = chunks[chunk_num]
            current_label = None
            start_idx = None

            for tr_idx, label in enumerate(labels):
                if label != current_label:
                    # End previous event (if exists and not rest)
                    if current_label is not None and current_label != "rest":
                        events_list.append(
                            {
                                "onset": start_idx * TR,
                                "duration": (tr_idx - start_idx) * TR,
                                "trial_type": current_label,
                            }
                        )
                    # Start new event
                    current_label = label
                    start_idx = tr_idx

            # Handle last event
            if current_label is not None and current_label != "rest":
                events_list.append(
                    {
                        "onset": start_idx * TR,
                        "duration": (len(labels) - start_idx) * TR,
                        "trial_type": current_label,
                    }
                )

        return (
            pd.DataFrame(events_list)
            if events_list
            else pd.DataFrame(columns=["onset", "duration", "trial_type"])
        )

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
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
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
        with open(labels_file, "r") as f:
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

            # Convert labels to events for this chunk
            events_df = _haxby_labels_to_events(labels_file, run_length)
            # Filter events for this chunk only (we need to track which chunk each event belongs to)
            # Actually, the helper function processes all chunks, so we need to filter by chunk

            # Better approach: process one chunk at a time
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
    else:
        # Multiple subjects: return nested lists
        return all_subjects_brain_data, all_subjects_design_matrix


# Deprecated functions - kept for backward compatibility but issue warnings
def get_collection_image_metadata(collection=None, data_dir=None, limit=10):
    """Get image metadata associated with collection.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Please use fetch_neurovault_collection instead.
    """
    warnings.warn(
        "get_collection_image_metadata is deprecated and will be removed in a future version. "
        "Please use fetch_neurovault_collection instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    metadata, _ = fetch_neurovault_collection(collection, data_dir)
    return metadata


def download_collection(
    collection=None, data_dir=None, overwrite=False, resume=True, verbose=1
):
    """Download images and metadata from Neurovault collection.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Please use fetch_neurovault_collection instead.
    """
    warnings.warn(
        "download_collection is deprecated and will be removed in a future version. "
        "Please use fetch_neurovault_collection instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return fetch_neurovault_collection(collection, data_dir, verbose)
