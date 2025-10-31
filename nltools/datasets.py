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
    # Deprecated functions - kept for backward compatibility
    "get_collection_image_metadata",
    "download_collection",
]

import pandas as pd
import warnings
from pathlib import Path
from nltools.data import BrainData

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
