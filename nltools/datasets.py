"""Dataset download and example-data utilities.

Functions to help download example datasets. The curated example datasets
(`fetch_pain`, `fetch_emotion_ratings`) are hosted on the ``nltools/niftis``
Hugging Face dataset and resolve through the same `fetch_resource` machinery
as the MNI templates and atlases. Arbitrary Neurovault collections are still
available via `fetch_neurovault_collection`.

"""

__all__ = [
    "download_nifti",
    "fetch_emotion_ratings",
    "fetch_neurovault_collection",
    "fetch_pain",
    "get_resource_path",
    "load_haxby_example",
]

import io
from contextlib import nullcontext, redirect_stdout
from os.path import dirname, join, sep as pathsep
from pathlib import Path

from nltools.data import BrainData
from nltools.data.simulator.haxby import load_haxby_example
from nltools.templates import fetch_resource

# Core dependencies
from nilearn.datasets import fetch_neurovault_ids

# Optional dependencies
try:
    import requests
except ImportError:
    requests = None


def get_resource_path():
    """Get the path to the nltools resource directory.

    Returns:
        str: Absolute path to `nltools/resources/`, with a trailing separator.
    """
    return join(dirname(__file__), "resources") + pathsep


# Curated datasets hosted on the ``nltools/niftis`` HF dataset. Each directory
# holds a ``metadata.csv`` whose ``filename`` column is the image manifest.
_PAIN_DIR = "datasets/pain"
_EMOTION_DIR = "datasets/emotion_ratings"


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
        verbose (int, optional): Verbosity level; `0` is silent, including the
            data-directory line nilearn reports whatever it is asked for.
            Default: 1

    Returns:
        tuple[pl.DataFrame, list[str]]: `(metadata, files)` — the image metadata
            table and the downloaded image paths.

    Raises:
        ValueError: If collection_id is invalid
        RuntimeError: If download fails
    """
    import polars as pl

    if not isinstance(collection_id, int) or collection_id <= 0:
        raise ValueError("collection_id must be a positive integer")

    try:
        # nilearn resolves its data directory with `get_dataset_dir("neurovault",
        # data_dir)` without forwarding `verbose`, so it announces that
        # directory's absolute path however quietly it was asked to work.
        # `verbose=0` has to mean silence: anything that captures a session and
        # publishes it — a notebook, the docs build — would otherwise carry the
        # path of the machine that ran it.
        quiet = redirect_stdout(io.StringIO()) if verbose == 0 else nullcontext()
        with quiet:
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
    locally on first use, so this works with no extra setup.

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
    locally on first use, so this works with no extra setup.

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
        metadata = pl.read_csv(fetch_resource(f"{_EMOTION_DIR}/metadata.csv"))
        files = [
            fetch_resource(f"{_EMOTION_DIR}/{fn}")
            for fn in metadata["filename"].to_list()
        ]
        return BrainData(data=files, X=metadata, verbose=verbose)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch emotion ratings dataset: {e}")
