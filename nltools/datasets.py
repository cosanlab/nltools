"""
NeuroLearn datasets
===================

functions to help download datasets

"""

## Notes:
# Need to figure out how to speed up loading and resampling of data

__all__ = [
    "download_nifti",
    "get_collection_image_metadata",
    "download_collection",
    "fetch_emotion_ratings",
    "fetch_pain",
]
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import pandas as pd
from nltools.data import Brain_Data
from nilearn.datasets.utils import _get_dataset_dir, _fetch_file
from pynv import Client

# Optional dependencies
try:
    import requests
except ImportError:
    pass


def download_nifti(url, data_dir=None):
    """ Download a image to a nifti file."""
    local_filename = url.split("/")[-1]
    if data_dir is not None:
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
    local_filename = os.path.join(data_dir, local_filename)
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename


def get_collection_image_metadata(collection=None, data_dir=None, limit=10):
    """
    Get image metadata associated with collection

    Args:
        collection (int, optional): collection id. Defaults to None.
        data_dir (str, optional): data directory. Defaults to None.
        limit (int, optional): number of images to increment. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with full image metadata from collection
    """

    if os.path.isfile(os.path.join(data_dir, "metadata.csv")):
        dat = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    else:
        offset = 0
        api = Client()
        i = api.get_collection_images(
            collection_id=collection, limit=limit, offset=offset
        )
        dat = pd.DataFrame(columns=i["results"][0].keys())
        while int(offset) < int(i["count"]):
            for x in i["results"]:
                dat = dat.append(x, ignore_index=True)
            offset = offset + limit
            i = api.get_collection_images(
                collection_id=collection, limit=limit, offset=offset
            )
        dat.to_csv(os.path.join(data_dir, "metadata.csv"), index=False)
    return dat


def download_collection(
    collection=None, data_dir=None, overwrite=False, resume=True, verbose=1
):
    """
    Download images and metadata from Neurovault collection

    Args:
        collection (int, optional): collection id. Defaults to None.
        data_dir (str, optional): data directory. Defaults to None.
        overwrite (bool, optional): overwrite data directory. Defaults to False.
        resume (bool, optional): resume download. Defaults to True.
        verbose (int, optional): print diagnostic messages. Defaults to 1.

    Returns:
        (pd.DataFrame, list): (DataFrame of image metadata, list of files from downloaded collection)
    """

    if data_dir is None:
        data_dir = _get_dataset_dir(str(collection), data_dir=data_dir, verbose=verbose)

    # Get collection Metadata
    metadata = get_collection_image_metadata(collection=collection, data_dir=data_dir)

    # Get images
    files = []
    for f in metadata["file"]:
        files.append(
            _fetch_file(
                f, data_dir, resume=resume, verbose=verbose, overwrite=overwrite
            )
        )

    return (metadata, files)


def fetch_pain(data_dir=None, resume=True, verbose=1):
    """Download and loads pain dataset from neurovault

    Args:
        data_dir: (string, optional) Path of the data directory. Used to force data storage in a specified location. Default: None

    Returns:
        out: (Brain_Data) Brain_Data object with downloaded data. X=metadata

    """

    collection = 504
    dataset_name = "chang2015_pain"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    metadata, files = download_collection(
        collection=collection, data_dir=data_dir, resume=resume, verbose=verbose
    )
    return Brain_Data(data=files, X=metadata)


def fetch_emotion_ratings(data_dir=None, resume=True, verbose=1):
    """Download and loads emotion rating dataset from neurovault

    Args:
        data_dir: (string, optional). Path of the data directory. Used to force data storage in a specified location. Default: None

    Returns:
        out: (Brain_Data) Brain_Data object with downloaded data. X=metadata

    """

    collection = 1964
    dataset_name = "chang2015_emotion_ratings"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    metadata, files = download_collection(
        collection=collection, data_dir=data_dir, resume=resume, verbose=verbose
    )
    return Brain_Data(data=files, X=metadata)
