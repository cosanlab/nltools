"""Lazy fetcher for files hosted in the `nltools/niftis` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(`~/.cache/huggingface/hub` by default); subsequent calls return the
cached path without touching the network.
"""

import functools

REPO_ID = "nltools/niftis"
REVISION = "main"


@functools.cache
def fetch_resource(relpath: str) -> str:
    """Return a local path to a file from the `nltools/niftis` HF dataset.

    Args:
        relpath (str): Path within the dataset repo, e.g.
            `'default/2mm-MNI152-2009fsl-mask.nii.gz'` or
            `'masks/k88_parcel_names.csv'`. Use `list_resources`
            to enumerate what's available.

    Returns:
        str: Absolute path to the cached file on disk. The returned path drops
            straight into anything that takes a NIfTI path — nilearn plotting
            and masking helpers, `nibabel.load`, and `BrainData(path)`.

    Note:
        Resolution is memoized per `relpath` for the session — repeated
        calls (e.g. every default-mask `BrainData` construction) return the
        cached path with no work. A file already in the HF cache is resolved
        offline, so only a genuine cache miss touches the network.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=relpath,
            repo_type="dataset",
            revision=REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=relpath,
            repo_type="dataset",
            revision=REVISION,
        )


@functools.lru_cache(maxsize=8)
def _list_repo_files_cached(repo_id: str, revision: str) -> tuple[str, ...]:
    """Single HF API hit per (repo, revision) for the session."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=revision
    )
    return tuple(sorted(files))


def list_resources(prefix: str | None = None) -> list[str]:
    """List files available in the `nltools/niftis` HF dataset.

    Companion to `fetch_resource` — surfaces what's downloadable
    without forcing users to remember relpath strings or visit the HF
    web UI.

    Args:
        prefix (str, optional): Path prefix to filter by (e.g. `'masks/'`,
            `'default/'`, `'fmriprep/'`). Matches with `str.startswith`.

    Returns:
        list[str]: Sorted relative paths usable with `fetch_resource`.

    Note:
        Hits the HF API once per session (cached).
    """
    files = _list_repo_files_cached(REPO_ID, REVISION)
    if prefix:
        return [f for f in files if f.startswith(prefix)]
    return list(files)
