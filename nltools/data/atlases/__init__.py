"""Atlas registry, lazy loading, and coordinate labeling.

Atlases are hosted at ``huggingface.co/datasets/nltools/niftis`` under
``atlases/`` and fetched on first use via
:func:`nltools.templates.fetch_resource`. Cached locally afterwards.

The labeling logic was adapted from
[atlasreader](https://github.com/miykael/atlasreader) (BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.
> https://doi.org/10.21105/joss.01257
"""

from .labeling import label_coords
from .loading import Atlas, load_atlas
from .registry import (
    ATLASES,
    DEFAULT_ATLASES,
    AtlasKind,
    AtlasMetadata,
    list_atlases,
)
from .reporting import ClusterReport, cluster_report_data

__all__ = [
    "ATLASES",
    "DEFAULT_ATLASES",
    "Atlas",
    "AtlasKind",
    "AtlasMetadata",
    "ClusterReport",
    "cluster_report_data",
    "label_coords",
    "list_atlases",
    "load_atlas",
]
