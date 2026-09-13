"""Atlas registry, lazy loading, and coordinate labeling.

Atlases are hosted at `huggingface.co/datasets/nltools/niftis` under
`atlases/` and fetched on first use via `fetch_resource`, then cached locally.

The labeling logic was adapted from
[atlasreader](https://github.com/miykael/atlasreader) (BSD-3-Clause); please
cite it when using these tools.

References:
    Notter, M. P., Gale, D., Herholz, P., Markello, R., Notter-Bielser, M.-L., &
    Whittingstall, K. (2019). AtlasReader: A Python package to generate
    coordinate tables, region labels, and informative figures from statistical
    MRI images. *Journal of Open Source Software*, 4(34), 1257.
    https://doi.org/10.21105/joss.01257
"""

# Internal package: these imports are re-exports for the rest of nltools, not
# an advertised surface, so there is no `__all__` to mark them as used.
from .labeling import label_coords  # noqa: F401
from .loading import _Atlas, load_atlas  # noqa: F401
from .registry import list_atlases  # noqa: F401
from .reporting import _ClusterReport, _cluster_report_data  # noqa: F401
