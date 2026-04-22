"""Global MNI brain-space configuration for nltools.

This module manages the default MNI template used by ``BrainData`` and
related classes when no explicit mask is provided. Set it once (e.g., at
the top of a notebook) and all subsequent operations pick it up
automatically.

Examples:
    Set the global brain space::

        import nltools
        nltools.set_brainspace(template="fmriprep", resolution=2)

    Inspect the current configuration::

        cfg = nltools.get_brainspace()
        print(cfg.mask)

    Scope a change to a block::

        with nltools.with_brainspace(resolution=1):
            brain = BrainData(...)
"""

from .config import (
    BrainSpaceConfig,
    get_brainspace,
    reset_brainspace,
    set_brainspace,
    with_brainspace,
)
from .matching import TemplateMatch, get_bg_image, match_resolution
from .paths import resolve_paths, resolve_template_name

__all__ = [
    "BrainSpaceConfig",
    "TemplateMatch",
    "get_bg_image",
    "get_brainspace",
    "match_resolution",
    "reset_brainspace",
    "resolve_paths",
    "resolve_template_name",
    "set_brainspace",
    "with_brainspace",
]
