"""Global MNI brain-space configuration for nltools.

This module manages the default MNI template used by `BrainData` and
related classes when no explicit mask is provided. Set it once (e.g., at
the top of a notebook) and all subsequent operations pick it up
automatically.

Examples:
    Set the global brain space:

    ```python
    import nltools

    nltools.set_brainspace(template="fmriprep", resolution=2)
    ```

    Inspect the current configuration:

    ```python
    cfg = nltools.get_brainspace()
    print(cfg.mask)
    ```

    Scope a change to a block:

    ```python
    with nltools.with_brainspace(resolution=1):
        brain = BrainData(...)
    ```
"""

# Internal package: these imports are re-exports for the rest of nltools, not
# an advertised surface, so there is no `__all__` to mark them as used. The
# four brain-space functions are advertised at `nltools`, `BrainSpaceConfig` at
# `nltools.data`, and `fetch_resource`/`list_resources` at `nltools.datasets`.
from .config import (  # noqa: F401
    BrainSpaceConfig,
    get_brainspace,
    reset_brainspace,
    set_brainspace,
    with_brainspace,
)
from .fetch import fetch_resource, list_resources  # noqa: F401
from .matching import (  # noqa: F401
    _TemplateMatch,
    _detect_resolution,
    _get_bg_image,
    _is_standard_space,
    _match_resolution,
)
from .paths import _resolve_paths, _resolve_template_name  # noqa: F401
