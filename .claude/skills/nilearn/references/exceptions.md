# nilearn.exceptions — Exceptions and Warnings

Custom warnings and errors used across nilearn. Useful for explicit handling around mask computation, scrubbing, dimension validation, and surface mesh checks.

**Source:** https://nilearn.github.io/dev/modules/exceptions.html

## Inventory

### Warnings (subclass `UserWarning`)
| Warning | Purpose |
|---|---|
| `MaskWarning` | Custom warning related to masks. |
| `NotImplementedWarning` | Custom warning to warn about not-implemented features. |

### Exceptions
| Exception | Base | Purpose |
|---|---|---|
| `AllVolumesRemovedError` | `Exception` | Raised when scrubbing/sample_mask removed every volume. |
| `DimensionError` | `TypeError`, `ValueError` | Custom error type for dimension checking. |
| `MeshDimensionError` | `Exception` | Raised when meshes have incompatible dimensions. |

## Catching warnings

```python
import warnings
from nilearn.exceptions import MaskWarning

with warnings.catch_warnings():
    warnings.simplefilter('error', MaskWarning)
    masker.fit(img)                   # MaskWarning is now an exception
```

## Catching exceptions

```python
from nilearn.exceptions import (
    AllVolumesRemovedError, DimensionError, MeshDimensionError,
)

try:
    masker.transform(img, sample_mask=very_strict_mask)
except AllVolumesRemovedError:
    # all volumes were censored; relax fd_threshold or scrub
    ...

try:
    flm.fit(wrong_dim_img, events=ev)
except DimensionError as e:
    print(e)        # tells you expected vs received dimension

try:
    plot_surf(mesh, surf_map=mismatched)
except MeshDimensionError:
    # mesh and data have different number of vertices
    ...
```

## Gotchas

- `DimensionError` inherits from both `TypeError` and `ValueError` — catching either will catch it.
- `AllVolumesRemovedError` typically signals an over-aggressive `fd_threshold` in `load_confounds` or a `sample_mask` that filtered everything; loosen the criteria.
- `MaskWarning` is emitted by the masker auto-mask logic when the inferred mask looks suspect (very small or very large).
- `NotImplementedWarning` flags surface/volume features that aren't yet supported in a given code path; usually the code falls back to a default.

## See also

- `nilearn.interfaces.fmriprep.load_confounds` — common source of `AllVolumesRemovedError` via scrubbing.
- `nilearn.image.check_niimg_3d`, `check_niimg_4d` — emit `DimensionError`.
- https://nilearn.github.io/dev/modules/exceptions.html
