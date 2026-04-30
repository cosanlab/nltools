# nilearn.utils — Utilities

Discovery helpers for the public nilearn API. Useful for tooling, tests, and documentation generation; rarely needed in user analysis code.

**Source:** https://nilearn.github.io/dev/modules/utils.html

## Inventory

### Functions
| Function | Purpose |
|---|---|
| `all_displays(type_filter=None)` | Get a list of all `displays` objects (slicers, projectors) from nilearn. |
| `all_estimators(type_filter=None)` | Get a list of all estimators (classes with `fit`) from nilearn. |
| `all_functions()` | Get a list of all public functions from nilearn. |

## Usage

```python
from nilearn.utils import all_displays, all_estimators, all_functions

# Every public function as (name, callable) pairs
for name, fn in all_functions():
    print(name)

# Every estimator class; type_filter restricts by mixin/base
for name, cls in all_estimators(type_filter='masker'):
    print(name, cls)

# Every display class (matplotlib slicers / glass-brain projectors)
for name, cls in all_displays():
    print(name, cls)
```

`type_filter` for `all_estimators` accepts strings like `'masker'`, `'decoder'`, `'classifier'`, `'regressor'`, `'transformer'`, `'cluster'` (or a list thereof) to narrow the result.

## Common patterns

Sanity-check imports across the public API in tests:

```python
from nilearn.utils import all_functions, all_estimators

def test_all_public_api_importable():
    for name, fn in all_functions():
        assert callable(fn), name
    for name, cls in all_estimators():
        assert hasattr(cls, 'fit'), name
```

Enumerate maskers programmatically:

```python
maskers = dict(all_estimators(type_filter='masker'))
print(sorted(maskers))
```

## Gotchas

- These are introspection helpers — they walk nilearn's import tree and may be slow on first call.
- `all_estimators` excludes private classes (those starting with `_`) but may include experimental ones; check `__module__` if you need stability guarantees.
- Output is sorted by name, but the exact contents depend on the installed nilearn version.

## See also

- `sklearn.utils.discovery.all_estimators` — sklearn equivalent for sklearn estimators.
- https://nilearn.github.io/dev/modules/utils.html
