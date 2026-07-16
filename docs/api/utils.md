(utils-utils)=
## `utils`

Provide cross-cutting utilities for nltools.

Cross-cutting utilities used across the nltools package.

**Methods:**

Name | Description
---- | -----------
[`all_same`](#utils-all-same) | Check if all items in a sequence are equal to the first item.
[`attempt_to_import`](#utils-attempt-to-import) | Attempt to import an optional dependency, returning None if unavailable.
[`coalesced_gc`](#utils-coalesced-gc) | Collapse nilearn's forced per-copy ``gc.collect()`` calls into ONE per operation.
[`concatenate`](#utils-concatenate) | Concatenate a list of BrainData() or Adjacency() objects.
[`get_resource_path`](#utils-get-resource-path) | Get path to nltools resource directory.

### Methods

(utils-all-same)=
#### `all_same`

```python
all_same(items)
```

Check if all items in a sequence are equal to the first item.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` |  | A sequence of items to compare. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` |  | True if all items equal the first item, False otherwise.

**Examples:**

```pycon
>>> all_same([1, 1, 1])
True
>>> all_same([1, 2, 1])
False
```

(utils-attempt-to-import)=
#### `attempt_to_import`

```python
attempt_to_import(dependency, name = None, fromlist = None)
```

Attempt to import an optional dependency, returning None if unavailable.

This function is used to handle optional dependencies gracefully. If the
import fails, the function returns None rather than raising an error,
allowing the calling code to check and handle missing dependencies.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dependency` |  | The module name to import (e.g., 'torch', 'cupy'). | *required*
`name` |  | Optional name to store the dependency under in module_names. Defaults to the dependency name. | <code>None</code>
`fromlist` |  | Optional list of names to import from the module. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | The imported module, or None if the import failed.

**Examples:**

```pycon
>>> torch = attempt_to_import('torch')
>>> if torch is not None:
...     # Use torch
...     pass
```

(utils-coalesced-gc)=
#### `coalesced_gc`

```python
coalesced_gc()
```

Collapse nilearn's forced per-copy ``gc.collect()`` calls into ONE per operation.

nilearn calls ``gc.collect()`` after every masked-array copy
(``_utils/niimg.py:safe_get_data``); a masking-heavy op — a GLM fit that
re-validates the same mask and builds several result maps — fires dozens.
With torch/nilearn/sklearn resident each sweep costs ~0.1s, so the storm
dominates the wall-clock of otherwise-trivial numerical work.

This no-ops the interim collects and runs a single real collect on exit,
so peak memory stays bounded to one operation's worth of cyclic garbage
(the ``gc.collect()`` nilearn calls is a peak-memory optimization, not a
correctness requirement — suppressing it only defers reclamation). Opt out
with ``NLTOOLS_NO_GC_COALESCE=1``.

Because ``@contextmanager`` results double as decorators, this can also be
used as ``@coalesced_gc()`` on an operation-boundary method.

Nesting is safe: each frame restores whatever it saved, so only the
outermost frame restores the real ``gc.collect`` and runs the final sweep;
inner frames' exit-time collect is a no-op.

Caveat: this swaps a process-global builtin. It is safe under the default
loky (process) worker backend — each worker has its own ``gc``. Under a
*threading* backend there is a brief window where a concurrent thread sees
the no-op collect; ``NLTOOLS_NO_GC_COALESCE=1`` is the escape hatch there.

(utils-concatenate)=
#### `concatenate`

```python
concatenate(data)
```

Concatenate a list of BrainData() or Adjacency() objects.

(utils-get-resource-path)=
#### `get_resource_path`

```python
get_resource_path()
```

Get path to nltools resource directory.

