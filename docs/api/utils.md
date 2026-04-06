## `nltools.utils`

NeuroLearn Utilities
====================

Cross-cutting utilities used across the nltools package.

**Functions:**

Name | Description
---- | -----------
[`all_same`](#nltools.utils.all_same) | Check if all items in a sequence are equal to the first item.
[`attempt_to_import`](#nltools.utils.attempt_to_import) | Attempt to import an optional dependency, returning None if unavailable.
[`concatenate`](#nltools.utils.concatenate) | Concatenate a list of BrainData() or Adjacency() objects
[`get_resource_path`](#nltools.utils.get_resource_path) | Get path to nltools resource directory.



### Attributes

### Functions#### `nltools.utils.all_same`

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

#### `nltools.utils.attempt_to_import`

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

#### `nltools.utils.concatenate`

```python
concatenate(data)
```

Concatenate a list of BrainData() or Adjacency() objects

#### `nltools.utils.get_resource_path`

```python
get_resource_path()
```

Get path to nltools resource directory.

