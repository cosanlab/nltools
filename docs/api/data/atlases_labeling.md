(data-atlases-labeling-labeling)=
## `labeling`

Coordinate-level atlas labeling.

Adapted from [atlasreader](https://github.com/miykael/atlasreader)
(BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.

**Methods:**

Name | Description
---- | -----------
[`label_coords`](#data-atlases-labeling-label-coords) | Look up anatomical labels for a set of MNI mm coordinates.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`CoordsLike` |  | 

### Methods

(data-atlases-labeling-label-coords)=
#### `label_coords`

```python
label_coords(coords: CoordsLike, *, atlas: str | Sequence[str] = 'harvard_oxford', prob_threshold: float = 5.0) -> pl.DataFrame
```

Look up anatomical labels for a set of MNI mm coordinates.

For each coordinate, returns the atlas region(s) it falls in. Works
for both deterministic atlases (single label per coord) and
probabilistic atlases (formatted ``"42.0% Foo; 18.0% Bar"`` strings,
sorted by descending probability).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coords` | <code>[CoordsLike](#nltools.data.atlases.labeling.CoordsLike)</code> | ``(N, 3)`` array-like of MNI mm coordinates ``(x, y, z)``. A single coord like ``(-42, -22, 56)`` is also accepted. | *required*
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

