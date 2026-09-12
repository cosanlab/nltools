# Adjacency specification

Approved through Kata `4v55`, following comparison with source at `9ead0ce4`
and released `v0.5.1`. Implementation and verification belong to `dp0q`.

## Approved public changes

- Reject malformed input and asymmetric matrices declared distance/similarity.
  Require an explicit flat type for rectangular arrays interpreted as stacks.
- Preserve stack rank for singleton slice/list/mask/tuple selections.
- Validate metadata shape and compatibility during construction and operations.
- Return valid coefficient maps and native arrays/scalars from regression,
  replacing malformed Adjacency results.
- Remove Adjacency spatial state and project values through explicit atlas/mask
  mapping outside the class.

These change accepted inputs or public results. The zero-diagonal policy
preserves existing symmetric storage and reconstruction behavior.

## Purpose and representation

`Adjacency` represents one relation matrix or a stack of relation matrices over
positional nodes. Keep NumPy storage, Polars `Y` and the existing statistical
and graph workflows. Do not add dataframe forwarding.

A distance or similarity matrix stores its strict upper triangle: `e = n(n-1)/2`
values. A directed matrix stores all `n*n` values in row-major order, including
its diagonal. Matrix type describes interpretation and storage; it does not
impose correlation bounds or nonnegative distances on statistical result maps.

Symmetric storage retains only edges. Both distance and similarity reconstruction
have a zero diagonal; input diagonal values are discarded. This also applies to
coefficient and probability maps stored in the same container.

| State | Logical `shape` | `data.shape` | `len` |
| --- | --- | --- | --- |
| Empty constructor | `(0, 0)` | `(0,)` | `0` |
| Single matrix | `(n, n)` | `(e,)` | `1` |
| Stack, including one matrix | `(m, n, n)` | `(m, e)` | `m` |
| Empty selection from a stack | `(0, n, n)` | `(0, e)` | `0` |

For directed matrices, `e = n*n`. The empty-input policy treats a flat
symmetric vector of length zero as a one-node matrix. `None`, an empty list and
a square `(0, 0)` input contain no matrices. Retain enough shape information to
distinguish these states without adding a constructor argument. A typed empty
stack retains its source node count and matrix type. `is_empty` means there are
no matrices; a one-node matrix is not empty merely because it has no edges.

## Construction

Retain square NumPy/pandas/Polars tables, explicitly flattened arrays, supported
CSV/HDF5 paths, and lists of matrices or CSV paths. Copy construction from
another Adjacency accepts `labels` and `Y` overrides; `None` inherits their
existing values. An explicit `matrix_type` must confirm the existing kind.
Use `distance_to_similarity` to convert distance values and their declared kind.
`Adjacency([])` is empty. This release does not add 3-D array
input or lists of HDF5 files; square matrices can be passed as a list.

Normalize input once, producing a valid matrix kind, node count and single/stack
state. Nonempty square input represents one matrix; a list represents a stack,
including a one-element list. Explicit flat input accepts a 1-D single vector or a 2-D
stack, without squeezing singleton axes. Retain the single-column CSV reader
convention at the file boundary, not as a general ndarray shape rule.

Flat symmetric lengths must be triangular and directed lengths perfect squares.
Lists must agree on node count and matrix type. Unsupported rank, ragged data or
inconsistent shapes raise during construction rather than failing later in
`squareform`. No arbitrary rank or rectangular-matrix guessing.

Retain inference for unambiguous inputs: symmetric square matrices whose entire
diagonal is zero infer distance; an all-one diagonal infers similarity;
asymmetric square matrices infer directed. A flat symmetric vector defaults to
distance because it has no diagonal evidence. Other symmetric diagonals require
an explicit type. Explicit distance/similarity inputs must be symmetric, including
matching missing-value positions. Floating-point symmetry allows roundoff with
`rtol=1e-12` and `atol=1e-12`; integer and boolean comparisons are exact. Reject
disagreement beyond that tolerance rather than discarding the lower triangle. Preserve off-diagonal NaNs for existing missing-data workflows.

## Selection and ownership

Indexing selects matrices, not nodes or individual edges. Keep integer, slice,
integer-list, boolean-mask and the existing tuple-of-indices grammar.

The rank policy makes integer indexing return one matrix.
Slice/list/mask/tuple selection returns a stack, even when it selects one matrix.
Empty selection retains `(0, n, n)`. A single matrix behaves as a one-element
sequence, so `a[0]` and iteration work. Selecting matrices never squeezes the edge axis.

Construction, `copy`, Python shallow/deep copy, selection and transformations
return independently owned mutable state, following the BrainData/DesignMatrix
ownership contract. Preserve internal aliases and cycles. `squareform` and
`to_square` return detached arrays, or a list of detached arrays for stacks,
including for directed storage. Raw `.data`, `Y` and labels remain mutable state;
there is no automatic inference after arbitrary raw edits.

## Labels and Y

`labels` describes nodes. Accept either a shared flat list of `n` labels or,
for a stack, an explicitly nested `(m, n)` label grid. Distinguish these by
structure, not by whether `m == n`. Keep shared labels shared in representation;
select rows of a nested grid with matrix selection. Never repeat a flat list
into a meaningless `m*n` list.

`Y` describes matrices: absent, or a Polars frame with exactly `len(adj)` rows.
Validate both constructor and setter, including single matrices. Pandas input
is converted at ingress. Selection subsets rows; an operation that changes the
meaning of the matrix axis clears `Y` unless it has a defined mapping.

## Spatial projection

Adjacency stores relation matrices and their node/matrix metadata. Remove its
`spatial_scale` constructor argument and attribute, the `SpatialScale` container,
`to_brain`, and the `project` option on `similarity`. Do not add compatibility
aliases or replacement spatial state inside Adjacency.

Retain `BrainData.distance` with `spatial_scale="roi"` or `"searchlight"` and
the underlying computations. These methods return ordinary Adjacency stacks.
Project per-matrix values through the existing mapping helpers outside Adjacency,
using an explicit atlas or mask and ROI or voxel ordering. Keep the mapping order
aligned with the returned stack; callers must subset it when selecting matrices.

Update the ROI RSA tutorial and relevant tests to use explicit projection.
Adjacency selection, copying, append and HDF5 need no spatial state policy.
Removing `spatial_scale` also removes the requirement to restore it after HDF5
loading.

## Append, arithmetic and transformations

Append stacks matrices with matching node count and matrix type. Empty inputs
are identities; typed empty inputs still validate their schema. Append `Y` by
rows, unioning columns by name and filling missing metadata rows with nulls.
Preserve shared node labels when equal; otherwise preserve explicit per-matrix
labels. If one input has node labels and another does not, require the caller
to supply consistent labeling rather than inventing labels or losing them.

Keep scalar arithmetic and same-shape Adjacency arithmetic. Binary Adjacency
operations require matching matrix type and node ordering when labels are
present; no implicit alignment or single-to-stack broadcasting. Results retain
the left operand's matrix metadata and independently own it.

Thresholding and value transforms preserve valid node/matrix metadata. A
conversion from distance to similarity changes the declared matrix type and
works for singles and stacks. Preserve the established numerical formulas.

## Reductions and statistical results

Keep existing reduction semantics: a single matrix reduces stored edges to a
scalar; a stack reduced on axis 0 returns one Adjacency; axis 1 returns one
scalar per matrix. Reject other axes consistently. Axis-0 results retain node
labels only when they agree across inputs, and clear `Y`.

`distance` compares matrices and returns a distance Adjacency whose nodes are
the input matrices. It must not inherit original node labels.
Retain `include_diag`, similarity/permutation modes, threshold, Fisher transforms,
cluster summaries, social-relations modeling and their accepted numerical fixes.

Regression result contract:

- With DesignMatrix predictors, observations are matrices and features are edges.
  `beta`, coefficient standard errors (currently called `sigma`), `t` and `p`
  contain one matrix per predictor; residuals retain the observation stack.
  One-predictor coefficient results retain the existing single-matrix result.
  Degrees of freedom are a scalar when common to all edges, not a malformed
  Adjacency. Predictor-axis maps clear observation `Y`.
- With Adjacency predictors, observations are edges and predictors are matrices.
  This release supports a single response matrix; it does not add stacked
  responses. Coefficients, standard errors and test values are arrays or scalars
  on the predictor axis. Degrees of freedom are a common scalar for the model.
  Only edge-shaped residuals are Adjacency.
  Do not put non-triangular coefficient vectors inside a relation-matrix facade.

Keep the regression dictionary keys. `sigma` is coefficient standard error.
Preserve RSS-based residual-scale calculations and tail semantics. Add numerical
reference tests and shape tests together.

Bootstrap aggregate maps use single-matrix shape and valid node metadata.
The broader BootstrapResult/API transition belongs to `5mz1`. The shared
[one-sample t-test contract](ttest.md) defines dictionary results, permutation
nulls and preservation of directed storage; implementation belongs to `hrgf`.

## Persistence and implementation boundaries

HDF5 round-trips matrix kind, exact single/stack shape, node labels and `Y`.
Legacy readers remain. CSV stores values only; readers require explicit flat
type where inference is ambiguous. Square CSV export remains single-only.
Unsupported loss of shape or metadata must be made
explicit rather than presented as a full round trip.

Use a small normalized state and shared result construction in internal modules;
keep the facade thin. No compatibility shims or general container framework.
Public breaks require migration examples, vocabulary-source updates where
applicable, generated docs and focused tests followed by `uv run poe ok`.

## Baseline evidence and implementation gaps

The tests below were reviewed at `9ead0ce4` under
`nltools/tests/data/adjacency/`. The gaps describe the baseline before this contract.

| Contract area | Existing evidence | Gap |
| --- | --- | --- |
| Logical/storage shapes | contract `test_owned_construction_copy_selection_and_directed_export`, core `test_directed_flat_stack_not_collapsed` | Invalid lengths, one-node, empty and singleton axes |
| Selection | core `test_indexing`, spatial `test_dropped_when_getitem_collapses_to_single` | Stable rank, single indexing, two-node edge axis, labels and empty selections; replace obsolete spatial-state tests |
| Ownership | core `test_copy` compares values | Input/output aliasing and mutable metadata graph ownership |
| Labels/Y/append | contract `test_labels_are_structural_and_y_rows_are_validated`, `test_append_compatibility_labels_and_y_union`, core `test_list_of_adjacency_preserves_y_and_labels` | Label grammar, Y row validation, right-empty identity, type checks |
| Reductions/transforms | core mean/std/median/sum/distance tests; stats threshold/Fisher tests | Metadata and singleton/directed coverage |
| Regression | modeling `test_regression` covers coefficients and intercept-only maps | Multiple-predictor axes, reconstructible results, df type, metadata |
| Inference | stats t-test/tail tests; modeling bootstrap reproducibility/tail tests | Shape policy and coordination with separate inference contracts |
| Persistence | IO Y/direct-label round trips | Empty/singleton shapes and detached reconstruction; remove spatial-state persistence obligations |

Read-only probes at `9ead0ce4` confirmed: empty length is 1; an empty list and
single-matrix indexing raise IndexError; a one-element list selection collapses;
a two-node stack slice collapses the edge axis; shared labels are repeated;
flat NumPy input is aliased; invalid flat lengths survive construction; mismatched
Y rows are accepted; distance/similarity append mixes types; HDF5 loading omits
`spatial_scale`; multiple-predictor regression transposes result axes and yields
an invalid `df` object. These observations establish the implementation gaps addressed by this contract.
