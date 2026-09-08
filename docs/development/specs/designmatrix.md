# DesignMatrix specification

Approved through Kata `ydcd`. This contract retains Polars and direct dataframe
method access. It supersedes the earlier names-only `select` proposal.
Implementation and verification are tracked in `kjtp`.

## Purpose and input

`DesignMatrix` stores observations as rows and uniquely named regressors as
columns, with sampling frequency, convolved-column names, confound-column names
and multi-run state. Rows are positional. Polars stores the underlying table.

The constructor accepts another `DesignMatrix`, a Polars or pandas DataFrame, a
NumPy array, a dictionary, `None`, or the existing supported file paths. Pandas
input is converted immediately; its index is not a design column. A one-dimensional
array is one column, a two-dimensional array is observations by regressors, and
other array ranks raise. `columns` supplies array column names only. Names must
remain unique after conversion to strings.

`sampling_freq` and `TR` are mutually exclusive and, when supplied, finite and
positive. `n_rows` is a nonnegative integer, excluding booleans. It must agree
with populated input. Copy construction inherits metadata when its corresponding
argument is `None`, preserving the current override convention.

In-memory tables remain tables. Existing text-file events detection, explicit
run length, default Glover convolution and boxcar opt-out remain supported.
HDF5 inputs restore design metadata and permit existing explicit overrides.

## Direct Polars methods and result construction

Users call Polars methods on the design itself, including full expressions:

```python
dm.select("stim")
dm.select(pl.col("stim").mean())
dm.with_columns(pl.col("stim") * 2)
dm.rename({"stim": "task"}).head(10)
```

The forwarding layer invokes the Polars method and supplies operation context
to `copy_with`. That shared constructor owns copying, effective row count and
metadata policy. It must not infer observation or column semantics solely from
matching output shapes or names.

Eager DataFrame results are independently owned `DesignMatrix` objects. Series,
scalars and other Polars return types remain native. Grouping and lazy builder
objects remain native in this release; a later operation on such an object
follows Polars' own return conventions. No generic builder-proxy framework is
required. Existing explicit nltools methods retain their own documented return
types when their names overlap Polars methods.

| Operation | Metadata policy |
| --- | --- |
| Known column selection, head/tail/slice/filter | Preserve applicable annotations, nominal sampling frequency and multi-run state; use the actual selected row count. |
| Rename | Translate both column-annotation families using the rename mapping. |
| Add or replace columns | Preserve known untouched annotations. New columns are untagged; replacing a known column clears its convolution annotation and preserves its confound role. |
| Aggregation or another operation changing row meaning | Clear sampling frequency, multi-run state and invalid column annotations. |
| Unrecognized operation or expression semantics | Conservatively clear metadata whose validity cannot be established. |

Use small explicit policies for known operations. Arbitrary Polars expressions
remain callable even when their metadata must be cleared; do not restrict them
to avoid making that decision. Do not implement an expression interpreter.
Known nltools fill, standardization and resampling transformations preserve
convolution provenance; resampling updates sampling frequency explicitly.

## Ownership, selection and mutation

Construction, `copy()`, Python `copy.copy()`/`copy.deepcopy()`, and operations
returning `DesignMatrix` independently own retained mutable state, including
supported mutable metadata cells. Copies preserve internal aliases and cycles.

`dm['stim']` returns a detached Polars Series; `dm[['stim']]` returns a
`DesignMatrix`. Bracket indexing by integer, tuple or row mask remains unsupported.
Use direct `slice`/`filter` methods for rows and `item` for scalar cells.

Column assignment remains in place. Design transformations return new objects.
The annotation properties return detached lists. Constructor annotations must
refer to existing columns. Dropping columns prunes annotations. Both `columns`
assignment and direct `rename` translate annotations. Assignment and
`with_columns` apply the replacement rules above. Raw `.data` access remains
possible but arbitrary edits there do not trigger metadata inference.

Forwarded in-place Polars mutations must not leave stale metadata on the design.
Handle their mutation semantics explicitly rather than blindly invoking a
mutator on the backing frame and returning `None`.

## Zero-column designs

An `(n, 0)` design retains its observation count through copying, selection,
append, NumPy conversion and HDF5. Selection computes its actual length, including
zero. Scalar assignment broadcasts over recorded observations. Empty column
selection retains the source observations. `is_empty` remains true when there
are no columns. Equality remains data-based but includes effective shape, so
`(3, 0)` differs from `(5, 0)`; other design metadata is not part of equality.

## Append

Retain vertical run separation, matching sampling frequencies, explicit dtype
matching, missing-value filling, raw-Polars-frame confound tagging for horizontal
append, and rejection of duplicate names and newly introduced duplicate values.
Inputs remain unchanged. Direct pandas append input is rejected; constructor
conversion is available when needed.

Horizontal append checks the row count of explicitly sized zero-column inputs.
Vertical append contributes their recorded rows and fills absent columns under
`fill_na`. Each positive-length single-run input contributes one run when
separating runs. Existing multi-run inputs retain their identities under the
currently accepted combinations; do not expand those combinations here.
Zero-length inputs contribute no run, but their schemas still participate in
column union and dtype validation. The completely default, unsized, untimed
empty design is an identity on either side and bypasses timing checks. Explicitly
timed or sized inputs retain timing validation.

Duplicate checks cover numeric and all-null columns and compare their final
values after filling. Other nonnumeric columns retain the existing unchecked
behavior. Numeric equality is exact
across numeric types without lossy Float64 conversion. Signed zeros match and
distinct large integers remain distinct. Missing values match the same kind at
the same position. NaN matches NaN, null matches null, and null differs from NaN.
Filling can therefore create a duplicate that is rejected. Pre-existing base
duplicates do not prevent an unrelated append.

## Pandas boundary and persistence

The constructor converts pandas input to Polars. For DesignMatrix,
nltools-maintained conversion from Polars to pandas occurs only inside nilearn adapters.
Remove the explicit `DesignMatrix.to_pandas()` method. Keep generic
Polars forwarding generic, without a pandas-specific guard or compatibility shim;
methods exposed by Polars itself follow the normal forwarding rules.
GLM/events adapters may exchange pandas frames internally where nilearn requires
them. Other DesignMatrix operations must not convert Polars to pandas.
Heatmap plotting uses arrays and explicit labels instead of converting
through pandas. This contract does not remove other classes' pandas ingress.

NumPy conversion returns a detached array preserving shape and column order,
without design annotations. HDF5 preserves supported values/types and design
metadata, including zero-column height. Text export omits design metadata and
rejects zero-column designs rather than silently losing their observation count.

Preserve the reserved `.nl_` naming helpers and legacy readers. Do not broaden
raw-input namespace rejection here. Existing internal construction, restored
files and fixtures legitimately carry generated columns.

## Verification

Retain existing reserved-name, no-spike, empty-height and HDF5 regressions.
Add behavior tests for input and copy isolation; direct Polars expressions and
result types; rename/replacement metadata; conservative unknown-operation
metadata; zero-column slicing, assignment, append and equality; exact duplicate
comparison; and the pandas boundary including GLM and plotting adapters.

Update vocabulary sources before generated API output. Document approved public
behavior changes in the migration guide. Run focused tests, docs generation and
the executed docs build, then `uv run poe ok`. The full slow/integration suite
and release publication remain separate.
