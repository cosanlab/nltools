# One-sample t-test specification

Dictionary results are approved through Kata `fmc0`. This contract covers
`BrainData.ttest` and `Adjacency.ttest`; implementation is tracked by `hrgf`.

## Inputs

Test a stack of at least two observations along axis 0. BrainData observations
are subject-level effect maps. Adjacency observations are matrices with the same
node order and storage kind. A single observation, including a one-row stack,
cannot supply the variance estimate and raises `ValueError`.

Both methods accept keyword-only `popmean=0.0`, `permutation=False`,
`n_permute=5000`, `tail=2`, and `return_null=False`. Add `popmean` to Adjacency;
retain each facade's existing execution controls and defaults. Use the vocabulary
manifest when editing signatures. This work does not redesign device selection,
random-number generation, two-sample testing, or bootstrap inference.

`tail=2`/`"two"` tests a difference from `popmean` in either direction.
`tail=1`/`"one"` tests a mean greater than `popmean`, following the existing
shared tail validator. Do not introduce directional aliases.

## Dictionary results

Always return `mean`, `t`, `z`, and `p`. Each value is one independent BrainData
image or Adjacency matrix, including when there is only one voxel or stored edge.
There is no new result class and no dictionary-to-attribute adapter.

| Key | Meaning |
|---|---|
| `mean` | Sample mean minus `popmean`: the effect relative to the tested null. |
| `t` | Observed one-sample t-statistic, including on the permutation path. |
| `p` | Parametric t-test p-value, or empirical sign-flip p-value when requested. |
| `z` | Normal-score equivalent of `p` using the existing shared tail-aware conversion. |

Parametric inference uses the existing SciPy one-sample test along axis 0.
Permutation inference passes the centered observations (`data - popmean`) to
the shared one-sample permutation engine once for the whole feature matrix.
It retains the engine's mean statistic, common sign flips across features,
p-value correction, and reproducibility rules. The returned `t` is a reference
statistic; the permutation null contains means, not t-statistics.

For two tails, `z = sign(t) * norm.isf(p / 2)`; for one upper tail,
`z = norm.isf(p)`. Use the existing shared helper's clipping policy for the
conversion only, without modifying returned p-values. Preserve the existing
SciPy and permutation-engine outputs for constant or missing data. This change
does not introduce missing-data omission or a new NaN policy; it does not
guarantee that permutation p-values propagate NaNs.

With both `permutation=True` and `return_null=True`, add `null_dist`: an owned
NumPy array shaped `(n_permute, n_features)`. Never squeeze its feature axis.
Features follow BrainData voxel order or Adjacency flat storage order. Values
are centered means in the same units as `mean`. Otherwise omit this key;
`return_null` has no effect on a parametric test because no permutation null
is computed. Do not add a validation branch solely to reject that combination.

## Shapes and ownership

Aggregate maps clear observation metadata: BrainData `X`/`Y` and Adjacency `Y`.
BrainData retains detached geometry using its established result constructor.
Adjacency retains node count and storage kind, including directed storage;
never infer kind from statistical output values. Retain shared node labels.
Retain per-observation label lists only when they agree across observations;
otherwise clear labels, following the existing Adjacency reduction policy.

Result maps and the null array do not alias input values, input metadata, or
other result maps. Adjacency reconstructs symmetric statistical maps with its
usual zero diagonal; directed maps retain every stored entry.

## Thresholding and migration

Return unthresholded maps. Apply a p-value cutoff or multiple-comparison
correction after testing, then apply the resulting mask to the chosen map.
Do not restore v0.5.1 `threshold_dict`, `return_mask`, `thr_t`, or `thr_mask`.

BrainData keeps its four existing keys; document that `mean` is centered when
`popmean` is nonzero. Adjacency gains `mean` and `z`, and its permutation `t`
changes from a mislabeled mean to the actual t-statistic. Both facades now
return the requested permutation null. Record these changes in the migration
guide when implementing them, without compatibility shims.

## Acceptance tests

- Compare parametric mean, t and p with direct NumPy/SciPy references for both
  tails, zero and nonzero `popmean`, and a one-feature input.
- Compare permutation p-values and centered mean nulls with the shared engine
  at a fixed seed; verify null shape and unchanged results when retaining nulls.
- Verify both z conversions and clipping at p-value endpoints. Check constant
  and missing inputs against the existing SciPy/engine behavior.
- Verify independent map/metadata ownership and cleared observation metadata.
- Cover symmetric and directed Adjacency stacks, node order, shared and
  consistent per-observation labels, clearing of inconsistent labels, and
  one-row rejection.
- Verify exact dictionary keys and demonstrate thresholding outside `ttest`.
- Use red-green tests, independent review, regenerated API documentation,
  executed tutorial checks where affected, and `uv run poe ok` for completion.
