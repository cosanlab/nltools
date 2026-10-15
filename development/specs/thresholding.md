# Unified statistical correction and thresholding

Status: proposed design for review; no new nltools API is implemented by this document.
Date: 2026-10-15.

## Intent and scope

Provide a consistent, teachable way to obtain uncorrected, Bonferroni,
Holm–Bonferroni, Benjamini–Hochberg FDR, maximum-statistic FWE, cluster-extent
FWE, cluster-mass FWE, and TFCE FWE results. Users should be able to inspect
adjusted probabilities and decisions, preserve their effect/statistic maps,
and apply the decision mask to the map they want to display.

Use an array foundation with BrainData convenience methods. This is the proposed
default scope, rather than a claim that arbitrary graphs, all ISC nulls, or every
repeated-measures design are supported in the first release. Preserve existing
public calls and return types. General inference belongs in nltools; synthetic
course simulations and their demonstrations may live in dartbrains-tools.

## Existing contracts

- `algorithms/corrections.py` exports `fdr`, `holm_bonf`, `threshold`, and
  `multi_threshold` through `nltools.algorithms`. There is no separate new
  `nltools.stats` namespace to introduce.
- `BrainData.threshold` cuts image values and optionally filters small clusters.
  `cluster_report` describes clusters. Neither establishes permutation FWE.
- `BrainData.ttest` and `Adjacency.ttest` return dictionaries with `mean`, `t`,
  `z`, and `p`. Their approved [contract](ttest.md) keeps correction separate
  and explicitly does not restore `threshold_dict` or thresholded return maps.
- Existing one-sample permutation nulls contain centered means, while the `t`
  field contains the observed parametric t statistic. They are not interchangeable.
- `ContrastResult.p_value` is directional; `ttest` defaults to two tails. A
  correction of supplied p-values must not reinterpret their tails.
- Result records are data, with independent ownership of their payloads. Facades
  delegate computation to pure internal functions.

## Decision

Separate three operations:

1. Test observations or a fitted hypothesis to obtain statistics and probabilities.
2. Correct across a declared family of hypotheses.
3. Apply the resulting decision mask to an effect or statistic for display/export.

Add a functional `nltools.algorithms.correct` entry point for existing evidence.
Add a separate `nltools.algorithms.permutation_test` entry point and a thin
`BrainData.permutation_test` facade for the initial one-sample permutation
workflow. These names are proposed public additions requiring the usual API
manifest and migration-guide changes during implementation.

Do not add a stateful Threshold estimator or change the return type of `ttest`.
Do not overload `BrainData.threshold` with data-dependent inference. A future
convenience spelling on `ttest` would be a separate API decision, not part of
this design.

## Proposed usage

Examples describe the proposed interface, not executable current APIs.

```python
from nltools.algorithms import correct

stats = brain.ttest()
correction = correct(p=stats['p'], method='fdr', alpha=0.05)
(stats['t'] * correction.mask).iplot()

contrast = brain.compute_contrasts('condition_a - condition_b', inference=True)
correction = correct(p=contrast.p_value, method='holm', alpha=0.05)

perm = brain.ttest(permutation=True, return_null=True, random_state=0)
correction = correct(
    statistic=perm['mean'], null=perm['null_dist'],
    method='max_stat', tail=2, alpha=0.05,
)

result = brain.permutation_test(
    statistic='t', correction='tfce', alpha=0.05,
    n_permute=5000, random_state=0,
)
(result.statistic * result.correction.mask).iplot()
```

The explicit `result.correction` record is shared with `correct` results. It
avoids a second set of mask/probability fields or new methods on result records.
Cluster calls use `correction='cluster_extent'` or `'cluster_mass'` and
`cluster_forming_p=0.001`. A forming p-value is required for those methods;
TFCE rejects a supplied cluster-forming threshold.

## Existing-evidence correction

`correct` takes keyword-only inputs. `method` is required; `alpha` defaults to
0.05 and must be finite and strictly between zero and one.

| Method | Inputs | Interpretation |
| --- | --- | --- |
| `uncorrected` | `p` | Pointwise decisions, no family-wise claim |
| `bonferroni` | `p` | Family-wise adjusted p-values |
| `holm` | `p` | Step-down family-wise adjusted p-values |
| `fdr` | `p` | BH-adjusted p-values under its dependence assumptions |
| `max_stat` | `statistic`, `null`, `tail` | Single-step FWE from joint resampling |

The p-value methods reject null/statistic inputs. `max_stat` rejects a supplied
p-map: its evidence is the observed statistic and matching permutation draws.
`tail` applies only to that statistic/null route; supplied p-values retain the
hypothesis and tail of their producer. Keep the established `2/'two'` and
`1/'one'` vocabulary. The latter means the positive direction; negate the
observations/statistics and null together to test the other direction.

For arrays, p-value correction uses `axis=-1` as one family; every slice on the
remaining axes is corrected separately. `axis=None` explicitly makes the whole
array one family. For a single BrainData map, the family is its stored in-mask
voxels. For a single Adjacency result, the family is its stored entries, avoiding
counting a symmetric edge twice. The first release rejects container stacks at
this entry point; users select a map/matrix explicitly. The low-level max-stat
route initially accepts one feature vector and a `(n_permute, n_features)` null.
No automatic family is inferred across contrasts or disconnected input objects.

Inputs inside the declared family must be finite; p-values must lie in [0, 1].
The initial API raises for undefined values instead of silently changing the
family size. Users explicitly restrict the family before calling it. This strict
policy applies only to the new APIs and does not change existing t-test NaN rules.
Validate input shapes, matching voxel geometry, and feature order wherever the
container represents them. Bare arrays cannot prove statistic identity or joint
randomization: these remain explicit caller requirements, documented with the
mean-versus-t example.

For Monte Carlo nulls, count ties as exceedances and use
`(1 + count(null_max >= observed_score)) / (B + 1)`. Two-sided scores use absolute
values; one-sided scores preserve direction. The correction must never use a
percentile cutoff as a substitute for empirical corrected p-values. Exact
enumeration has a different denominator contract and is deferred.

Cluster and TFCE methods are not accepted by the existing-evidence API initially:
a p-map or a vector of map maxima lacks the spatial information they require.
They are unified through the correction result contract, not through ambiguous
argument combinations on `correct`.

## Permutation inference

The initial `permutation_test` implements a one-sample sign-flip test across
independent participant observations. Paired comparisons use participant-level
difference maps. `statistic='t'` is the initial supported statistic; the existing
mean-based `ttest(permutation=True)` path remains unchanged.

The array entry point supports max-statistic FWE without spatial geometry. The
BrainData facade additionally supports volumetric cluster extent, cluster mass,
and TFCE by passing detached mask/affine information to the internal engine.
Do not reconstruct spatial neighbours from the order of a flattened voxel vector.
The array API rejects spatial methods until explicit topology ingress is added.
Do not treat an Adjacency matrix's node connections as connectivity among its
stored hypothesis edges.

Use `correction='max_stat'` for the t-statistic maximum (described as max-t in
reports), plus `cluster_extent`, `cluster_mass`, and `tfce`. One requested
correction per call is the initial interface. Internal reuse of permutations for
several summaries is allowed, but a public multi-method container is deferred.

Other proposed arguments are `popmean=0`, `tail=2`, `alpha=0.05`,
`n_permute=5000`, `cluster_forming_p=None`, `return_null=False`, and the existing
execution controls in canonical order: `n_jobs=-1`, `random_state=None`,
`progress_bar=False`. A new memory-budget argument, if exposed, uses the existing
`memory_budget_gb` vocabulary and shared backend budget owner.

Cluster topology is fixed and reported as six-connected volumetric neighbours
in the first release. Positive and negative clusters are formed separately.
For two-sided inference, one permutation maximum spans both signs and the entire
specified search mask. Cluster extent is voxel count; cluster mass follows the
selected validated backend's precisely documented excess-above-threshold
statistic. The backend probe must verify this definition against its public
outputs before the implementation contract is finalized. Never join opposite
signs through an absolute-valued mask as the toy course simulation does.

TFCE integrates height and connected extent before permutation calibration.
The default backend settings (including exponents and threshold integration)
are fixed for a run and recorded. TFCE scores alone are not corrected p-values.
Cluster inference produces a cluster-level statement, even when its probabilities
are painted back onto all voxels in the cluster; TFCE/max-stat results retain
feature-level interpretation.

The initial supported sample is finite data with at least two observations and
nondegenerate test statistics. Fail before dispatch when the observed model is
undefined; reject nonfinite replicate statistics rather than dropping draws or
features. Repeated-measures exchangeability blocks, arbitrary nuisance-adjusted
GLMs, first-level autocorrelated residual permutations, surface inference,
generic graphs, bootstrap-derived nulls, and GPU execution are later extensions.
Neither a bootstrap distribution nor an independently shuffled feature-wise
array is automatically a valid joint permutation null.

## Result records

Introduce independently owned, frozen-binding records through the established
result namespace. Users read them rather than constructing or fitting them.

`CorrectionResult`:

- `p_adjusted`: probabilities in [0, 1], in the input payload's feature geometry.
  For `uncorrected`, an owned copy of the supplied p-values. For FDR, label these
  BH-adjusted p-values, not an estimate of the false-discovery probability of an
  individual voxel.
- `mask`: rejection mask using `p_adjusted <= alpha`.
- `method`, `alpha`, `inference_level`, and `family_size`: explicit decision
  metadata. Family size follows the chosen axis; it is scalar for one map.
- `null_max`: optional one-dimensional null maxima, with one entry per draw.
  Retaining it never changes probabilities or decisions.
- `clusters`: optional table with stable signed cluster identifiers, extent,
  mass where relevant, and corrected cluster p-values; `cluster_labels` records
  their spatial membership. Voxels outside every formed cluster have adjusted
  p=1 and are never significant.

`PermutationResult`:

- `statistic`: observed t map/vector, unthresholded.
- `correction`: the `CorrectionResult` above.
- `metadata`: statistic identity, tail, centering, resampling scheme, number of
  draws, seed, search geometry, topology/forming settings, TFCE settings, and
  backend/version provenance. No raw participant data are retained by default.

Do not add result methods to recompute or mask. Existing BrainData arithmetic and
plot/export methods consume the payloads. Selection masks and probability maps
must not inherit observation metadata or stale fitted models. Preserve source
geometry and take independent ownership of each result payload.

## Backend and performance requirements

Use pure internal numerical functions under `algorithms/corrections.py` and
`algorithms/inference/`; container adapters live at the facade boundary. The
numerical core never imports BrainData or Adjacency. Export only the approved
public entry points, and update the vocabulary manifest and API designation when
implementing them.

Nilearn's public `mass_univariate.permuted_ols` is the preferred candidate for the
initial spatial t-inference backend. It already supplies maximum-t, cluster
extent/mass, TFCE scores, and null maxima. Do not call private Nilearn functions
as an undocumented permanent dependency or copy a general inference engine into
dartbrains-tools.

A required implementation probe must establish two things before adopting it:

1. Match nltools' worker-count-independent randomization contract. The inspected
   Nilearn implementation seeds worker-sized chunks, so passing `n_jobs` through
   directly does not meet that contract. Evaluate fixed logical permutation
   blocks with deterministic per-block seeds, independent of worker scheduling,
   running the public backend sequentially inside each block and pooling null
   maxima before calculating probabilities. Block boundaries must be independent
   of worker count and available memory. Verify observed maps/settings agree
   across blocks and measure the extra observed-TFCE computation cost.
2. Verify sign separation, tail combination, cluster-mass definition, TFCE
   settings, and p-value/tie conventions against the public outputs. Normalize
   Nilearn's `-log10(p)` representation to probabilities. Do not silently label
   cluster probabilities voxelwise or import its parameter-unit ambiguities.

If the probe cannot meet these requirements with acceptable performance, record
that result and revise the backend design before implementation proceeds. Do not
weaken nltools' reproducibility contract implicitly to fit a wrapper.

Retain observed statistics, bounded worker buffers, exceedance counts, and at
most one maximum per permutation per requested summary. Never allocate the full
`B x V` null merely to discard it when `return_null=False`. For B=5000 and
V=240000, one float64 full null is 9.6 GB, while one vector of B maxima is 40 KB;
these figures exclude participant data, worker buffers, and spatial scratch space.
Use bounded dispatch and the shared `algorithms.backends` memory planning.

The new engine's `return_null=True` returns maxima through `null_max`, not every
permuted map; this is a new API and does not redefine legacy `null_dist`.
Changing alpha on retained adjusted probabilities does not need a new simulation.
Changing the forming threshold or TFCE settings generally does.

Numba is an optional optimization decision after numerical correctness and
memory behavior are established. Compare cold compilation, warm kernels, and
complete inference. No fastmath, precision reduction, permutation-budget
reduction, or GPU claim may be introduced as an undocumented speed optimization.

## Compatibility and rollout

1. Add array p-value corrections and existing-null max-statistic correction,
   the new result record, and adapters for single BrainData/Adjacency maps.
2. Validate the spatial backend probe and implement the new bounded-memory
   one-sample t permutation engine plus BrainData convenience method.
3. Update course examples to use the nltools API after its supporting release;
   retain course-specific simulations in dartbrains-tools.

Keep `fdr`, `holm_bonf`, both existing threshold functions, and both t-test
contracts unchanged. Do not fix old strict-versus-inclusive comparisons by
silently changing their historical behavior. The new correction masks explicitly
preserve boundary discoveries, including p=0, and return all-false masks when
nothing survives instead of a sentinel cutoff.

The documentation must distinguish family definition, test tail, correction
level, and display threshold. Changes to tested families selected after seeing
results are not automatically valid small-volume corrections.

## Acceptance criteria

- Hand-computable BH/Holm/Bonferroni cases cover ties, p=0, no discoveries, and
  original ordering; array axis and family-size behavior are explicit.
- A hand-computable joint null verifies plus-one probabilities, ties, both tails,
  and max-mean versus max-t input separation.
- Shape, finite-value, probability-range, unsupported-input, and geometry errors
  are clear; inputs and sibling result payloads are not mutated or aliased.
- Cluster examples cover empty maps, opposite signs, disconnected regions, image
  boundaries, six-connected topology, tied sizes, and correct cluster-level labels.
- Small spatial fixtures verify nltools' adapters and probability normalization
  against the selected public backend; do not re-test the backend's entire math.
- Seeded results agree across supported worker counts, retention settings, and
  memory-driven worker limits. Test worker failure propagation, not silent
  replacement or dropping of replicates.
- Peak-memory tests verify bounded dispatch and absence of an implicit full null;
  retained maxima have the promised shape and do not alter results.
- Calibrated synthetic null experiments are a separate statistical check with
  binomial uncertainty; do not turn a noisy empirical FWER estimate into a brittle
  exact-value unit test. Power is evaluated separately from type-I error.
- Existing t-test, threshold, metadata, and public API tests remain unchanged and
  pass. Run `uv run poe ok`; regenerate API documentation and run affected
  tutorial/docs checks before release.

## References

- [nltools architecture](../index.md), [inference internals](../inference-internals.md),
  and [one-sample t-test contract](ttest.md).
- [Nilearn permuted_ols](https://nilearn.github.io/stable/modules/generated/nilearn.mass_univariate.permuted_ols.html).
- [Statsmodels multipletests](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
- [MNE permutation_cluster_1samp_test](https://mne.tools/stable/generated/mne.stats.permutation_cluster_1samp_test.html)
  as a reference for later explicit-topology support, not an initial dependency.
