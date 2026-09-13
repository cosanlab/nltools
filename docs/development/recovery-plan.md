# 0.6.0 recovery inventory and pilot

Status: review snapshot, based on source at `5b0080be` on `backout-specs`.
Kata issue `0a0q` is the system of record for the recovery plan, decisions,
dependencies and progress. Read it with `kata show 0a0q --agent`; update the
issues rather than maintaining a second live task list here. The release scope
below is approved; implementation has not begun at the time of this snapshot.

The first implementation decision is `31n2`. DesignMatrix, Adjacency and ttest
contract decisions are `ydcd`, `4v55` and `fmc0`; release verification scope is
`9kgh`. Deferred BrainCollection work is parked separately as `sw05` for 0.6.1.
The remainder of this document preserves the evidence behind those issues.

## Release decision

Defer `BrainCollection`, its collection-specific execution and persistence,
and supporting code used only by that feature to 0.6.1. Retain the other
planned 0.6.0 features. Deferral does not authorize removing shared behavior
because it was introduced during collection development.

Keep the current implementation as the recovery base. Static imports support
removing the collection package from this tree: the only production import of
that package outside itself is its `nltools.data` export.
Some collection-only helpers also live in otherwise retained modules and need
targeted removal. Confirm this boundary with import and workflow checks during
implementation.

Preserve the current source and collection specification in Git before removal.
Do not rewrite shared history or mechanically revert commits made during
collection development.
No new branch, commit, push, or source removal is part of this planning change.

## How to read the evidence

- v0.5.1 (`649fd6c0`) is the released compatibility baseline. Its bugs are not
  desired numerical behavior.
- Current code establishes what is implemented; tests establish what the suite
  checks, including obsolete expectations.
- Settled sections of the four existing specifications establish intended
  behavior. Collection-specific requirements now target 0.6.1.
- Missing coverage below means not established by this review, unless a direct
  contradiction or absent implementation is identified. This is a bounded
  inventory, not a complete method-by-method audit.

## Retained 0.6.0 inventory

| Area | Released baseline | Intended/current difference | Evidence and next action |
| --- | --- | --- | --- |
| BrainData ownership | `Brain_Data.copy()` and mutable data facade | Approved independent ownership includes mask, masker, metadata and fitted results. Current internal result helper still shares mask state by default. | [Ownership spec](specs/braindata.md#purpose-and-ownership), `nltools/data/braindata/utils.py::_copy_without_fit_state`; `nltools/tests/data/braindata/test_braindata_ownership.py` holds the ownership and mutation-safety pins. The copy-counting tests that required sharing for scale/arithmetic are gone. |
| BrainData fitting | `regress(mode=...)`; decoding via `predict(algorithm=..., cv_dict=...)` | Retain modern `fit`, Glm/Ridge estimators and fitted BrainData return. `fit(inplace=False)` independence is partly implemented and tested. | `nltools/data/braindata/modeling.py`; `test_braindata_modeling.py` covers non-inplace fitting, GLM/Ridge prediction and refitting. Extend source/result mutation tests rather than replacing this coverage. |
| Model contracts | No equivalent public Glm/Ridge estimator package in the release | Retain ordinary/banded ridge, cross-validation, GLM and their approved contracts. Specs reject a common estimator base; Glm and Ridge are internal estimators with no shared exported base. Collection concurrency and fit-bundle requirements are deferred. | [GLM spec](specs/glm.md), [Ridge spec](specs/ridge.md), `nltools/models/`, `nltools/tests/models/`. Resolve the shared base through retained estimator contracts, not collection removal. Map remaining clauses to tests; check numerical behavior independently of facade wrapping. |
| DesignMatrix | `Design_Matrix` subclasses pandas DataFrame | Retain Polars-backed DesignMatrix, generated-name rules and row-count fixes. No dedicated complete class spec exists. | `nltools/data/designmatrix/`, `nltools/tests/data/designmatrix/`. Specify construction, indexing, append, generated columns and persistence next; fitting depends on these contracts. |
| Adjacency | Existing public matrix facade | Retain the modern implementation and fixes. Full released/current/desired comparison remains to be done. | `nltools/data/adjacency/`, `nltools/tests/data/adjacency/`. Audit matrix type/shape, selection, arithmetic, metadata, inference and round trips after DesignMatrix. |
| Decoding and Predict | Brain_Data prediction methods return older result structures | Retain BrainData whole-brain, ROI and searchlight decoding and its result contract. Current Predict fields still differ from the approved mode/shape rules. | [BrainData prediction contract](specs/braindata.md#prediction-and-decoding), `nltools/data/results.py::Predict`, `nltools/data/braindata/prediction.py`. Keep Predict; separate collection-only result fields/helpers from the broader approved result redesign. |
| Bootstrap and inference | Brain_Data bootstrap/ttest plus statistical functions | Retain bootstrap, permutation tests and GPU execution. Approved streaming BootstrapResult API is ahead of current `stat`/`save_boots`/`percentiles` implementation; BrainData ttest remains explicitly open. | [Bootstrap contract](specs/braindata.md#bootstrap-results), `nltools/data/braindata/bootstrap.py`, `nltools/algorithms/inference/`. Separate result/API work from numerical and memory behavior; resolve ttest before changing its signature. |
| Persistence | NIfTI/HDF5 and historical serialized metadata | Keep BrainData/DesignMatrix/Adjacency persistence for the current layout; pre-0.6 HDF5 files are rejected with an export instruction rather than read. Removing collection bundles does not remove these needs. | `nltools/data/braindata/io.py`, `nltools/io/h5.py`, [BrainData persistence contract](specs/braindata.md#persistence). Test intended lossless formats separately from image export. |
| Other features | Existing plotting, simulation, ROC, alignment and helpers | Retain their planned 0.6.0 functionality; they are not implicitly deferred. | `nltools/data/__init__.py`, `nltools/algorithms/__init__.py`, `nltools/models/__init__.py` plus migration guide. Finish the export inventory, including returned records, before declaring API coverage complete. |

## Collection deferral boundary

| Disposition | Current location | Reason / implementation check |
| --- | --- | --- |
| Defer entire package | `nltools/data/collection/` | Constructor/discovery, member operations, cache lineage, fit/predict bundles, collection inference and cleanup belong to BrainCollection. Retained code has no direct import dependency except the export. |
| Defer result container | `PredictCollection` in `nltools/data/results.py` and `nltools/data/__init__.py` | Aggregates collection decoding. Preserve the adjacent BrainData `Predict` record. |
| Defer collection-only helpers within retained module | `_serialize_model_spec`, `_model_from_spec`, `_cv_mean_score`, `_cv_roi_mean_scores`, `_cv_searchlight_scores` in `nltools/data/braindata/prediction.py` | Production consumers are collection paths. Remove corresponding helper tests selectively; keep decoding, estimator construction and preprocessing used by BrainData. |
| Review collection-only fields | `Predict.permutation_scores`, `Predict.permutation_pvalue` | Current producers belong to collection group prediction. Check callers and the approved Predict contract before removal; do not remove general inference algorithms. |
| Update tests and release documentation | `nltools/tests/data/collection/`, `test_predict_collection.py`, collection tutorials/guides, API generator entries, site navigation, vocabulary enforcement, benchmarks and task definitions | Remove active 0.6.0 references and tests for the deferred feature. Keep shared assertions in mixed test files. Preserve design evidence for 0.6.1; regenerate generated API pages from source. |
| Keep shared implementation | `nltools/algorithms/backends.py`, inference/ridge/alignment, `nltools/models/`, `nltools/io/h5.py`, BrainData neighborhood caching and `_coalesced_gc` | These serve retained features independently of collection orchestration. Collection history alone is not a reason to revert them. |
| Reassess explanation, not dependency by default | `h5py` rationale in `pyproject.toml` | The comment cites collection fit bundles, but retained persistence still uses HDF5. Keep the dependency unless a separate supported-install decision justifies changing it. |

Historical pipeline moves are not pending cleanup: commit `8b77e013` moved
pipeline helpers under collection; `8988bf1a` subsequently removed the legacy
`cv()` pipeline. Neither `nltools/tasks/` nor collection `pipesteps/` exists
at this source revision. Remove stale documentation references where found;
do not recreate or revert those modules.

Keep the functional-core/facade separation and current module organization
where retained callers use them. Deferring BrainCollection does not justify
moving algorithms back into facades or restoring `nltools.stats` wholesale.
NeuroVault collections are unrelated and remain supported.

## Historical fix inventory

These are representative provenance records, not a cherry-pick list. Inspect
each patch and its regression before adapting it to another implementation.

| Commit | Value to preserve | Handling |
| --- | --- | --- |
| `eb942d07` | NIfTI serialization correction with a regression test | Preserve the data-loss regression and corresponding behavior. |
| `16c2b795` | OLS residual standard errors without an intercept; separate `all_same` correction | Track as distinct fixes even though they share a commit. |
| `ac6d1cd9` | Several p-value, confidence-interval and null-distribution corrections | Review each numerical obligation separately. |
| `54a65658` | Statistical work bundled with a new timeseries inference module | A `fix` label does not make the whole patch a portable bug fix. |
| `1bb91524` | Stats-to-algorithms consolidation across public API, callers and tests | Retain the current organization; do not replay/revert it to remove collection. |
| `d5853840`, `85693213` | Shared GPU execution, budgeting and explicit-device policy | Retain shared safeguards and regressions; exclude collection-only adaptations. |
| `71c75934` | Simplified fitting/results plus unrelated correctness work | Preserve fitted BrainData semantics; do not reverse the entire mixed patch. |

The alternative restart bases remain v0.5.1 and early packaging commit
`2b01307e`. Neither environment was exercised here. With only BrainCollection
deferred, a restart would also require rebuilding the retained modern features;
the import boundary gives no current reason to incur that work.

## Implementation sequence

### 1. Isolate the deferred feature

Capture the starting revision and baseline gate result. Remove the collection
package/export and collection-only helpers, result container, tests and active
documentation references as one reviewable change. Keep approved collection
design material clearly marked for 0.6.1. Update architecture guidance and the
migration guide to describe the revised release scope.

Acceptance: retained public modules import without the collection package;
BrainData GLM/Ridge fit-predict, decoding/searchlight, LocalAlignment, inference,
BrainData HDF5 and DesignMatrix HDF5 smoke tests pass;
no active generated API entry requires collection. Update `__all__` and the
vocabulary source before regeneration. Run `uv run poe docs-generate`,
`uv run poe docs-build`, and `uv run poe ok`. Do not fix unrelated failures
silently or remove shared tests to obtain a pass.

### 2. Pilot BrainData ownership and fitting

Use a small deterministic brain fixture with row metadata and a fitted model.
Start in `nltools/tests/data/braindata/test_braindata_modeling.py`,
`test_braindata_core.py` and `test_braindata_ownership.py`.

Write or extend failing behavioral tests before implementation:

- `copy()`, `copy.copy()` and `copy.deepcopy()` own mutable data, mask state,
  metadata, fitted estimator and attached results independently. Mutating either
  source or copy leaves the other usable and unchanged.
- `fit(inplace=False)` preserves an already fitted source and returns an
  independently usable GLM/Ridge result. Refit replaces obsolete estimator state.
- Attached coefficient and training-prediction maps, plus predictions for a new
  design, own their mutable state independently of the fitted object, source and
  sibling maps. Prediction rows must not inherit incompatible training metadata.
- Scaling and arithmetic preserve the correct row metadata, clear derived state
  and independently own mask state. Replace mask-identity assertions with
  mutation isolation and output correctness assertions.
- Fit then predict gives the same numerical outputs before and after the copy
  repair. Use simple OLS and fixed-alpha ridge cases with explicit preprocessing
  and independently calculated expected values.

Implement the existing approved graph-copy design and semantic entry points;
do not introduce another fitted wrapper or general copy-policy API. Audit every
caller of the replaced helper so no obsolete sharing path remains. New-grid
resampling semantics and bootstrap/result redesign are subsequent slices, not
extra requirements for this pilot; any affected existing behavior must continue
to pass its tests.

Run focused tests, then `uv run poe ok`. Generate docs if public docstrings or
signatures change. Use the pilot to judge whether ownership is now traceable
through a small number of explicit operations and whether unrelated APIs stayed
stable. Reconsider the base only if actual dependencies force broad redesign.
Base that decision on implementation dependencies and workflow results.

Keep the full fitted-state migration separate: the spec forbids retained `X_`,
but current prediction and bootstrap use training features. Removing it requires
coordinating stored-training prediction with the bootstrap input contract.
Existing tests also assert obsolete `X_` and `ridge_scores` state. Do not remove
those attributes incidentally during the ownership pilot.

### 3. Finish retained contracts in dependency order

Specify DesignMatrix, then complete retained Glm/Ridge and BrainData contracts,
including decoding records, bootstrap and persistence as separate changes.
Resolve the open ttest contract before implementation. Audit Adjacency next,
then remaining exports and user-visible return types. Maintain a clause-to-test
table as each slice is taken up rather than prewriting a large new specification
for every class.

For each approved public change, record the released-to-new migration example,
add the behavioral regression, implement the smallest change, regenerate docs
as needed, and run the project gate. Release validation must also cover selected
integration workflows, executed tutorials and real GPU behavior for retained
GPU features. The fast gate alone does not cover those claims; do not run the
entire slow/integration suite without checking scope first.

## Review boundary

This proposal records source and test inspection, not completed recovery.
The unchanged implementation passed `uv run poe ok` during this review:
lint, formatting, types and API checks passed; the fast suite finished with
2,277 passed and 64 warnings in 76.74 seconds. This establishes a baseline,
not conformance to the target specifications. No full integration suite or
rendered-site build was run for this standalone planning document.

The next implementation decision is approval of the collection-removal change
and ownership pilot above. Any commits still require approval; no history rewrite
or push is proposed.
