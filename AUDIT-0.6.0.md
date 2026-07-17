# nltools v0.6.0 — Pre-Release Codebase Hygiene Audit

_Generated 2026-07-16 · branch `uv-cleanup` · working doc for next session_

---

## ▶ NEXT SESSION — START HERE

_Updated 2026-07-17 (end of lower-priority-sweep session). HEAD: `88c10d0e`._

**Where we are.** 🎉 **AUDIT-0.6.0 WORKLIST FULLY CLOSED.** All 4 priority buckets (correctness, api-consistency, docstrings, dead-code) + signature-lies (F068/F021/F182) closed in prior sessions. This session closed ALL 5 recommended lower-priority sections in 5 commits — test-gaps (F188/F191), type-safety (F107/F009/F017/F046/F085), refactor/dedup (F006/F061/F154/F183/F192), UX lows (F098/F031/F159/F157), F189 (which surfaced + fixed a dead-on-arrival `create_cov_data` bug) — then the 3 remaining stragglers in a 6th commit: **F190** (stats.trim tests), **F193** (make_cosine_basis/calc_bpm numeric tests), **F045** (perf hoist). Nothing left in the catalog except the 4 dismissed false positives (do not re-raise). Full non-slow suite green (1630 passed).

**Recommended next (in order):**
1. ✅ **F188 + F191 DONE (2026-07-17)** — one `test-gap` commit. **F188**: added `tests/stats/test_regression.py` (8 tests, scipy.stats.linregress reference). **F191**: replaced 3 wall-clock assertions in `test_efficient_copy.py` with structural checks. Gate clean (lint + lint-api 0/0, 90 targeted tests green). See the F188/F191 entries below.
2. ✅ **Type-safety bucket (5) DONE (2026-07-17)** — F107 (models layer annotated) + F009/F017/F046/F085. One commit. Gate clean (lint + lint-api 0/0; models/designmatrix/alignment/timeseries/atlas suites green; docs regenerated, diff confined to the touched modules). See per-finding entries below; key learning: BaseModel abstract methods stay param-unannotated because Glm(Nifti)/Ridge(array) aren't LSP-substitutable.
3. ✅ **Refactor/dedup bucket DONE (2026-07-17)** — F006, F061, F154, F183, F192 (F118 already resolved by pool.py removal). One commit. Single-source-of-truth extractions (`detect_resolution`, `_select_corr_func`), frozen dataclass, drop `object.__setattr__`, collapse import-only tests. Gate clean; docs regenerated (only `templates.md` gained `detect_resolution`).
4. ✅ **UX lows (4) DONE (2026-07-17)** — F098 (Roc both-classes validation), F031 (n_iter accepts np.int64), F159 (h5 ValueError), F157 (collapse_mask raises instead of silent None + docs/typo). One commit. Gate clean; docs regenerated (algorithms/data/mask.md).
5. ✅ **F189 DONE (2026-07-17)** — added fast tests for `sphere`/`gaussian`/`create_cov_data` (single + multi-subject). **Surfaced + fixed a real bug**: `create_cov_data` crashed on its default/documented single-3-D-mask path (1-D `apply_mask` result indexed as 2-D); one-line `np.atleast_2d` fix. Gate clean.

**Open decisions — RESOLVED (2026-07-17 discussion session):**
- ✅ **F118** — REMOVED `pool.py` entirely (dead surface; no `.pool()` producer). Also disposed F111 + F121/F181. See the re-framed F118 entry.
- ✅ **`apply_mask` latent bug** — FIXED + tested. `apply_mask` now coerces raw-Niimg masks with `check_brain_data(mask, mask=bd.mask)` so the mask inherits the target's space (`data/braindata/analysis.py:483`); red-green regression test `test_apply_mask_raw_niimg_inherits_target_space` uses a non-MNI 4mm fixture.
- ✅ **F182** — IMPLEMENTED (2026-07-17 functional-GLM session, commits `103d95bf`/`99140652`/`3f8e809f`/`7fb76b4b`). GLM/Ridge `.predict()` parity closed: `Glm.coef_` + `Glm.predict(X)=X@coef_`, `bd.predict(X_new)` works for GLM. See the F182 entry below and the session note.

**Do NOT re-open (verified already dead/done):** F014, F076 (deleted code), F187, F037, F134 (tests now exist). See the 2026-07-17 signature-lies session note below for evidence.

**Discipline reminder:** re-verify every finding against CURRENT code before acting — this audit predates several sessions and multiple findings self-resolved. Gate: `uv run poe lint` + `uv run poe lint-api` (0 semgrep / 0 check_kwonly) + targeted tests. Regenerate docs (`poe docs-generate`) after any signature/docstring change and isolate your effect with the two-regen technique (gotcha #3711).

---

> **Session update (2026-07-16):** ✅ = fixed + tested this session (49 findings, red-green TDD, committed on `uv-cleanup`). 💬 = logged as a ledger discussion thread (needs Eshin's call). Still open (no strike): the api-consistency sweep (46, semgrep-enforced), docstrings (44), dead-code (23). F111 (repool) and F043 (cluster_id join) were decided by Eshin and fixed (ledger #88/#89). Semgrep enforcement setup landed in `.semgrep/` + `scripts/check_kwonly.py` + `poe lint-api`.
>
> **Session update (2026-07-17):** Fixed the stale-venv pytest shebangs (71 console scripts pointed at another machine's venv path). **ICC strip executed** (ledger #86/#3732): removed `nltools/algorithms/inference/icc.py`, `stats.compute_icc`, `BrainData.icc` (+ its `parallel=`/`n_jobs` plumbing), all icc tests/fixtures, `__all__` exports, migration-guide + API-docs references. This resolves **F012** (correctness: ICC1==ICC3) and **F048/F140/F177/F194** (api-consistency: `icc_type=`/`parallel=` leakage, `n_jobs` no-op) by removal. `lint-api` worklist dropped 65→62; full lint (ruff+format+ty) clean; affected suites green (637 passed).
>
> **Session update (2026-07-17, docstrings):** ✅ **Entire docstring/RST-leakage bucket done (44 findings)** — 5 file-disjoint parallel subagents + serial handling of F018. Converted all numpydoc/RST sections to Google-style, fixed stale examples (`mode=`/`cv_dict=`, `cv(scheme=)`, `create_data(y=/n_reps=)`, `fit(n_permute=)`, `ridge_cv` return), phantom params, wrong summaries; **F165 documented ~30 blank BrainCollection facade methods**; F065 added `img_type`/`img_modality` None-guards (only code change). A coupled cross-check confirmed F130's four `plotting/prediction.py` funcs now return figures (F125 landed) so Returns made accurate not deleted. **Then regenerated `docs/api` (`poe docs-generate`, 59 pages, 0 failures) — the deferred large sweep**; this also swept in last session's kwonly/`progress_bar` changes that never got regenerated. Verified: 0 RST role leaks, 0 NeuroLearn RST titles, `brain_collection.md` +375 lines. (24 residual `Parameters\n----` underlines remain in `algorithms.md` — pre-existing, from nilearn HRFs re-exported via `algorithms/hrf.py`, out of scope.) Full lint + lint-api green.
>
> **Session update (2026-07-17, dead-code):** ✅ **Entire dead-code & deferred-gaps bucket done (23 findings)** — commits `36bf2695` (`refactor(dead-code)!:`, 20 files, +94/−730) + `6d55bdb3` (`feat(ridge):` F022 completion). Per-finding disposition is in the note under the `## Dead code & deferred gaps` section header. Highlights: **F114/F179/F115/F116 were already resolved by the previously-executed Layer-B pipelines strip** (verified — base.py has no CVScheme Protocol/Terminal, steps.py no AlignStep), so the "reconcile before deleting" caveat is discharged; F070 (FittedBrainCollection ~290 LOC), F138 (dead ISC trio ~180 LOC), F094/F095 (simulator), F121/F181, F185 (regress/predict_multi shims removed — Adjacency.regress kept) deleted; **F022 wired complete** — `max_gpu_memory_gb` now drives GPU target-batching in the three CV solvers AND `core.ridge_svd` via generalized `_auto_n_targets_batch`, verified numerically on torch-MPS (forced 0.001 GB budget → multi-batch, matches numpy to 8e-7). Full lint + lint-api green; ~900 targeted tests green. **All four priority buckets (correctness, api-consistency, docstrings, dead-code) now CLOSED. Remaining audit sections: refactor (8), test-gap (8), ux (7), type-safety (5), performance (1) — lower priority, not yet scoped.**


> **Session update (2026-07-17, signature-lies + low-priority scoping):** Scoped the 5 remaining sections (29 findings) by re-verifying each against CURRENT code. **5 are already dead or done — do not re-open:** **F014** (ICC formula triplication — `icc.py` deleted by the ICC strip), **F076** (contrast parser in `collection/pipeline.py` — deleted with FittedBrainCollection in `36bf2695`), **F187** (Roc untested — `tests/data/roc/test_roc.py` now exists, 5 tests), **F037** (Adjacency list-construction Y/labels — covered by `test_list_of_adjacency_preserves_y_and_labels`, credited F032), **F134** (prediction plotting untested — `tests/plotting/test_f123_prediction.py` covers all 4 funcs). **F189 is PARTIAL** (4 fast simulator tests landed; `create_cov_data`/`gaussian`/`sphere` still uncovered). **F118 survives but is RE-FRAMED**: the audit pitted `pool.py` against `collection/pipeline.py`, but the latter is gone — the live divergence is now `pipelines/pool.py:227` (`r"([+-])(\w+)"`, no coefficient support) vs `collection/execution.py:775` `_parse_contrast_string` (handles `2*A-B`, validates against regressor_names). So the real remaining scope is **24, not 29**.
>
> ✅ **"Signature lies" bucket done (F068/F021/F182)** — commits `d2a77e27` (`fix(api)!:`) + `9c96c1da` (docs regen). Grouped because they share one defect: a param the body never reads. **F068 was the only HIGH left and is misfiled under UX — it is a correctness trap**: `isc`/`isc_test` accepted `roi_mask` and silently returned a whole-brain map (wrong maps in published analyses, no warning). `roi_mask` is now implemented; `radius_mm`/`device`/`n_jobs`/`progress_bar` removed. **F021** n_jobs removed from the 3 ridge solvers (acceleration is the GPU backend, not joblib). **F182** X **kept** (BaseModel.predict(X) is abstract with X required — unlike F185, removal would break the contract); docstring made honest + Raises: section; it fails LOUDLY so it cannot corrupt an analysis, and implementing new-design prediction remains a deferred FEATURE (nilearn exposes no betas; a single X is ambiguous for a multi-run fit) — **needs Eshin's call**. Gate: `poe lint` + `lint-api` clean, `poe test` 1632 passed.
>
> **Two learnings worth keeping.** (1) The `test_construction.py` "parallel ops" conformance tests asserted only that `n_jobs`/`progress_bar` **exist**, never that they do anything — that is *exactly* how F068 shipped. Presence-only signature tests actively certify lies; `isc`/`isc_test` now assert the inverse. (2) `BrainData.apply_mask(raw_niimg)` re-homes the mask onto the **default MNI152** mask via `check_brain_data(mask)` with no mask context, so it breaks for any collection in a non-MNI space (it raises loudly on affine mismatch, so it is latent, not silent). Coerce with `check_brain_data(roi, mask=<target>.mask)` first. **✅ FIXED (2026-07-17 discussion session)** — `apply_mask` now passes `mask=bd.mask` into `check_brain_data` (`analysis.py:483`) + red-green regression test on a non-MNI fixture.
>
> **Session update (2026-07-17, functional GLM + F182 impl):** ✅ **F182 CLOSED** and the GLM path made functional ("prefer B"). Four commits on `uv-cleanup`:
> - `103d95bf` `feat(modeling)!:` **preprocessing redesign** — replaced the shared grand-mean `scale`/`scale_value` knob (which the GLM path silently erased via nilearn's per-voxel `signal_scaling=0`) with two orthogonal, model-aware controls: `scale` (bool|'auto' → nilearn `mean_scaling`, opt-in for both models) and `standardize` ('center'|'zscore'|None|'auto' → ridge defaults 'zscore', glm None). `Glm` now sets `signal_scaling=False` so scaling is explicit, not inherited. Loud guardrails on the two redundant combos (`scale`+`zscore`; ridge `fit_intercept` when centered). Shared `resolve_preprocessing_defaults()` keeps `BrainData.fit`/`BrainCollection.fit` in sync; collection GLM bundle schema v1→v2 (`standardize` replaces `scale_value`). **Found+fixed a latent bug**: the old `scale_value` was inert for GLM (nilearn re-scaled on top).
> - `99140652` `feat(models):` **F182 predict-parity + report** — `Glm.coef_` (betas assembled from `run_glm` `labels_`/`results_` theta, no unmask), `Glm.predict(X)=X@coef_` (Ridge-parity 2-D ndarray), `bd.predict(X_new)` now works for GLM (the `prediction.py` NotImplementedError fork is gone), `bd.report(contrasts=)`→nilearn HTMLReport (kept the FirstLevelModel object as fit engine + report source), and `BrainCollection.fit(noise_model='ar1')` raises (closed-form contrast path is OLS-only; AR stays correct in-memory via `BrainData`).
> - `3f8e809f` + `7fb76b4b` `refactor(modeling):` **functional GLM extraction** — both eager (`fit_glm`) and on-demand (`compute_contrasts`) now read betas from `coef_` and t/z/p/se from nilearn's *functional* `compute_contrast(labels_, results_, con)` as masked-space arrays, killing the per-regressor/per-contrast Nifti round-trips. Verified numerically identical (t-vs-beta pinning, 'all'-bundle consistency). Only remaining conversions are by design: the one fit-time `to_nifti()` (produces the object the report needs) and `glm_residual` from nilearn (AR whitened residuals).
> - **Design decisions (Eshin's calls this session):** keep AR for BrainData but raise in BrainCollection so manual per-subject looping still works; user surface stays `BrainData` maps (arrays strictly internal to the estimator layer); defer Nifti to the true edge; the nilearn report is a wanted 0.6.0 feature. Gate each commit: `poe lint` + `lint-api` clean, targeted suites green (braindata 365, collection+models incl. slow 389, F182 8). Docs regenerated.

> **How this was produced.** 19 parallel Claude auditors swept the codebase — 15 by subsystem, 4 cross-cutting (API-consistency, docstrings, dead-code/gaps, tests) — against the CLAUDE.md conventions. Every `correctness` finding was then independently re-checked by an adversarial **codex / gpt-5.6-sol** verifier (read-only, instructed to *refute*). Severity shown as **auditor → codex** where they diverged; the adjudicated call is mine. 4 correctness findings were refuted and are quarantined at the end.

## At a glance

**197 findings** across 19 auditors — 4 critical · 31 high · 79 medium · 83 low.

| Category | Count | | Category | Count |
|---|---|---|---|---|
| correctness | 55 | | api-consistency | 46 |
| docstring | 44 | | dead-code | 23 |
| refactor | 8 | | test-gap | 8 |
| ux | 7 | | type-safety | 5 |
| performance | 1 | |  |  |

Correctness cross-check: **51/55 confirmed** by codex, 4 refuted.

## Executive summary

The functional-core/imperative-shell architecture is fundamentally healthy: the four facades (BrainData, Adjacency, DesignMatrix, BrainCollection) delegate cleanly to pure functions, ruff is clean, and the correctness-critical inference/algorithms core is well-tested and mathematically sound in its linear algebra. However, this audit surfaced a dangerous cluster of statistically-invalid results — bootstrap/permutation p-values and CIs centered on the wrong reference (isc_test, isc_group_permutation_test, procrustes_distance, permutation CV), holm_bonf ignoring alpha, ICC1 using the ICC3 error term — which are the scariest class of bug for a scientific library because they return plausible-but-wrong numbers silently. A second cluster is crashes on real (non-synthetic) inputs that unit tests miss: cluster_report on sub-peak clusters, Adjacency-from-list dropping Y/labels, filter_data, upload_neurovault, expand_mask on non-contiguous atlases, and Simulator's always-raising `~isinstance`. The legacy modules that predate the functional-core refactor — Roc, Simulator, plotting/prediction.py, and pipelines/ — concentrate the critical bugs and have near-zero test coverage, which is precisely why breakage (seaborn 0.13.2, `~isinstance`) went unnoticed. Cross-cutting API drift against the v0.6.0 conventions is pervasive: banned kwarg names leak through public facades, the keyword-only `*` marker is missing systemically, and many advertised parameters (n_jobs, roi_mask, radius_mm, progress_bar, max_gpu_memory_gb) are silently ignored. Because v0.6.0 is an intentional breaking release, the API-convention cleanup is a now-or-never opportunity — shipping the banned names locks them in. Docs are largely Google-style clean, but auto-generation makes the BrainCollection facade's ~30 undocumented public methods and the NumPy/RST-style legacy docstrings render as broken/blank reference pages.

## Release blockers

_Ship-gating: silently-wrong science, crashes on realistic input, or breaking-release API decisions that lock in the wrong vocabulary._

- cluster_report crashes on any cluster with sub-peaks (atlases/reporting.py F042) — the common case in real stat maps, completely missed by synthetic single-peak tests
- Simulator.__init__ `~isinstance(...)` bitwise-negates a bool so passing a valid nibabel mask ALWAYS raises (simulator F086) — a critical, untested regression
- plotting/prediction.py: plot_dist_from_hyperplane and plot_probability crash under the pinned seaborn 0.13.2 (positional args, F123/F124); plot_roc returns None, silently breaking Roc.plot's advertised figure return (F125). Zero test coverage (F134)
- Statistically-invalid inference results: isc_test bootstrap p-value against ISC=0 (F066), isc_group_permutation_test CI centered on 0 (F013), procrustes_distance disparity-vs-similarity mislabel (F136), holm_bonf ignores its alpha argument (F135), permutation CV yields an invalid null through the facade (F112)
- HyperAlignment auto_pad silently truncates voxels to the smallest subject while its docstring promises zero-padding (F001) — silent data loss
- Adjacency constructed from a list of Adjacency objects silently drops Y and labels with no error and no test coverage (F032) — silent metadata loss
- filter_data raises TypeError on the documented filter(detrend=True) usage (double-passes to nilearn.signal.clean) (F047)
- solve_banded_ridge_cv in-place unscaling divides X by sqrt(0) when a Dirichlet weight is exactly 0, poisoning X for all subsequent random-search iterations with NaN/Inf (F020)
- correlation_permutation_test rejects the very tail values its own docstring recommends ('two'/'upper'/'lower'/-1), a guaranteed crash (F011)
- expand_mask returns empty masks for any non-contiguous-label atlas (iterates nonzero indices instead of label values) (F149); KFoldStratified silently ignores its documented shuffle/random_state (F150)
- upload_neurovault UnboundLocalError when create_collection fails (F057); BrainCollectionPipeline.n_subjects crashes on a nonexistent BrainCollection.n_images (F067)
- Roc.accuracy_se computes p*p/n instead of p*(1-p)/n — wrong standard error of a proportion (F088); create_ncov_data checks type(cor) instead of type(cov) (F087)
- Banned kwarg names leaking through public v0.6.0 facades: mode= (regress), icc_type= (compute_icc/BrainData.icc), parallel= (permutation, braindata.icc), scheme= (collection.align, should be spatial_scale), algorithm= (pipeline.predict), n_iter (Ridge) — embarrassing to lock in during the one breaking release meant to fix them
- isc/isc_test silently ignore roi_mask and radius_mm (F068) — users believe an ROI-scoped ISC ran when it did not
- BrainCollection facade: ~30 public methods/properties render as blank API-reference pages because they have no docstrings (F165)

## Cross-cutting themes

### [CRITICAL] Statistically-invalid inference: wrong p-value/CI reference and wrong statistical model
*Affected:* collection/inference.py (isc_test F066), algorithms/inference/isc.py (isc_group_permutation_test CI F013), algorithms/inference/icc.py (ICC1 uses ICC3 error term F012), stats/alignment.py (procrustes_distance disparity-vs-similarity F136), stats/corrections.py (holm_bonf ignores alpha F135), pipelines/cv.py (permutation scheme invalid null F112, bootstrap n_splits F113), data/roc (accuracy_se p*p F088, Gaussian ROC F089)

*Fix:* Treat as the top release gate. For a scientific library, silently-wrong statistics erode all trust. Fix the null/CI centering so bootstraps bracket the estimate and permutation p-values test against the true null; reconcile ICC1 with its cited Shrout & Fleiss one-way model; fix holm_bonf's alpha and accuracy_se's p*(1-p). Add regression tests that assert on numeric values, not just shapes — the existing synthetic tests encoded some of these bugs as intended behavior.

### [CRITICAL] Legacy pre-refactor modules concentrate critical bugs and have near-zero test coverage
*Affected:* data/roc (394 LOC, empty tests/data/roc/ F187; ~isinstance-adjacent bugs), data/simulator (critical ~isinstance F086, type(cor) F087, all tests slow-marked so default run exercises nothing F189), plotting/prediction.py (two functions crash under seaborn 0.13.2 F123/F124, all return None F125, zero coverage F134), pipelines/ (repool F111, permutation F112, stubs F121)

*Fix:* These modules predate functional-core and are the audit's true hot spot: the coverage gap is why the seaborn break and the always-raising Simulator constructor shipped undetected. Before release, either fix+test or explicitly quarantine. Minimum: add smoke tests that instantiate and call each public entry point on a real input, and un-slow-mark enough Simulator/Roc tests that the default suite exercises the constructors.

### [CRITICAL] Crashes/data-loss on real inputs that synthetic tests miss
*Affected:* atlases/reporting.py (cluster_report on sub-peak clusters, the common case F042), data/adjacency (list-of-Adjacency drops Y and labels silently F032), braindata/analysis.py (filter_data double-passes detrend/standardize F047), braindata/io.py (upload_neurovault UnboundLocalError F057), mask.py (expand_mask empty masks on non-contiguous atlases F149), algorithms/alignment (auto_pad truncates voxels vs documented zero-pad F001), collection/pipeline.py (n_subjects calls nonexistent n_images F067), correlation.py (permutation_test rejects its own documented tail values F011)

*Fix:* Each is invisible to the current synthetic single-peak / small-fixture tests but fires on realistic stat maps, subject lists, and NIfTIs. Add fixtures that reproduce the real-world shape (multi-peak clusters, ragged-voxel subjects, list construction with metadata) and fix. auto_pad and the Adjacency-list path are silent data-loss, not loud crashes — highest priority within this theme.

### [HIGH] Banned/non-canonical kwarg names leak through public facades
*Affected:* stats.regress mode= F139, compute_icc/BrainData.icc icc_type= F140/F194, permutation.py & braindata.icc parallel= F142/F177/F048, collection.align scheme= (should be spatial_scale) F073/F164/F174, BrainCollectionPipeline.predict algorithm= F071, models/ridge n_iter F105, roc plot_method=/threshold_type= F097, collection compute_contrasts contrast_type= F077, adjacency cluster_summary/plot_mds metric= misuse F034/F035, compute_similarity method-for-metric F148, BrainData.fit model= F175

*Fix:* v0.6.0 is the one breaking release where these can be renamed for free; shipping them locks the wrong vocabulary in. Sweep every public facade signature against the canonical table (method/metric/spatial_scale/n_jobs/n_permute/n_samples/device), applying the facade-translation rule at the boundary so internal algorithm-layer names can stay. This is a mechanical but high-value pass best done as one coordinated change.

### [HIGH] Advertised-but-ignored parameters mislead users about what ran
*Affected:* collection isc/isc_test ignore roi_mask/radius_mm/device/n_jobs/progress_bar F068, align ignores cache/progress_bar F073, ridge solvers n_jobs F021 and max_gpu_memory_gb F022 never used, SRM/DetSRM.transform ignore n_jobs F002, LocalAlignment forces tqdm with no progress_bar control F004, braindata.icc n_jobs no-op F048

*Fix:* Silently dropping a spatial-scope or parallelism kwarg is worse than not offering it — users believe an ROI/searchlight or parallel run happened when it didn't. Either wire the parameter through or remove it from the signature (and raise on unexpected kwargs). roi_mask/radius_mm on isc is the most scientifically consequential.

### [HIGH] Auto-generated docs break: undocumented facade + NumPy/RST-style legacy docstrings
*Affected:* BrainCollection facade ~30 public methods/properties with no docstring render blank F165, fitresults NumPy-style section underlines F166/F099, simulator/roc/pipeline RST title underlines and numpydoc sections F074/F170/F171/F173, stale/broken doc examples (mode=/cv_dict= F167, create_data y=/n_reps= F092, SimulateGrid.fit(n_permute) F093, Predict method= vs spatial_scale F091, GLMModel phantom class F104, create_cov/ncov_data phantom params F168/F169), lone :func: leak F172

*Fix:* Docs are auto-generated by griffe2md, so these render as the user-facing API reference. BrainCollection is the primary entry point — blank method pages are the single most embarrassing doc defect. Add one-line Google-style summaries to the facade delegators and convert the four legacy modules' NumPy/RST sections to Google-style. The stale examples that reference removed kwargs are copy-paste traps for users.

### [MEDIUM] Missing keyword-only `*` marker on 3+-kwarg public methods (systemic)
*Affected:* Glm.fit (real footgun: positional call binds design matrix to unused y F103), Ridge/Glm __init__ F106, adjacency F036, braindata plot/plot_flatmap/resample_to/icc F053/F060, designmatrix.clean F078, plot_flatmap F129, srm estimators F005, ridge solvers F025

*Fix:* The convention mandates `*` after the primary data arg and in any public method with 3+ kwargs. Glm.fit is the one with a live correctness consequence and should be fixed regardless; the rest are a mechanical sweep worth doing in the same breaking release as the kwarg-rename pass.

### [MEDIUM] Dead / orphaned / stub public surface should be pruned or finished
*Affected:* pipelines dead abstractions (Terminal F115, duplicate CVScheme Protocol shadowing the real class F114/F179, orphaned AlignStep/FittedAlign F116), permanent NotImplementedError stubs (StatResult/PooledData.to_nifti F121/F181, PooledData.repool F111, Glm.predict(X) F182), FittedBrainCollection never instantiated F070, dead ISC helper cluster in stats/intersubject.py F138, _run_permutation never called F094, duplicated batch_or_skip F023, leftover debug prints F095

*Fix:* A breaking release is the moment to cut orphaned surface rather than document it. Prioritize the CVScheme name collision (the public export is the wrong object) and public methods that only raise NotImplementedError — those are traps that look supported. Remove dead helpers and debug prints as trivial cleanup.

### [MEDIUM] **kwargs forwarded across internal nltools->nltools boundaries
*Affected:* designmatrix downsample/upsample F080, adjacency plot_silhouette/plot_mds F041, pipelines AlignStep forwards to internal SRM/HyperAlignment F117, stats.align *args/**kwargs to internal SRM F141, BrainData.distance F186, roc F096

*Fix:* The convention permits **kwargs only when forwarding to a genuine third-party API; internal delegation must use explicit signatures. Replace with explicit params so signatures are self-documenting and the auto-docs are complete. Low individual risk but a consistent architectural rule violation.

## Quick wins (do first)

- holm_bonf: use the passed alpha instead of hardcoded 0.05 (F135) — one-line correctness fix
- Simulator.__init__: replace `~isinstance(...)` with `not isinstance(...)` (F086) — one-character critical fix
- Roc.accuracy_se: p*p -> p*(1-p) (F088); create_ncov_data: type(cor) -> type(cov) (F087) — trivial correctness fixes
- BrainCollectionPipeline.n_subjects: point at the real BrainCollection image-count property instead of nonexistent n_images (F067)
- correlation_permutation_test: replace hand-rolled `if tail not in [1,2]` with the shared validate_tail_parameter (F011) — removes the crash and de-duplicates
- Adjacency.threshold: fix the `if cutoff` truthiness test so 0.0 is treated as provided (F033)
- Remove leftover debug print() calls that dump full arrays during simulation (F095)
- Fix docstrings referencing the nonexistent 'GLMModel' class in 4 places incl. a broken import example (F104)
- Fix nltools/__init__.py __all__ advertising datasets/cross_validation submodules that aren't imported, so attribute access fails (F152)
- Remove the duplicated dead _batch_or_skip (F023) and the double p-value validation in fdr (F147)
- roi_to_brain: initialize the 2-D output background to 0.0 instead of 1.0 (F151)
- Delete the dead ISC helper cluster (~130 lines) in stats/intersubject.py superseded by the inference module (F138)

## Correctness bugs (codex-verified)

_Sorted by severity. Where auditor and codex diverged on severity, both are shown; use your judgment — codex tended to downgrade crash-severity, but a guaranteed crash in a public function is still user-facing._

### Critical severity

#### ✅ ~~F042 · CRITICAL · codex: confirmed (→medium)~~
`nltools/data/atlases/reporting.py:330-335` — **cluster_report crashes on any cluster with sub-peaks (empty-string Cluster Size -> ValueError)**  
nilearn's get_clusters_table emits one row per peak AND sub-peak; sub-peak rows carry an empty string '' in the 'Cluster Size (mm3)' column (verified at runtime: the column dtype is object, values like [80, '', '']). `_build_peaks_dataframe` does `table["Cluster Size (mm3)"].to_numpy(dtype=float)` for both `volume_mm3` and `n_voxels`, which raises `ValueError: could not convert string to float: ''` the moment any cluster has more than one local maximum within `min_distance`. This is the normal case for real fMRI stat maps. The existing tests use uniform synthetic blobs (one peak each), so they never exercise it and the bug ships silently.  
```python
"volume_mm3": table["Cluster Size (mm3)"].to_numpy(dtype=float),
"n_voxels": (
    table["Cluster Size (mm3)"].to_numpy(dtype=float) / voxel_volume_mm3
).round().astype(int),
```
*Fix (small):* Coerce the Cluster Size column through a NaN-tolerant path (e.g. `pd.to_numpy` via `pd.to_numeric(table['Cluster Size (mm3)'], errors='coerce')`) so sub-peak rows become NaN, then decide sub-peak fill semantics (propagate the parent cluster's size, or leave volume/n_voxels null). Also guard `.astype(int)` against NaN (NaN.astype(int) yields platform garbage/0). Add a regression test with a two-local-maxima cluster.
*Codex trace:* BrainData.cluster_report calls cluster_report_data, which unconditionally calls _build_peaks_dataframe for the thresholded image. With the repo-pinned nilearn 0.13.1, I reproduced get_clusters_table returning [11, '', ''] for a connected cluster with multiple local maxima. _build_peaks_dataframe then calls to_numpy(dtype=float) on that column, producing ValueError: could not convert string to floa

#### ✅ ~~F086 · CRITICAL · codex: confirmed (→medium)~~
`nltools/data/simulator/__init__.py:67-68` — **`~isinstance(...)` bitwise-negates a bool, so passing a valid nibabel image always raises**  
The elif uses bitwise inversion on a boolean: `~isinstance(brain_mask, Nifti1Image)`. `~True == -2` (truthy) and `~False == -1` (truthy), so whenever this branch is reached (i.e. brain_mask is a real Nifti1Image, not a str and not None) it ALWAYS raises ValueError('brain_mask is not a string or a nibabel instance'). The documented use case of passing a nibabel image object is completely broken. Python 3.16 will also remove `~` on bool. The test suite only exercises `Simulator()` (None path), so this is untested.  
```python
elif ~isinstance(brain_mask, nib.nifti1.Nifti1Image):
    raise ValueError("brain_mask is not a string or a nibabel instance")
```
*Fix (trivial):* Use `elif not isinstance(brain_mask, nib.nifti1.Nifti1Image):`. Add a test that passes a Nifti1Image.
*Codex trace:* In Simulator.__init__, a Nifti1Image is neither str nor None, so execution reaches line 67. isinstance(...) returns True; ~True evaluates to -2, which is truthy, so the ValueError is always raised. All located Simulator tests construct Simulator() without a brain_mask, leaving this documented input path uncovered. The proposed replacement with `not isinstance(...)` is correct.

#### ✅ ~~F123 · CRITICAL · codex: confirmed (→low)~~
`nltools/plotting/prediction.py:25-40` — **plot_dist_from_hyperplane crashes: positional catplot args unsupported in seaborn 0.13.2**  
sns.catplot is called with x and y as positional args ('subject_id', 'dist_from_hyperplane_xval'). seaborn >=0.12 made catplot signature `catplot(data=None, *, x=None, y=None, ...)` — positional x/y are keyword-only, so passing them positionally together with data= raises `TypeError: catplot() got multiple values for argument 'data'`. The project pins seaborn>=0.13.2, so this public function fails on every call.  
*Evidence:* `sns.catplot("subject_id", "dist_from_hyperplane_xval", hue="Y", data=stats_output, kind="point")`  
*Fix (trivial):* Pass x/y as keywords: sns.catplot(data=stats_output, x="subject_id", y=col, hue="Y", kind="point"). Add a smoke test so this stays green.
*Codex trace:* Both branches of plot_dist_from_hyperplane pass two positional strings to sns.catplot while also passing data=stats_output. The project requires seaborn>=0.13.2 and locks 0.13.2; its actual installed signature is catplot(data=None, *, x=None, y=None, ...). Python argument binding therefore raises TypeError: catplot() got multiple values for argument 'data' before plotting. The function is publicly

#### ✅ ~~F124 · CRITICAL · codex: confirmed (→low)~~
`nltools/plotting/prediction.py:78-81` — **plot_probability crashes: positional lmplot args unsupported in seaborn 0.13.2**  
sns.lmplot is called with x and y positionally ("Y", "Probability_xval"). In seaborn >=0.12 lmplot is `lmplot(data=None, *, x=None, y=None, ...)`, so this raises `TypeError: lmplot() got multiple values for argument 'data'`. Verified against the installed seaborn 0.13.2. The sibling plot_scatter already uses the correct keyword form, so this is an inconsistency as well as a crash.  
*Evidence:* `sns.lmplot("Y", "Probability_xval", data=stats_output, logistic=True)`  
*Fix (trivial):* sns.lmplot(data=stats_output, x="Y", y=col, logistic=True).
*Codex trace:* Both branches of plot_probability call sns.lmplot with two positional arguments plus data=stats_output. The repository requires seaborn>=0.13.2 and locks 0.13.2; its installed lmplot signature is lmplot(data, *, x=None, y=None, ...). Therefore the first positional argument binds data and the explicit data keyword binds it again, raising TypeError: lmplot() got multiple values for argument 'data' b

### High severity

#### ✅ ~~F001 · HIGH · codex: confirmed (→medium)~~
`nltools/algorithms/alignment/hyperalignment.py:267-294` — **auto_pad silently truncates voxels to the smallest subject instead of zero-padding as documented**  
The `auto_pad` docstring (lines 151-154, 212-214) says it will "automatically zero-pad matrices to standardize sizes." The actual Stage 0 code does the opposite for the feature/voxel axis: `R = min(sizes_0)` then `y = x[0:R, :]` truncates every subject down to the minimum number of voxels, silently discarding data from subjects with more voxels. Meanwhile the sample-padding branch (`C = max(sizes_1)`; append zeros) is dead code, because fit already validates at lines 258-265 that all subjects have identical sample counts, so `missing` is always 0. Net effect: a user who passes subjects of shape (300, T) and (250, T) loses 50 voxels of the first subject with no warning, contradicting the documented behavior.  
```python
R = min(sizes_0)
C = max(sizes_1)
...
y = x[0:R, :]  # Truncate to min features
missing = C - y.shape[1]
if missing > 0:  # never true: samples already validated equal
```
*Fix (medium):* Either implement real zero-padding of the voxel axis to `max(sizes_0)` (matching the docstring and `_procrustes_pairwise`'s own padding behavior), or keep truncation but rename the flag and rewrite the docstring to say it truncates to the minimum voxel count; also drop the dead sample-padding branch. Add a regression test with unequal voxel counts.
*Codex trace:* fit() first requires identical sample counts, so C=max(sizes_1) equals every input's column count and the sample-padding branch cannot run. With auto_pad=True it then sets R=min(sizes_0) and slices every subject as x[:R, :], silently discarding all features beyond the smallest subject. Subsequent stages operate only on these truncated matrices. This directly contradicts the public zero-padding doc

#### ✅ ~~F011 · HIGH · codex: confirmed (→medium)~~
`nltools/algorithms/inference/correlation.py:721-722` — **correlation_permutation_test rejects documented tail values ('two','upper','lower',-1)**  
The signature declares `tail: int | str = 2` and the docstring (lines 653-657) advertises 'two'/2, 'upper'/1, 'lower'/-1 -- explicitly recommending 'upper'/'lower' for FDR. But validation is `if tail not in [1, 2]: raise ValueError`, so every string form and tail=-1 raises. This is the ONLY function in the module that hand-rolls this check; one_sample.py:294, two_sample.py:342 and matrix.py:320 all call `validate_tail_parameter(tail)` which normalizes strings and -1. Downstream `_compute_pvalue` already normalizes via `validate_tail_parameter`, so the inline check is pure breakage. A user copying the documented `tail='lower'` idiom for a one-tailed negative-correlation FDR test gets a crash.  
```python
if tail not in [1, 2]:
    raise ValueError(f"tail must be 1 or 2, got {tail}")
```
*Fix (trivial):* Replace the inline check with `tail = validate_tail_parameter(tail)` (as the other four public tests do), so strings and -1 are accepted and normalized consistently.
*Codex trace:* The public nltools.stats facade forwards tail unchanged. correlation_permutation_test then rejects every value except numeric 1 or 2 before backend dispatch, so documented values 'two', 'upper', 'lower', and -1 deterministically raise ValueError. All CPU/GPU paths eventually call _compute_pvalue, which already uses validate_tail_parameter and supports those values. The shared validator is used by 

#### ✅ ~~F020 · HIGH · codex: confirmed (→medium)~~
`nltools/algorithms/ridge/solvers.py:322-323, 490-491` — **In-place X unscaling divides by sqrt(gamma), producing NaN/Inf when a Dirichlet weight is 0**  
The random-search loop scales feature blocks in place (`X[:, slices[kk]] *= xp.sqrt(gamma[kk])`) and later restores them (`X[:, slices[kk]] /= xp.sqrt(gamma[kk])`). np.random.dirichlet with a small concentration (the default includes 0.1) can return components that are exactly 0.0 (underflow). When gamma[kk]==0, the block is zeroed on scale-up and then divided by 0 on restore, permanently writing NaN/Inf into X. Because X is mutated in place and reused, every subsequent gamma iteration then operates on a corrupted X, silently poisoning deltas/coefs/cv_scores for the whole fit. Even without an exact zero, the repeated float32 multiply/divide round-trip lets X drift from its original values across up to n_iter iterations.  
```python
for kk in range(n_spaces):
    X[:, slices[kk]] *= xp.sqrt(gamma[kk])
...
for kk in range(n_spaces):
    X[:, slices[kk]] /= xp.sqrt(gamma[kk])
```
*Fix (small):* Do not mutate/restore X in place. Build the scaled matrix on a copy per iteration (e.g. `X_scaled = X * sqrt_gamma_broadcast`) so the original X is never divided by a possibly-zero weight, or guard/clamp zero gammas. This also removes the cumulative round-trip drift.
*Codex trace:* `Xs` is concatenated into a private working `X`; generated gammas are then cast to `X.dtype`. With float32 inputs, tiny positive Dirichlet components can become exactly zero (reproduced with default sampling, `n_iter=100`, two spaces, `random_state=13`). At lines 323/491 the corresponding block undergoes `X_block *= 0` followed by `X_block /= 0`, producing NaNs. The next gamma iteration reuses thi

#### ✅ ~~F150 · HIGH · codex: confirmed (→medium)~~
`nltools/cross_validation.py:40-46` — **KFoldStratified silently ignores shuffle= and random_state=**  
`_make_test_folds` builds folds purely from `np.argsort(y, kind='stable')` round-robin assignment; it never consults `self.shuffle` or `self.random_state`. Both are accepted in `__init__` and documented, but have zero effect. Verified: shuffle=True with random_state=1 vs 999 vs shuffle=False all yield identical folds. Users passing random_state expecting reproducibly-randomized stratified folds get deterministic output regardless — a silent contract violation.  
```python
def _make_test_folds(self, X, y=None, groups=None):
    y_arr = np.asarray(y).ravel()
    order = np.argsort(y_arr, kind="stable")
    ...  # shuffle / random_state never used
```
*Fix (small):* Either honor the params (seeded within-stratum tie permutation when shuffle=True) or drop shuffle/random_state from the signature and docstring.
*Codex trace:* KFoldStratified.__init__ passes shuffle/random_state to sklearn's _BaseKFold, which only validates and stores them. KFoldStratified.split() delegates through _BaseKFold.split() and BaseCrossValidator.split(), ultimately calling KFoldStratified._iter_test_masks(). That calls _make_test_folds(), whose stable argsort and round-robin assignment never read either attribute. An isolated reproduction pro

#### ✅ ~~F032 · HIGH · codex: confirmed (→medium)~~
`nltools/data/adjacency/__init__.py:180` — **Constructing Adjacency from a list of Adjacency objects silently drops Y and labels**  
In the list-of-Adjacency branch (lines 82-113), the concatenated result's Y is copied onto self via setattr for the item 'Y' (line 85-86), and matrix data/type/symmetry are set. But this branch does NOT return early. Execution falls through to line 180 `self.Y = Y`, where `Y` is the constructor parameter (default None). The setter then replaces the concatenated Y with an empty polars frame. Labels are never copied from the concatenated object at all, and label handling at line 192 sets `self.labels = []` when the `labels` param is None. So `Adjacency([adj1, adj2])` returns an object with the right data but empty Y and empty labels, even though `concatenate` -> `append` correctly preserved them. The h5/legacy branches return early (lines 144, 160) and are unaffected, which is why this only bites the list path.  
```python
for item in ["data", "matrix_type", "Y", "issymmetric"]:
    setattr(self, item, getattr(tmp, item))
...
# Setup Y dataframe — setter validates + converts to polars
self.Y = Y   # <- overwrites the concatenated Y with the (None) param
```
*Fix (small):* Return early after the list-of-Adjacency concatenation branch (mirroring the h5 branches), or capture the concatenated Y/labels and only fall through to `self.Y = Y` when the user explicitly passed a Y. Also carry `tmp.labels` over. Add a regression test asserting Y and labels survive `Adjacency([adj1, adj2])`.
*Codex trace:* In `Adjacency.__init__`, list input containing `Adjacency` objects calls `concatenate(data)`. That helper repeatedly invokes `append`; the first append copies the first object, and subsequent appends concatenate Y when it is non-empty while retaining copied labels. Lines 85–86 copy tmp.Y to self, but execution continues. Line 180 unconditionally assigns the constructor argument Y; with the default

#### ✅ ~~F043 · HIGH · codex: confirmed (→medium)~~
`nltools/data/atlases/reporting.py:325-338` — **peaks.cluster_id and clusters.cluster_id are unrelated id spaces (different order AND dtype) so the two tables can't be joined**  
`peaks.cluster_id` comes straight from nilearn's 'Cluster ID' (string, e.g. '1','1a','1b', ordered by descending peak stat). `clusters.cluster_id` is produced independently by `_renumber_labels`, which numbers 1..K by descending cluster SIZE (int64). The two orderings differ, so `cluster_id==1` in `peaks` generally refers to a different physical cluster than `cluster_id==1` in `clusters`, and the dtypes (Utf8 vs Int64) don't even match for a join. A user cross-referencing peaks to clusters (the obvious use) gets silently wrong associations. The ClusterReport docstring presents both columns as `cluster_id` with no caveat.  
```python
"cluster_id": [str(c) for c in table["Cluster ID"].tolist()],   # peaks: nilearn order, str
... vs _renumber_labels(...)  # clusters: size order, int
```
*Fix (medium):* Derive peak cluster IDs from the same renumbered label volume (look up each peak's voxel in `renumbered`) so both tables share one integer id space, or rename the peaks column (e.g. `nilearn_cluster_id`) and document that the tables are not joinable. Add a test asserting a peak's cluster_id maps to the matching clusters row.
*Codex trace:* `cluster_report_data` independently creates `renumbered` labels using 26-connectivity and descending cluster size, then passes only the thresholded intensity image to `_build_peaks_dataframe`. Nilearn re-labels that image using 6-connectivity and numbers clusters by descending peak statistic; `_build_peaks_dataframe` converts those IDs, including subpeak IDs such as `1a`, to strings. `_build_clust

#### ✅ ~~F047 · HIGH · codex: confirmed (→medium)~~
`nltools/data/braindata/analysis.py:827-840` — **filter_data crashes when detrend/standardize passed via kwargs (the documented usage)**  
detrend and standardize are read non-destructively with kwargs.get(), then passed to clean() BOTH explicitly and again via **kwargs. Any call that sets detrend or standardize in kwargs raises TypeError: got multiple values for keyword argument 'detrend'. The BrainData.filter docstring and filter_data docstring explicitly instruct users to 'Pass detrend=True or standardize=True via kwargs to enable' — so following the documentation crashes.  
```python
standardize = kwargs.get("standardize", False)
    detrend = kwargs.get("detrend", False)
    ...
    out.data = clean(bd.data, t_r=1.0 / sampling_freq, detrend=detrend, standardize=standardize, high_pass=high_pass, low_pass=low_pass, **kwargs)
```
*Fix (trivial):* Pop instead of get: detrend = kwargs.pop('detrend', False); standardize = kwargs.pop('standardize', False). Then the explicit args are the single source and **kwargs no longer collides. Add a regression test filter(sampling_freq=2.0, high_pass=0.01, detrend=True).
*Codex trace:* BrainData.filter receives detrend/standardize in **kwargs and forwards that dict unchanged to filter_data. filter_data reads each with kwargs.get(), leaving the key present, then calls nilearn.signal.clean with the same argument both explicitly and through **kwargs. Python raises TypeError during argument binding before clean executes. Existing tests only exercise ensure_finite passthrough and mis

#### ✅ ~~F057 · HIGH · codex: confirmed (→low)~~
`nltools/data/braindata/io.py:807-816` — **upload_neurovault leaves `collection` unbound when create_collection fails, causing UnboundLocalError**  
When collection_id is None and api.create_collection(collection_name) raises ValueError (e.g. name already exists), the except block only prints a message and falls through. `collection` is never assigned, so the very next code path (add_image_to_collection accessing collection['name'] / collection['id']) raises UnboundLocalError. The intended 'friendly' print is useless: the function neither returns nor raises a clean error, and the return `collection` at the end would also be undefined.  
```python
try:
    collection = api.create_collection(collection_name)
except ValueError:
    print("Collection Name already exists.  Pick a ...")
... (falls through) ...
add_image_to_collection(api, collection, bd, tmp_dir, ...)
```
*Fix (trivial):* Re-raise (or `raise ValueError(...) from e`) inside the except, or `return` after printing. Do not continue with an unbound `collection`.
*Codex trace:* When collection_id is None, collection is assigned only by api.create_collection(). Its ValueError handler merely prints and falls through. Execution then creates tmp_dir and, for any nonempty BrainData, evaluates add_image_to_collection(api, collection, ...), where loading the unassigned local raises UnboundLocalError before the helper runs. An empty iterable would instead reach return collection

#### ✅ ~~F066 · HIGH · codex: confirmed~~
`nltools/data/collection/inference.py:518-551` — **isc_test bootstrap produces a statistically invalid p-value against ISC=0**  
isc_test resamples subjects with replacement and recomputes ISC for each draw, building `null`. But a nonparametric bootstrap of subjects produces a distribution centered on the OBSERVED ISC (H1), not on 0 (H0). The p-value `p = (sum(|null| >= |obs|)+1)/(n_permute+1)` therefore compares the observed ISC against a null that is itself centered at the observed value, yielding p ≈ 0.5 for strongly synchronized voxels and never rejecting H0 correctly. The docstring even claims the null is 'centered at 0', which the code does not do. A valid ISC test needs a sign-flip / phase-randomization / subject-wise permutation null (or the bootstrap CI must be reported, not converted to a p vs 0).  
```python
null[k] = _aggregate_corrs(corrs, metric).reshape(-1)  # bootstrap of subjects (H1-centered)
...
p = (np.sum(np.abs(null) >= np.abs(obs_map), axis=0) + 1) / (n_permute + 1)
```
*Fix (medium):* Replace the subject bootstrap with an H0 null (sign-flipping of per-subject correlations or circular/phase-shift of timeseries), or subtract the bootstrap mean to center before the comparison; align the docstring with whatever null is actually generated.
*Codex trace:* `BrainCollection.isc_test()` delegates directly to this function. It computes the observed ISC, resamples subject indices with replacement, recomputes ISC from the unmodified resampled time series, and stores the raw bootstrap statistics in `null`. No sign-flip, time-series permutation, or centering occurs before `abs(null) >= abs(obs_map)`. Thus the bootstrap distribution is centered near the obs

#### ✅ ~~F067 · HIGH · codex: confirmed (→low)~~
`nltools/data/collection/pipeline.py:66` — **BrainCollectionPipeline.n_subjects references nonexistent BrainCollection.n_images**  
The property returns `self._bc.n_images`, but BrainCollection exposes `n_subjects` / `__len__`, never `n_images` (see collection/core.py properties). Any access to `pipe.n_subjects` — including `__repr__`, which interpolates `self.n_subjects` — raises AttributeError. The behavior tests only call `.predict()` (which counts via `len(self._bc)`), so this path is untested and silently broken.  
```python
@property
    def n_subjects(self) -> int:
        """Number of subjects/images."""
        return self._bc.n_images
```
*Fix (trivial):* Return `len(self._bc)` (or `self._bc.n_subjects`). Add a test that touches `pipe.n_subjects` / `repr(pipe)`.
*Codex trace:* pipeline.py:66 returns self._bc.n_images. The concrete BrainCollection defines n_subjects and __len__ from _items, has no n_images property, and has no attribute fallback. Therefore pipe.n_subjects raises AttributeError. BrainCollectionPipeline.__repr__ interpolates self.n_subjects, so repr(pipe) raises the same error. Existing pipeline tests exercise construction, n_steps, and predict(), but not 

#### ✅ ~~F088 · HIGH · codex: confirmed (→medium)~~
`nltools/data/roc/__init__.py:229-231` — **accuracy_se uses p*p instead of p*(1-p) — wrong standard error of a proportion**  
The standard error of a proportion (accuracy) is sqrt(p*(1-p)/n). The code computes sqrt(p * p / n) = p/sqrt(n), which is wrong for every accuracy != 0.5 and systematically overstates SE. `np.mean(~self.misclass)` is the accuracy p; the second factor should be (1 - p).  
```python
self.accuracy_se = np.sqrt(
    np.mean(~self.misclass) * (np.mean(~self.misclass)) / self.n
)
```
*Fix (trivial):* Use `p = np.mean(~self.misclass); self.accuracy_se = np.sqrt(p * (1 - p) / self.n)`. Add a regression test on a known case.
*Codex trace:* `calculate()` constructs `misclass` as the Boolean union of false positives and false negatives. It sets `n = len(misclass)`, so `mean(~misclass)` is exactly the observed proportion correct, p. Lines 229–231 compute `sqrt(p*p/n)`, while the binomial standard error is `sqrt(p*(1-p)/n)`. No caller transforms this value; `summary()` reports it directly as “Accuracy SE.” The error does not affect clas

#### ✅ ~~F087 · HIGH · codex: confirmed (→low)~~
`nltools/data/simulator/__init__.py:392` — **Copy-paste bug: cov listification checks `type(cor)` instead of `type(cov)`**  
The block that wraps scalar `cov` into a list tests `type(cor) is int` in its second clause instead of `type(cov) is int`. So when `cov` is an int (not float) and `cor` has already been listified, `cov` is not wrapped, leaving downstream `len(cov)` / `cov[0]` indexing to behave incorrectly or raise. The immediately preceding block correctly uses `type(cor)` twice for `cor`.  
```python
if type(cov) is float or type(cor) is int:
    cov = [cov]
```
*Fix (trivial):* Change the second clause to `type(cov) is int`.
*Codex trace:* In `create_ncov_data`, scalar `cor` is first converted to `[cor]`. Therefore, at line 392, `type(cor) is int` is necessarily false for an integer `cor`. With integer `cov`, neither condition is true, so `cov` remains an int and line 400 executes `len(cov)`, raising `TypeError`. This violates the documented acceptance of integer `cov`. However, the proposed fix is incomplete: wrapping it as `[cov]`

#### ✅ ~~F149 · HIGH · codex: confirmed (→medium)~~
`nltools/mask.py:119` — **expand_mask iterates over nonzero INDICES of unique labels, not the label values**  
The loop `for i in np.nonzero(np.unique(mask.data))[0]` takes np.nonzero of the *unique-values array*, which returns positions of non-zero entries, not the label values. It only coincidentally works when labels are the contiguous set {0,1,...,n} (index == value). For any non-contiguous atlas (e.g. labels {0,5,10} after subsetting a parcellation) the loop runs over indices [1,2] and does `mask.data == 1`/`== 2`, matching zero voxels. Empirically verified: expand_mask on a {0,5,10} mask returns two all-zero masks. Silently produces empty ROI masks for a common input.  
```python
for i in np.nonzero(np.unique(mask.data))[0]:
    tmp.append((mask.data == i) * 1)
```
*Fix (trivial):* Iterate over the actual non-zero unique labels: `for i in np.unique(mask.data[mask.data != 0]):`.
*Codex trace:* At nltools/mask.py:117 labels are rounded to integers. Line 119 computes sorted unique label values, but `np.nonzero(...)[0]` returns positions within that unique-values array. Line 120 then incorrectly treats those positions as labels. For values `{0,5,10}`, iteration is over `1,2`, so both comparisons produce empty masks. No downstream logic corrects this, and no expand_mask test covers the beha

#### ✅ ~~F112 · HIGH · codex: confirmed~~ — **RESOLVED (commit `e55f6ff5`)**
> **Fixed before this discussion.** Commit `e55f6ff5` (`fix(pipelines)!: drop invalid permutation CVScheme, add predict(n_permute=) null`) dropped `'permutation'` from `CVSchemeType` (now `kfold|loso|loro|bootstrap`) and moved the label-permutation accuracy null to a dedicated `BrainCollectionPipeline.predict(n_permute=...)` outer loop over shuffled targets — not a train/test split. The 💬 marker was stale. Ledger thread #87 closed.

`nltools/pipelines/cv.py:212-244` — **Permutation CV scheme violates the (train_idx, test_idx) split contract, producing an invalid null**  
split() is documented to yield (train_indices, test_indices), and consumers rely on that: BrainCollectionPipeline._execute_pooled_cv does `test_data = pooled_data[test_idx]; test_y = y[test_idx]`. But _permutation_split yields (all_indices, perm_idx) where the SECOND element is a shuffled index vector meant to permute targets, not a test set. Routed through the facade, this trains on all data and 'scores' on pooled_data[perm_idx] / y[perm_idx] — i.e. X and y shuffled together with the SAME permutation, which is just a coherent subset and yields a normal (non-null) score. The intended permutation null (fit on X with shuffled y) never happens. Permutation testing via cv(method='permutation') is effectively broken.  
```python
for _ in range(self.n):
    perm_idx = self._rng.permutation(n_samples).astype(np.intp)
    yield indices.copy(), perm_idx
```
*Fix (medium):* Don't overload the CV tuple to mean 'permute y'. Either drop 'permutation' from CVScheme (it is not a train/test split) and handle permutation nulls in a dedicated code path, or have consumers special-case it. As-is it silently returns wrong statistics.
*Codex trace:* Traced the public path: BrainCollection.cv(method='permutation') constructs CVScheme(scheme='permutation'); predict() routes it to _execute_pooled_cv because is_loso is false. _permutation_split yields (all_indices, perm_idx), but _execute_pooled_cv universally interprets these as (train_idx, test_idx). It therefore fits on pooled_data/all y in original order, then scores on pooled_data[perm_idx] 

#### ✅ ~~F111 · HIGH · codex: confirmed (→medium)~~
`nltools/pipelines/pool.py:255-261` — **PooledData.repool() is broken for any real fitted_state produced by the facade**  
repool() calls _extract_param(), which only handles fitted_state that is a `list` of dicts and otherwise raises. But the only producer of PooledData.fitted_state — FittedBrainCollection.pool() in collection/pipeline.py (line 565) — sets `fitted_state=self._fitted`, which is a BrainCollection or a dict[str, BrainCollection], never a list of per-subject dicts. So repool() will always hit `raise ValueError(f"Cannot extract {param}...")` in practice. The only test (test_pipeline_pool.py:129) covers the no-fitted-state path; the working path is untested and non-functional. This is a publicly documented feature (`save_fitted=True` ... use repool()) that cannot succeed.  
```python
if isinstance(self.fitted_state, list):
    return np.stack([s.get(param) for s in self.fitted_state])
raise ValueError(f"Cannot extract {param} from fitted state")
```
*Fix (medium):* Either implement _extract_param for the actual fitted_state shape (BrainCollection / dict of BrainCollections) or remove repool()/_extract_param and the save_fitted plumbing until it is implemented. Add a green-path test.
*Codex trace:* FittedBrainCollection.pool(save_fitted=True) assigns self._fitted to PooledData.fitted_state. That value is explicitly a BrainCollection or dict[str, BrainCollection]. PooledData.repool() calls _extract_param(), whose sole success path requires a list, so both actual producer shapes raise ValueError. The only repool test covers fitted_state=None. Impact is reduced because current BrainCollection.f

#### ✅ ~~F125 · HIGH · codex: confirmed (→low)~~
`nltools/plotting/prediction.py:88-105` — **plot_roc returns None despite docstring; breaks Roc.plot's advertised return value**  
plot_roc creates a figure via plt.figure() but ends with a bare `return` (None). Its docstring says 'Will return a matplotlib ROC plot'. The only caller, Roc.plot (nltools/data/roc/__init__.py:298/301), does `fig = plot_roc(...)` and `return fig`, and Roc.plot's own docstring promises `Returns: fig`. As written, Roc.plot ALWAYS returns None — users who capture the figure get nothing. plot_scatter/plot_probability/plot_dist_from_hyperplane have the same return-None-vs-docstring mismatch.  
```python
plt.figure()
    plt.plot(fpr, tpr, color="red", linewidth=3)
    ...
    return   # <- returns None; docstring claims a figure
```
*Fix (trivial):* Capture and return the figure: `fig = plt.figure(); ...; return fig` (and return the FacetGrid/Axes for the seaborn-based functions).
*Codex trace:* `plot_roc()` creates a new Matplotlib figure with `plt.figure()` but does not retain it and ends with bare `return`, so it returns `None`. `Roc.plot()` assigns that result to `fig` in both successful branches (`gaussian` and `observed`) and returns it unchanged. Thus every successful `Roc.plot()` call returns `None`, contradicting both documented return contracts. The other three cited helpers lik

#### ✅ ~~F136 · HIGH · codex: confirmed~~
`nltools/stats/alignment.py:353-363` — **procrustes_distance compares observed disparity against a null of similarities**  
`sse` is the Procrustes disparity M^2 (lower = more similar). It is stored as `stats['similarity']` (mislabeled — it is a distance, not a similarity, contradicting the docstring 'similarity bounded 0 and 1') and passed to `_compute_pvalue` as the observed statistic. But the null distribution is built as `1 - x[2]` (similarity = 1 - disparity), so the observed statistic and null are on inverted scales. The resulting p-value compares apples to oranges and is not a valid permutation p-value. The test only checks p in [0,1], so it does not catch this.  
```python
stats = {"similarity": sse}
... all_p = [1 - x[2] for x in all_p]
stats["p"] = float(_compute_pvalue(np.array(sse), np.array(all_p), tail=tail)[0])
```
*Fix (small):* Put observed and null on the same scale. Either use similarity consistently — `observed = 1 - sse`, null `= [1 - x[2] ...]`, store `stats['similarity'] = 1 - sse` — or use disparity consistently for both. Add a test where two near-identical matrices yield a small p-value.
*Codex trace:* `procrust()` returns `disparity = sum((mtx1-mtx2)^2)`, where lower means more similar. `procrustes_distance()` uses the raw observed disparity as both `stats['similarity']` and the observed statistic, but converts every permuted disparity to similarity with `1 - x[2]`. `_compute_pvalue()` then directly compares these incompatible quantities. In a faithful reproduction using near-identical matrices

#### ✅ ~~F135 · HIGH · codex: confirmed (→medium)~~
`nltools/stats/corrections.py:65` — **holm_bonf ignores its `alpha` parameter (hardcoded 0.05)**  
The function accepts `alpha=0.05` and documents it as the alpha level, but the threshold array is computed with a hardcoded literal `0.05` instead of `alpha`. Calling holm_bonf(p, alpha=0.01) silently returns the alpha=0.05 result, giving wrong (anti-conservative) correction thresholds. No test exercises a non-default alpha, so this is uncaught.  
*Evidence:* `null = 0.05 / (nvox - np.arange(1, nvox + 1) + 1)`  
*Fix (trivial):* Replace the literal 0.05 with `alpha`: `null = alpha / (nvox - np.arange(1, nvox + 1) + 1)`. Add a regression test with alpha=0.01.
*Codex trace:* `holm_bonf(p, alpha=0.05)` documents and accepts `alpha`, but constructs its comparison thresholds as `0.05 / (...)`; `alpha` is never referenced. A discriminating run returned the same threshold (`0.02`) for alpha values 0.01, 0.05, and 0.10. For alpha=0.01, the correct threshold vector would reject all values in that example and return `-1`, so the current result is anti-conservative. Repository

### Medium severity

#### ✅ ~~F152 · MEDIUM · codex: confirmed (→low)~~
`nltools/__init__.py:1-16` — **__all__ advertises submodules (datasets, cross_validation) that are never imported, so attribute access fails**  
`__all__` lists `datasets`, `cross_validation`, `io`, `stats`, `utils`, `plotting`, `data`, `mask`, `templates`, but `__init__` only imports from `.data`, `.templates`, `.mask`, `.algorithms`, `.version`. `datasets` and `cross_validation` are never imported, so `import nltools; nltools.datasets` / `nltools.cross_validation` raise AttributeError (verified), despite being in `__all__`. `import nltools; nltools.datasets.fetch_pain()` fails confusingly.  
```python
"datasets",
...
"cross_validation",
# but __init__ never does `from . import datasets` / `import cross_validation`
```
*Fix (trivial):* Explicitly `from . import datasets, cross_validation, io, stats, utils, plotting, data` (or add a PEP 562 `__getattr__`) so every __all__ name is a real attribute.
*Codex trace:* `nltools/__init__.py` lists `datasets` and `cross_validation` in `__all__`, but lines 30–47 import only `.data`, `.templates`, `.version`, `.mask`, and `.algorithms`. There is no `__getattr__` or other lazy loader. A repository-wide trace found no import-time path loading `nltools.datasets` or `nltools.cross_validation`; their uses/tests explicitly import those submodules. Consequently, after only

#### ✅ ~~F012 · MEDIUM · codex: confirmed~~ — **RESOLVED by stripping ICC entirely (ledger #86/#3732): removed icc.py, stats.compute_icc, BrainData.icc + all icc tests & docs**
`nltools/algorithms/inference/icc.py:234-235` — **ICC1 is computed with the ICC3 error term, contradicting the cited Shrout & Fleiss one-way model**  
The module docstring cites Shrout & Fleiss (1979) and describes icc1 as 'One-way random effects'. But icc1 uses the same formula as icc3: `(MSR - MSE) / (MSR + (n_sessions-1)*MSE)`, where MSE = SSE/((n-1)(k-1)) excludes the session (column) sum of squares SSC. The Shrout & Fleiss ICC(1,1) one-way model has no column effect and must use the within-subject mean square MSW = (SSC+SSE)/(n*(k-1)), so ICC1 generally differs from ICC3. As written, icc1 and icc3 are numerically identical for every input. This matches the stats-layer convention (tests test_icc1_equals_icc3 and test_icc_formula_manual assert it), so it is intended, but it is statistically wrong for the stated one-way model and silently returns ICC3 values to users who select icc1 expecting Shrout-Fleiss ICC(1,1). The error is triplicated (vectorized 234-235, gpu_batch 345-346, single 393-394).  
```python
if icc_type == "icc1":
    ICC = (MSR - MSE) / (MSR + (n_sessions - 1) * MSE + EPSILON)
...
elif icc_type == "icc3":
    ICC = (MSR - MSE) / (MSR + (n_sessions - 1) * MSE + EPSILON)
```
*Fix (small):* Either compute true ICC(1,1) using MSW = (SSC+SSE)/(n_subjects*(n_sessions-1)) in place of MSE for icc1, or (if the identity is a deliberate project decision) drop the Shrout & Fleiss 'one-way random effects' framing and document that icc1==icc3 here. Align the docstrings/reference either way.
*Codex trace:* Traced public path: BrainData.icc(method='icc1') -> data/braindata/analysis.py::icc -> compute_icc_voxelwise -> vectorized, per-voxel CPU, or GPU-batch implementation. All three calculate SSE = SST - SSR - SSC and MSE = SSE/((n_subjects-1)*(n_sessions-1)), then use exactly the ICC3 formula for both icc1 and icc3. True one-way ICC(1,1) instead requires MSW = (SST-SSR)/(n_subjects*(n_sessions-1)) = 

#### ✅ ~~F013 · MEDIUM · codex: confirmed~~
`nltools/algorithms/inference/isc.py:1186-1194` — **isc_group_permutation_test bootstrap CI is centered on 0, not on the observed difference**  
For method='bootstrap', each bootstrap draw is centered inside _bootstrap_isc_group_numpy: `boot_diff = (isc1_boot - isc2_boot) - observed_diff` (line 836). The reported 'ci' is then `np.percentile(null_dist, ...)` on that centered distribution, so the confidence interval brackets ~0 rather than the observed group difference. Contrast isc_permutation_test (line 1847-1853), which correctly computes the CI from the UNCENTERED `bootstraps`. A returned CI that does not contain the reported point estimate is misleading and will be reported/plotted as-is.  
```python
ci_lower = np.percentile(null_dist, (100 - ci_percentile) / 2)
ci_upper = np.percentile(null_dist, ci_percentile + (100 - ci_percentile) / 2)
```
*Fix (small):* Compute the CI from the uncentered bootstrap draws (add observed_diff back, or return the uncentered distribution separately) for method='bootstrap', matching isc_permutation_test. For method='permute' a CI from the label-permutation null is conceptually a null band, not a CI of the estimate -- document or omit it.
*Codex trace:* Traced both bootstrap paths. `_bootstrap_isc_group_numpy` returns `(isc1_boot - isc2_boot) - observed_diff` at lines 835–838. Both sequential and CPU-parallel branches assign these centered draws directly to `null_dist` at lines 1121–1155. Lines 1185–1194 then compute `ci` from `null_dist` without restoring `observed_diff`, so the reported interval is centered near zero rather than around the esti

#### ✅ ~~F033 · MEDIUM · codex: confirmed~~
`nltools/data/adjacency/stats.py:264-269` — **threshold() treats a cutoff of 0.0 as 'not provided' due to truthiness test**  
The branch selection uses `if upper and lower:` / `elif upper:` / `elif lower:`. A legitimate numeric cutoff of 0.0 is falsy, so `threshold(lower=0.0, upper=5)` skips the two-sided branch and falls into `elif upper`, ignoring the lower bound entirely; `threshold(upper=0.0)` is silently a no-op. Users thresholding around zero (very common for correlation/z matrices) get silently wrong results.  
```python
if upper and lower:
    b.data[(b.data < upper) & (b.data > lower)] = 0
elif upper:
    b.data[b.data < upper] = 0
elif lower:
    b.data[b.data > lower] = 0
```
*Fix (trivial):* Test against None explicitly: `if upper is not None and lower is not None:` etc. (after the percentile-string coercion). Add a test with a 0.0 cutoff.
*Codex trace:* The public Adjacency.threshold() forwards upper/lower unchanged to nltools/data/adjacency/stats.py. After optional percentile coercion, that function uses truthiness. Thus lower=0.0, upper=5.0 fails the two-sided condition and executes only the upper branch, zeroing every value below 5—including negative values that should survive the lower bound. upper=0.0 matches no branch, so no thresholding oc

#### ✅ ~~F058 · MEDIUM · codex: confirmed (→low)~~
`nltools/data/braindata/io.py:540-543` — **load_from_url never removes the temp directory it creates (resource leak)**  
load_from_url makes a temp dir, downloads the NIfTI into it, and loads it, but never calls shutil.rmtree. Every URL load leaks a directory (with the downloaded file) into the system temp location. Contrast upload_neurovault, which does clean up via shutil.rmtree at line 856.  
```python
tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
os.makedirs(tmp_dir)
downloaded_file = nib.load(download_nifti(url, data_dir=tmp_dir))
load_from_file(bd, downloaded_file)  # tmp_dir never cleaned
```
*Fix (trivial):* Wrap in try/finally with shutil.rmtree(tmp_dir, ignore_errors=True), or use tempfile.TemporaryDirectory() as a context manager.
*Codex trace:* BrainData.__init__ classifies strings containing "://" as URLs and calls load_from_url. That function creates a directory under tempfile.gettempdir(), passes it to download_nifti, which writes the HTTP response into that directory, then nib.load opens the file and load_from_file eagerly materializes the masked image into bd.data via nilearn_apply_mask. No cleanup, context manager, destructor, or r

#### ✅ ~~F059 · MEDIUM · codex: confirmed (→low)~~
`nltools/data/braindata/io.py:540` — **Temp dir named from os.times()[-1] is collision-prone and os.makedirs will crash on collision**  
Both load_from_url (540) and upload_neurovault (818) name their temp dir with str(os.times()[-1]) (elapsed wall-clock, low resolution). Two calls in quick succession, or coarse clock ticks, can yield the same string; os.makedirs without exist_ok then raises FileExistsError. This is a fragile ad-hoc unique-name scheme.  
```python
tmp_dir = os.path.join(tempfile.gettempdir(), str(os.times()[-1]))
os.makedirs(tmp_dir)
```
*Fix (trivial):* Use tempfile.mkdtemp() or tempfile.TemporaryDirectory() instead of hand-rolling a name from os.times(); guarantees uniqueness and (with the context manager) cleanup.
*Codex trace:* Both `load_from_url` and `upload_neurovault` derive `tmp_dir` from `str(os.times()[-1])` and call `os.makedirs(tmp_dir)` without `exist_ok`. On this system, 10,000 immediate `os.times()[-1]` calls produced only 2 distinct values, confirming the clock is too coarse for uniqueness. Concurrent calls within one tick therefore select the same path, and one raises `FileExistsError`. `load_from_url` neve

#### ✅ ~~F050 · MEDIUM · codex: confirmed~~
`nltools/data/braindata/modeling.py:164-238` — **fit(inplace=False) still mutates bd.X_, bd.model_, and design_matrix, contradicting 'self unchanged'**  
The docstring states for inplace=False 'return Fit dataclass with results (bd unchanged)' and the worked example asserts brain_data is unchanged. But line 165 sets bd.X_ unconditionally, and lines 231-235 set bd.model_ (and bd.design_matrix for GLM) on the original bd even when inplace=False. A user who does fit1 = bd.fit(X=A, inplace=False); fit2 = bd.fit(X=B, inplace=False); bd.predict() will silently predict using the last throwaway fit's model_/X_, not any fit they think they discarded.  
```python
# Always store model_ and X_ for predict() to work (even if inplace=False)
    bd.X_ = X_model
    ...
    if not inplace:
        bd.model_ = target.model_
```
*Fix (small):* Either keep bd truly immutable for inplace=False (attach model_/X_ only to the returned Fit) or fix the docstring to state that model_/X_/design_matrix are updated on self so predict()/compute_contrasts() work. Prefer the former to match documented semantics.
*Codex trace:* `BrainData.fit()` delegates directly to `modeling.fit`. That function unconditionally assigns `bd.X_` at line 165. With `inplace=False`, it fits a copy, then assigns the fitted `target.model_` back to `bd.model_` and, for GLMs, assigns `target.design_matrix` to `bd.design_matrix` at lines 231–235. `predict_timeseries()` subsequently reads `bd.model_` and defaults to `bd.X_`, so successive non-inpl

#### ✅ ~~F051 · MEDIUM · codex: confirmed~~
`nltools/data/braindata/modeling.py:361-369` — **_assemble_ridge_cv_results uses np.searchsorted on model.alphas, silently wrong if alphas unsorted**  
best_idx = np.searchsorted(alpha_grid, best_alpha_arr) assumes alpha_grid (bd.model_.alphas) is sorted ascending. Ridge lets users pass an arbitrary alphas list; if it is unsorted, searchsorted returns wrong indices, so the extracted per-fold 'scores' cube is indexed at the wrong alpha and the reported per-fold/mean CV scores no longer correspond to the selected alpha — a silent statistical error with no exception raised.  
```python
alpha_grid = np.asarray(bd.model_.alphas)
    best_idx = np.searchsorted(alpha_grid, best_alpha_arr)
    best_idx = np.clip(best_idx, 0, n_alphas - 1)
```
*Fix (small):* Match values by identity rather than order, e.g. build an index via np.argmin(np.abs(alpha_grid[:,None]-best_alpha_arr[None,:]), axis=0), or sort alpha_grid (and cv_scores_ columns) before searchsorted. Add a test with unsorted alphas=[10.0, 0.1, 1.0].
*Codex trace:* Ridge passes self.alphas unchanged to solve_ridge_cv. The solver stores cv_scores_ in that supplied order and derives alpha_ as alphas[argmax_index]. _assemble_ridge_cv_results then reconstructs the index using np.searchsorted, which requires ascending input. For [10.0, 0.1, 1.0], searchsorted returns [3, 0, 2] for those values; clipping maps alpha 10.0 to index 2 (alpha 1.0) and alpha 0.1 to inde

#### ✅ ~~F049 · MEDIUM · codex: confirmed (→low)~~
`nltools/data/braindata/prediction.py:186-189` — **predict() MVPA hardcodes random_state=42 and exposes no random_state param**  
predict_mvpa builds StratifiedKFold/KFold with shuffle=True, random_state=42 whenever cv is an int. BrainData.predict has no random_state argument at all, so integer-cv decoding is silently pinned to a single fixed shuffle and cannot be varied for honest run-to-run variability or reproducibility control. This also violates the canonical trailing-kwarg order, which includes random_state=None for methods with stochastic behavior.  
```python
cv_splitter = (
    StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    if classifier
    else KFold(n_splits=cv, shuffle=True, random_state=42)
)
```
*Fix (small):* Add random_state: int | None = None to BrainData.predict and predict_mvpa and thread it into the KFold/StratifiedKFold construction (and the searchlight/ROI splitters).
*Codex trace:* `BrainData.predict` exposes no `random_state` and forwards integer `cv` to `predict_mvpa`, which constructs shuffled `StratifiedKFold`/`KFold` with an unconditional seed of 42. That shared splitter reaches whole-brain, searchlight, and ROI execution. Thus integer-CV folds cannot use another seed. Users can mitigate this by passing a custom CV splitter with their desired `random_state`, so results 

#### ✅ ~~F083 · LOW · codex: confirmed (→medium)~~
`nltools/data/designmatrix/transforms.py:142` — **downsample produces a lopsided final group for non-integer sampling-freq ratios**  
n_samples = sampling_freq/target may be non-integer; grouping uses int(n_samples) per group but n_groups = int(shape/n_samples), then all leftover rows are lumped into a single extra group via the remainder branch. For e.g. sampling_freq=2, target=0.7 (n_samples~2.857) most groups aggregate 2 rows while the trailing group aggregates all ~30 remainder rows, yielding a meaningless final downsampled sample. Only exact integer ratios are safe.  
*Evidence:* `n_samples = dm.sampling_freq / target\nn_groups = int(dm.shape[0] / n_samples)\nidx = pl.Series(np.repeat(np.arange(n_groups), int(n_samples)))\nif dm.shape[0] > len(idx): remainder = np.repeat(idx[-1]+1, dm.shape[0]-len(idx))`  
*Fix (small):* Either reject non-integer ratios with a clear ValueError, or distribute remainder rows evenly (e.g. np.floor(np.arange(n)/n_samples)) so the last bin isn't oversized. At minimum document the integer-ratio assumption.
*Codex trace:* Traced DesignMatrix.downsample() into transforms.downsample(). For 100 rows, sampling_freq=2, target=0.7: n_samples=2.857, n_groups=int(100/2.857)=35, and np.repeat(arange(35), int(2.857)) creates only 70 indices—35 groups of 2. The remainder branch assigns rows 70–99 to one additional group, so group sizes end [...,2,2,2,30]. Its mean is 84.5, while the preceding group mean is 68.5, silently coll

#### ✅ ~~F090 · MEDIUM · codex: confirmed~~
`nltools/data/roc/__init__.py:63-71` — **calculate() stores binary_outcome without asarray/flatten, unlike __init__, breaking boolean indexing for list input**  
__init__ normalizes `self.binary_outcome = np.asarray(binary_outcome).flatten()`, but calculate() does `self.binary_outcome = deepcopy(binary_outcome)` with no conversion. Since the rest of calculate relies on boolean-array semantics (`self.input_values[self.binary_outcome]`, `~self.binary_outcome`, `np.sum(self.binary_outcome)`), passing a plain Python list to calculate() will raise or silently misbehave (`~list` is a TypeError; list boolean-indexing an ndarray does integer fancy-indexing). input_values is likewise never coerced to ndarray in either place, so `.squeeze()` assumes ndarray.  
```python
if binary_outcome is not None:
    self.binary_outcome = deepcopy(binary_outcome)
```
*Fix (small):* Normalize consistently: `self.binary_outcome = np.asarray(binary_outcome).astype(bool).flatten()` and `self.input_values = np.asarray(input_values)` in both __init__ and calculate.
*Codex trace:* `__init__` converts `binary_outcome` to an ndarray, but `calculate(binary_outcome=...)` stores it unchanged. A Boolean Python list survives `wh[self.binary_outcome]` because NumPy recognizes Boolean lists as masks, contrary to one detail in the report, but execution then reaches `wh[~self.binary_outcome]`, where `~list` raises TypeError. Separately, list-valued `input_values` fails at `self.input_

#### ✅ ~~F151 · MEDIUM · codex: confirmed~~
`nltools/mask.py:247` — **roi_to_brain initializes the 2-D output background to 1.0 instead of 0.0**  
The 1-D branch uses `out.data = np.zeros(...)` so uncovered voxels are 0. The 2-D branch uses `out.data = np.ones((arr.shape[1], out.data.shape[1]))`. Expanded-mask rows need not tile every voxel, so background voxels stay at 1.0 in the stacked result — inconsistent with the 1-D case and almost certainly unintended (corrupts downstream maps/plots).  
```python
out = mask_x.copy()
out.data = np.ones((arr.shape[1], out.data.shape[1]))
```
*Fix (trivial):* Use `np.zeros(...)` to match the 1-D branch background fill.
*Codex trace:* In the 1-D path, `roi_to_brain` copies one ROI image, zero-initializes every voxel, then writes ROI values. In the 2-D path, it copies the expanded mask, initializes every output voxel to 1.0, and overwrites only voxels where each ROI row equals 1. `expand_mask` creates rows only for nonzero labels and does not guarantee their union covers the underlying BrainData mask. The test suite’s sphere ROI

#### ✅ ~~F113 · MEDIUM · codex: confirmed (→low)~~
`nltools/pipelines/cv.py:268-269` — **n_splits() over-reports bootstrap folds relative to what split() actually yields**  
_bootstrap_split() does `if len(test_idx) == 0: continue`, skipping any bootstrap draw with no out-of-bag samples, so the number of yielded folds can be < self.n. But n_splits() returns self.n unconditionally for bootstrap. Any consumer that pre-allocates or divides by n_splits() (progress bars, mean aggregation counts) will disagree with the true fold count.  
```python
if self.scheme == "bootstrap":
    return self.n
```
*Fix (small):* Document that bootstrap n_splits is an upper bound, or make split() always yield exactly n folds (redraw until OOB is non-empty) so the two agree.
*Codex trace:* `split()` dispatches bootstrap requests to `_bootstrap_split(n_samples)`, which performs exactly `self.n` draws but skips every draw whose OOB `test_idx` is empty. Conversely, `n_splits()` returns `self.n` unconditionally and documents this as the number of folds that will be generated. The existing test explicitly acknowledges `len(splits) <= n`. A deterministic check with `n=20, random_state=42`

#### ✅ ~~F120 · LOW · codex: confirmed (→medium)~~
`nltools/pipelines/pool.py:379-384` — **StatResult.threshold silently falls back to uncorrected on any FDR exception**  
The FDR branch wraps false_discovery_control in a bare `except Exception` and, on any failure, falls back to `mask = self.p_map < alpha` (uncorrected). This masks real bugs (e.g. NaNs, shape errors) and can silently return uncorrected significance while the user believes FDR was applied. The comment justifies it as 'older scipy', but false_discovery_control has been available since scipy 1.11.  
```python
except Exception:
    # Fallback for older scipy
    mask = self.p_map < alpha
```
*Fix (trivial):* Drop the bare-except fallback (require the scipy version the project already depends on), or at minimum warn loudly and narrow the caught exception.
*Codex trace:* StatResult.threshold(method="fdr") calls false_discovery_control(self.p_map.ravel()) inside a bare except Exception; any runtime failure then executes the uncorrected mask self.p_map < alpha without warning. A real p_map containing [0.001, 0.04, 0.5, 0.6, NaN] makes SciPy raise ValueError; threshold() silently returns both 0.001 and 0.04 as significant, although FDR adjustment of the finite values

#### ✅ ~~F126 · MEDIUM · codex: confirmed~~
`nltools/plotting/adjacency.py:69-75` — **plot_stacked_adjacency swaps which input maps to upper vs lower triangle depending on normalize**  
In the non-normalized branch the upper triangle comes from adjacency2 and the lower from adjacency1. In the normalize=True branch (the DEFAULT) it is reversed: upper is built from adjacency1 and lower from adjacency2. So the same two inputs land in opposite triangles depending on the normalize flag, which silently changes the meaning of the plot. Additionally `upper / np.max(upper)` divides by the max over the full (mostly-zero) matrix, not the max-abs of the triangle, so all-negative mean-centered values can yield max==0 -> division producing inf/nan.  
```python
upper = np.triu(adjacency2.squareform(), k=1); lower = np.tril(adjacency1.squareform(), k=-1)
if normalize:
    upper = np.triu((adjacency1 - adjacency1.mean()).squareform(), k=1)
    lower = np.tril((adjacency2 - adjacency2.mean()).squareform(), k=-1)
```
*Fix (small):* Keep a consistent mapping (adjacency1->upper or ->lower) across both branches, and normalize by max-abs of the triangle values to avoid /0.
*Codex trace:* At lines 69–75, normalize=False maps adjacency2 to the upper triangle and adjacency1 to the lower; normalize=True explicitly rebuilds them with adjacency1 upper and adjacency2 lower. Thus toggling a scaling option silently swaps dataset identity. The only caller uses the default, and the sole test checks only that plotting runs, so neither catches this. Normalization can also divide by zero for co

#### ✅ ~~F127 · MEDIUM · codex: confirmed (→low)~~
`nltools/plotting/adjacency.py:223-226` — **plot_between_label_distance spawns a stray blank figure when ax is supplied**  
When the caller passes an ax, the else-branch calls plt.figure(), creating an extra empty figure that is never used or closed. This leaks a blank figure into pyplot's state every call and will show up as an empty panel in notebooks. The heatmap is then drawn onto the supplied ax, so the new figure serves no purpose.  
```python
if ax is None:
        _, ax = plt.subplots(1)
    else:
        plt.figure()  # <- creates and abandons a blank figure
```
*Fix (trivial):* Drop the else branch entirely; only create a figure/axes when ax is None.
*Codex trace:* At adjacency.py:223-226, a non-None ax unconditionally triggers plt.figure(), creating a new pyplot-managed figure. Both plotting paths later call sns.heatmap(..., ax=ax), and seaborn draws—including the colorbar—through the supplied axis and its owning figure. The newly created figure is therefore unused and never closed. Matplotlib retains pyplot-created figures until closed, and inline notebook

#### ✅ ~~F137 · MEDIUM · codex: confirmed (→low)~~
`nltools/stats/alignment.py:59` — **align() same-type validation is a no-op**  
The guard is meant to ensure all list elements are the same type (per its error message), but `all(type(x) for x in data)` evaluates the truthiness of each `type(x)` object, which is always truthy. The check never raises regardless of mixed types, so a list mixing BrainData and ndarray passes validation and later fails opaquely.  
```python
if not all(type(x) for x in data):
    raise ValueError("Make sure all objects in the list are the same type.")
```
*Fix (trivial):* Check homogeneity, e.g. `if len({type(x) for x in data}) > 1:` or `if not all(isinstance(x, type(data[0])) for x in data):`.
*Codex trace:* In align(), `all(type(x) for x in data)` is always true for any non-empty list because every `type(x)` object is truthy. The subsequent branch checks only `data[0]` and assumes every element matches it. With BrainData first, `[x.data.T for x in data]` reaches the ndarray's `.data` memoryview and fails; with ndarray first, `[x.T for x in data]` reaches BrainData, which has no `.T`. Existing alignme

#### ✅ ~~F153 · MEDIUM · codex: confirmed (→low)~~
`nltools/templates/matching.py:189` — **get_bg_image truncates voxel size with int() while sibling functions round**  
`get_bg_image` uses `resolution = int(voxel_dims[0])` (truncation), whereas `match_resolution` (l.62) and `is_standard_space` (l.140) use `np.round`/`round`. A near-2mm affine with float error (e.g. 1.999mm zooms) truncates to 1, so the SUPPORTED_RESOLUTIONS check spuriously fails and the function silently falls back to `getattr(cfg, img_type)` (the config default-resolution image) instead of the correct 2mm background — mismatched anatomical under plotted stats.  
*Evidence:* `resolution = int(voxel_dims[0])`  
*Fix (trivial):* Use `int(round(float(voxel_dims[0])))` to match the sibling functions.
*Codex trace:* `is_standard_space()` rounds 1.999 mm to an integer-compatible 2 mm and permits plotting. The slice/viewer callers then invoke `get_bg_image()`, where `int(1.999)` becomes 1. For the `default` template, 1 is unsupported, so it silently returns the configured background instead of resolving the 2 mm image. If the valid configuration is `BrainSpaceConfig(template="default", resolution=3)`, this prod

### Low severity

#### ✅ ~~F016 · LOW · codex: confirmed~~
`nltools/algorithms/inference/timeseries.py:74-77` — **circle_shift 1D can draw a zero (identity) shift and is inconsistent with the 2D path**  
For 1D data the random shift is `rng.choice(np.arange(len(data)))`, whose support includes 0. A shift of 0 yields `np.concatenate([data[0:], data[:0]])`, i.e. the untransformed series, so the permutation null occasionally includes the identity. The 2D path uses `rng.randint(1, n_samples)` which correctly excludes 0. This makes 1D vs 2D nulls behave differently and slightly biases 1D p-values conservative. (`replace=False` on the scalar draw is also a no-op arg.)  
```python
if shift_amount is None:
    shift_amount = rng.choice(np.arange(len(data)), replace=False)
```
*Fix (trivial):* Use `rng.randint(1, len(data))` for the 1D random shift to match the 2D path and exclude the identity.
*Codex trace:* The public wrapper documents random shifts as [1, n_timepoints−1], but the 1D engine samples from np.arange(len(data)), including 0. For shift_amount=0, both slices resolve to data[0:] and data[:0], returning the original series. The correlation permutation callers use this 1D path, so an identity draw places the observed correlation into the null distribution; _compute_pvalue counts it as at leas

#### ✅ ~~F038 · LOW · codex: confirmed~~
`nltools/data/adjacency/utils.py:70-74` — **import_single_data flattens a stack when matrix_type='directed_flat'**  
For the 'directed_flat' explicit type, the code unconditionally does `data = np.asarray(data).flatten()`, collapsing any 2-D stack of already-flattened directed matrices into a single 1-D vector and then reporting is_single_matrix via a 1-D test. The 'distance_flat'/'similarity_flat' branches correctly leave 2-D stacks alone. Passing a stack of directed flat vectors therefore silently corrupts the object into a single matrix.  
```python
elif matrix_type.lower() == "directed_flat":
    matrix_type = "directed"
    data = np.asarray(data).flatten()
    issymmetric = False
    is_single_matrix = test_is_single_matrix(data)
```
*Fix (small):* Only flatten when input is 1-D (mirror the *_flat symmetric branches): `data = np.asarray(data)` and leave 2-D stacks intact; derive is_single_matrix from ndim.
*Codex trace:* `Adjacency.__init__` passes a non-list 2-D array directly to `import_single_data`. In the `directed_flat` branch, `np.asarray(data).flatten()` collapses `(n_matrices, n_nodes**2)` to one vector; `test_is_single_matrix` then necessarily returns True because it tests `ndim == 1`. The constructor stores that result unchanged. Downstream `shape`, `len`, iteration, and `squareform` consequently treat t

#### ✅ ~~F063 · LOW · codex: confirmed~~
`nltools/data/braindata/cache.py:148` — **CacheManager.load uses allow_pickle=True unnecessarily, weakening cache-poisoning safety**  
Cached npz files store only plain numeric arrays (indices, indptr, shape, radius, etc. — see neighborhoods.py save call), yet load passes allow_pickle=True. This is unnecessary and turns any tampered/corrupt cache file in ~/.nltools/cache into an arbitrary-code-execution vector on load.  
*Evidence:* `return dict(np.load(path, allow_pickle=True))`  
*Fix (trivial):* Drop allow_pickle=True (default False). If any caller genuinely stores object arrays, gate that per-key rather than globally.
*Codex trace:* CacheManager.load converts the entire NpzFile to a dict with allow_pickle=True, which accesses every member and unpickles object arrays. The sole in-repo production save caller in neighborhoods.py writes only integer arrays and numeric/string scalars, none requiring pickle; tests likewise use numeric and Unicode arrays. A crafted object-array member executes its pickle payload during dict(np.load(

#### ✅ ~~F082 · LOW · codex: confirmed~~
`nltools/data/designmatrix/append.py:28` — **_check_dtype_compatibility only diffs each frame against dfs[0], missing mismatches among later frames**  
base_schema is captured once from dfs[0] and never updated. A column that is absent from dfs[0] but present in dfs[1] and dfs[2] with mismatched dtypes is not detected, so the actionable error is skipped and users fall back to the cryptic polars SchemaError the helper was written to prevent (e.g. diagonal vertical append of run-separated columns that only appear in later runs).  
*Evidence:* `base_schema = dict(dfs[0].schema)\nfor idx, df in enumerate(dfs[1:], start=1):\n    for col, dtype in df.schema.items():\n        if col in base_schema and base_schema[col] != dtype: ...`  
*Fix (small):* Accumulate seen column->dtype across all frames (first occurrence wins) and compare each subsequent frame against the accumulated map, recording which dm index first defined the dtype for the error message.
*Codex trace:* Both vertical append paths pass all frames to _check_dtype_compatibility immediately before strict pl.concat(..., how="diagonal"). The helper compares every later schema only with dfs[0].schema and never records columns first encountered later. I reproduced the exact case with schemas [{'base': Int64}, {'later': Int64}, {'later': Float64}]: the helper performs no comparison for 'later', then Polar

#### ✅ ~~F101 · LOW · codex: confirmed~~
`nltools/data/simulator/__init__.py:789-804` — **plot_grid_simulation skips thresholding when already fit, leaving self.thresholded possibly None**  
plot_grid_simulation only calls fit()+threshold_simulation inside `if not self.isfit`. If the object was already fit but threshold_simulation was never run, self.thresholded is None and `a[1].imshow(self.thresholded)` / the threshold title fail or plot nothing. The guard conflates 'fit' with 'thresholded'.  
```python
if not self.isfit:
    self.fit()
    self.threshold_simulation(...)
self.run_multiple_simulations(...)
```
*Fix (small):* Guard fit() and threshold_simulation independently (e.g. threshold whenever self.thresholded is None or params changed).
*Codex trace:* `__init__` initializes `isfit=False` and `thresholded=None`. `fit()` sets only `t_values`, `p_values`, and `isfit=True`; it does not threshold. Therefore `sim.fit(); sim.plot_grid_simulation(...)` skips `threshold_simulation()` because `isfit` is already true. `run_multiple_simulations()` populates `multiple_thresholded`, not `thresholded`, so `a[1].imshow(self.thresholded)` receives `None` and ca

#### ✅ ~~F102 · LOW · codex: confirmed~~
`nltools/data/simulator/__init__.py:157-189` — **n_spheres default-center construction is malformed and only handles int radius**  
When center is None, `center = [[dims0/2, dims1/2, dims2/2] * len(radius)]` uses list-repetition, producing a single 3*len-element list wrapped once — not one [x,y,z] per radius; it only coincidentally works for len(radius)==1. Also `isinstance(radius, int)` misses numpy ints/floats, and float centers (dims/2) are passed as ogrid slice bounds. Multi-sphere default centering is effectively broken.  
```python
center = [
    [dims[0] / 2, dims[1] / 2, dims[2] / 2] * len(radius)
]  # default value for centers
```
*Fix (small):* Build `center = [[dims[0]//2, dims[1]//2, dims[2]//2] for _ in radius]`; broaden the radius scalar check and cast centers to int.
*Codex trace:* Traced `n_spheres`: Python `int` radius is wrapped in a list. With `radius=[r1, r2]` and `center=None`, the expression creates one outer element containing six coordinates, so `len(center)==1` while `len(radius)==2`; the validation fails and raises `ValueError` before calling `sphere`. Scalar Python floats and NumPy integer/float scalars are not wrapped and subsequently fail at `len(radius)`. Howe

#### ✅ ~~F160 · LOW · codex: confirmed~~
`nltools/datasets.py:120` — **download_nifti issues a streaming request with no timeout**  
`requests.get(url, stream=True)` has no timeout, so a stalled server hangs the call indefinitely. Poor default for a user-facing download helper — a network hiccup blocks the kernel forever with no feedback.  
```python
r = requests.get(url, stream=True)
r.raise_for_status()
```
*Fix (trivial):* Pass a `timeout=` (e.g. `(10, 60)`) and use the response as a context manager to guarantee closure.
*Codex trace:* `download_nifti()` directly calls `requests.get(url, stream=True)` without a timeout, then synchronously consumes `r.iter_content()`. Requests therefore has no connect or read timeout, and a server that stops responding can block either the initial request or a streamed read indefinitely. The `RequestException` handler does not help because no timeout exception will be generated. The only caller f

#### ✅ ~~F158 · LOW · codex: confirmed~~
`nltools/io/h5.py:57` — **is_h5_path uses substring matching instead of suffix matching**  
`return ".h5" in file_name or ".hdf5" in file_name` matches anywhere in the path, so a non-h5 file whose path merely contains the substring (e.g. `results.h5.summary.csv`, or a dir `.h5cache/data.nii`) is misclassified as HDF5, routing load/save down the wrong serializer.  
*Evidence:* `return ".h5" in file_name or ".hdf5" in file_name`  
*Fix (trivial):* Use `file_name.lower().endswith((".h5", ".hdf5"))`.
*Codex trace:* `is_h5_path` literally uses substring membership. Its callers use the result for serializer dispatch: BrainData validation routes matching paths to `load_from_h5`, which opens them with h5py; BrainData.write routes them to `to_h5`. Adjacency load/write and DesignMatrix.write make the same HDF5 dispatch. Therefore paths such as `results.h5.summary.csv` or `.h5cache/data.nii` are genuinely misclassi

#### ✅ ~~F156 · LOW · codex: confirmed~~
`nltools/mask.py:83-95` — **create_sphere mishandles non-int radius when multiple coordinates are given**  
With a list of coordinate triples, radius is normalized only when `isinstance(radius, int)`; a float radius (5.0) or numpy int falls through both branches and stays scalar, so `zip(radius, coordinates)` raises 'float object is not iterable'. Default 5 is int so the happy path works, but any float radius with multiple spheres crashes.  
```python
elif isinstance(radius, int):
    radius = [radius] * len(coordinates)
...
for r, c in zip(radius, coordinates):
```
*Fix (trivial):* Broaden to `np.isscalar(radius)` or `isinstance(radius, (int, float, np.integer, np.floating))`.
*Codex trace:* For nested list coordinates, `any(isinstance(i, list) for i in coordinates)` enters the multi-sphere branch. Only a Python `list` radius is validated and only a Python `int` is replicated. Scalar `float`, `np.int64`, and `np.float64` values remain scalar and reach `zip(radius, coordinates)`, which raises `TypeError: '<type>' object is not iterable`. Existing tests cover a radius list and Python `i

#### ✅ ~~F108 · LOW · codex: confirmed~~
`nltools/models/ridge.py:93` — **Mutable default argument 'concentration=[0.1, 1.0]'**  
concentration uses a mutable list as a default, unlike the adjacent alphas which correctly uses None + fallback. It is currently only stored (not mutated), so no live bug, but it is the exact mutable-default pattern flagged as a correctness smell and is inconsistent with the alphas handling two lines below.  
*Evidence:* `concentration=[0.1, 1.0],  ...  self.alphas = alphas if alphas is not None else [0.1, 1.0, 10.0]`  
*Fix (trivial):* Default concentration=None and assign self.concentration = [0.1, 1.0] if concentration is None else concentration.
*Codex trace:* `Ridge.__init__` uses one shared list object as the default and assigns it directly to `self.concentration`. Thus two default-constructed instances share the same object; mutating `model1.concentration` also changes `model2.concentration` and future instances. During banded fitting, that changed value is passed through `solve_banded_ridge_cv` to `generate_dirichlet_samples`, affecting results. The

#### ✅ ~~F119 · LOW · codex: confirmed~~
`nltools/pipelines/pool.py:152-166` — **_fit_one applies contrast for ttest but ignores it for paired_ttest/anova**  
For model='ttest', `data` is the contrast-applied array. But the paired_ttest and anova branches operate on `self.data[:, 0/1/i, :]` directly, ignoring any `contrast` argument the caller passed (which is still stored on the returned StatResult.contrast). A user passing a contrast to paired_ttest/anova gets silently un-contrasted results labeled with that contrast.  
*Evidence:* `t_vals, p_vals = stats.ttest_rel(self.data[:, 0, :], self.data[:, 1, :], axis=0)`  
*Fix (small):* Either reject `contrast` for paired_ttest/anova (raise) or apply it consistently; don't accept-and-ignore.
*Codex trace:* `PooledData.fit()` passes `contrast` into `_fit_one()`. `_fit_one()` computes contrast-applied local `data`, but `paired_ttest` calls `ttest_rel()` on `self.data[:, 0, :]` and `self.data[:, 1, :]`, while `anova` calls `f_oneway()` on every condition in `self.data`. Both return `StatResult(..., contrast=contrast)`. A runtime check confirmed that distinct valid contrasts produced identical statistic

## API consistency (46)

_v0.6.0 is the one breaking release where these rename for free. Canonical table in CLAUDE.md._

### High severity

#### ✅ ~~F048 · HIGH~~ — **RESOLVED by ICC strip (ledger #3732): BrainData.icc removed entirely**
`nltools/data/braindata/__init__.py:948-985` — **icc() exposes internal parallel= vocabulary and n_jobs is a silent no-op by default**  
Per v0.6.0 conventions the facade must not expose parallel= (canonical is n_jobs; stats-layer internals may keep parallel= but the facade translates). BrainData.icc leaks parallel=None|'cpu'|'gpu' straight through. Worse, the facade defaults parallel=None while the underlying compute_icc_voxelwise defaults parallel='cpu', and n_jobs is 'Only used when parallel=cpu'. So the documented n_jobs=-1 default does nothing: setting n_jobs alone never parallelizes because parallel stays None. Users must set BOTH parallel='cpu' and n_jobs. This also diverges from bootstrap(), which selects GPU via backend= rather than parallel=, so two methods on the same facade use two different, non-canonical device vocabularies.  
```python
def icc(self, n_subjects, n_sessions, method="icc2", parallel=None, n_jobs=-1, max_gpu_memory_gb=4.0):
    ...
    return icc(self, n_subjects, n_sessions, method=method, parallel=parallel, n_jobs=n_jobs, ...)
```
*Fix (small):* Drop parallel= from the facade; derive the internal parallel string from n_jobs (n_jobs==1 -> None/serial, n_jobs!=1 -> 'cpu') and translate at the boundary, matching the facade-translation rule. Unify GPU selection with bootstrap() under one kwarg name.

#### ✅ ~~F103 · HIGH~~ — implemented (`*` marker added: `fit(self, X, y=None, *, design_matrices=None, events=None, **kwargs)`)
`nltools/models/glm.py:131` — **Glm.fit lacks keyword-only '*' marker; positional call binds design matrix to y**  
fit(self, X, y=None, design_matrices=None, events=None, **kwargs) has 3+ kwargs after the primary data arg but no keyword-only '*' marker, violating the project rule ('*' required in any public method with 3+ kwargs). Worse, it is a genuine footgun: a user with sklearn muscle memory calling model.fit(img, my_design_matrix) silently binds the design matrix to y (which is 'Not used') and leaves design_matrices=None, so nilearn tries to build a design from events=None and either errors confusingly or fits garbage. The intended call is design_matrices=.  
*Evidence:* `def fit(self, X, y=None, design_matrices=None, events=None, **kwargs):`  
*Fix (trivial):* Change signature to fit(self, X, y=None, *, design_matrices=None, events=None, **kwargs) so run images and design matrices can't be swapped positionally.

#### ✅ ~~F139 · HIGH~~ — implemented (`mode`→`method` in `regress()`)
`nltools/stats/regression.py:18` — **regress() uses `mode=` for algorithm choice (banned name; inconsistent with facade)**  
`mode` is explicitly on the v0.6.0 banned list for algorithm/variant choice (canonical = `method`). The BrainData/Adjacency facade methods already use `method='ols'`, so the standalone `regress` (re-exported at top level) is the odd one out. v0.6.0 is the breaking release specifically for canonicalizing these names.  
*Evidence:* `def regress(X, Y, mode: str = "ols", stats: str = "full"):`  
*Fix (trivial):* Rename `mode` to `method`. Update NotImplementedError message and the docstring accordingly.

#### ✅ ~~F164 · HIGH~~ — implemented (`scheme`→`spatial_scale` + `parcellation`→`roi_mask` on BrainCollection.align, translated to LocalAlignment at boundary)
`nltools/data/collection/__init__.py:1062` — **collection.align leaks legacy names**  
align exposes scheme=/parcellation=/n_iter= untranslated.  
*Evidence:* `def align(self, *, method, scheme, parcellation, n_iter)`  
*Fix (small):* Rename to spatial_scale=/roi_mask=.

#### ✅ ~~F177 · HIGH~~ — **RESOLVED by ICC strip (ledger #3732): BrainData.icc removed entirely**
`nltools/data/braindata/__init__.py:948-987` — **BrainData.icc leaks stats-layer `parallel=` into the public facade instead of translating from `n_jobs`**  
The v0.6.0 convention says facades expose `n_jobs: int = -1` and translate the stats-layer `parallel=` name at the boundary; here the public method exposes BOTH `parallel=None` and `n_jobs=-1`. Users get two overlapping knobs for the same concept, and the illegal legacy name is now part of the public API of a breaking release. It also has 6 params with no keyword-only `*` marker (rule requires `*` in any public method with 3+ kwargs).  
*Evidence:* `def icc(self, n_subjects, n_sessions, method="icc2", parallel=None, n_jobs=-1, max_gpu_memory_gb=4.0):`  
*Fix (small):* Drop `parallel=` from the facade signature; keep only `n_jobs`, and translate to `parallel=` when calling `.analysis.icc`. Insert `*` after `n_sessions` so method/n_jobs/max_gpu_memory_gb are keyword-only.

### Medium severity

#### ✅ ~~F004 · MEDIUM~~ — implemented (LocalAlignment gained `progress_bar: bool = False`; all tqdm gated on it; facade forwards it)
`nltools/algorithms/alignment/local.py:583` — **LocalAlignment unconditionally prints tqdm progress bars and has no progress_bar parameter**  
fit() creates a tqdm bar unconditionally (line 583) and the sequential transform() calls `iter_neighborhoods(progress_bar=True)` hardcoded (line 711). The class exposes no `progress_bar` control, violating the canonical `progress_bar: bool = False` (default-off) convention. Consequently the `align` facade (data/collection/inference.py:554) accepts a `progress_bar` argument but never forwards it to LocalAlignment — it is dead at the facade and uncontrollable at the algorithm layer, so users always get progress output whether or not they asked for it.  
```python
pbar = tqdm(total=n_regions, desc=region_type.capitalize(), unit="regions")  # fit, always on
... iter_neighborhoods(progress_bar=True)  # transform line 711, hardcoded
```
*Fix (small):* Add `progress_bar: bool = False` to LocalAlignment, gate all tqdm usage on it, and have the facade forward its `progress_bar` argument into the constructor.

#### ✅ ~~F005 · MEDIUM~~ — implemented (`*` marker added to SRM/DetSRM/HyperAlignment fit/transform after the sklearn data args)
`nltools/algorithms/alignment/srm.py:212` — **Public estimator methods with 3+ kwargs lack the required keyword-only '*' marker**  
CLAUDE.md requires a keyword-only `*` marker in any public method with 3+ kwargs. SRM.fit(X, y, parallel, n_jobs, max_gpu_memory_gb, pad_samples) exposes five kwargs with no `*`; likewise SRM.transform, DetSRM.fit, DetSRM.transform, and HyperAlignment.fit/transform (hyperalignment.py:218, 388). These are public (exported in __all__ and re-exported via nltools.algorithms), so callers can pass them positionally and future signature reordering will silently break code.  
*Evidence:* `def fit(self, X, y=None, parallel="cpu", n_jobs=-1, max_gpu_memory_gb=4.0, pad_samples=True)`  
*Fix (small):* Insert `*` after the primary data arg (and after `y` where the sklearn signature must keep X, y positional): e.g. `def fit(self, X, y=None, *, parallel=..., n_jobs=..., ...)`.

#### ✅ ~~F002 · MEDIUM~~ — implemented (SRM/DetSRM.transform now prefer the explicit `n_jobs` arg, mirroring HyperAlignment)
`nltools/algorithms/alignment/srm.py:352` — **SRM.transform / DetSRM.transform silently ignore the caller's n_jobs argument**  
Both transform methods declare `n_jobs: int = -1` but then compute `n_jobs_to_use = getattr(self, "_n_jobs", n_jobs)`, which unconditionally prefers the value stored during fit and only falls back to the argument if `_n_jobs` was never set. So `srm.transform(X, n_jobs=2)` after `srm.fit(X, n_jobs=-1)` runs with -1, not 2 — the documented parameter is effectively dead. This is also inconsistent with HyperAlignment.transform (hyperalignment.py:417) which prefers the explicit argument (`n_jobs if n_jobs != -1 else getattr(...)`).  
*Evidence:* `n_jobs_to_use = getattr(self, "_n_jobs", n_jobs)  # SRM line 352, DetSRM line 897`  
*Fix (small):* Prefer the explicitly-passed argument when it differs from the default (mirror HyperAlignment: `n_jobs if n_jobs != -1 else getattr(self, "_n_jobs", -1)`), or drop the transform-level n_jobs param entirely if fit-time settings are meant to win.

#### ✅ ~~F025 · MEDIUM~~ — implemented (`*` marker added to solve_ridge_cv / solve_banded_ridge_cv after data args)
`nltools/algorithms/ridge/solvers.py:26, 618` — **Public solvers lack the keyword-only `*` marker that their sibling cross_val_predict_ridge uses**  
cross_val_predict_ridge (line 901) correctly puts `*` after X, Y so its ~10 options are keyword-only. solve_ridge_cv and solve_banded_ridge_cv have 15+ parameters with no `*` marker, so callers can pass a long positional tail (n_iter, concentration, alphas, cv, local_alpha, ...) positionally — fragile and inconsistent with the project's keyword-only convention for 3+ kwargs and with the sibling function. v0.6.0 is the moment to enforce it.  
*Evidence:* `def solve_ridge_cv(X, Y, alphas=..., cv=5, local_alpha=True, n_targets_batch=None, ...):  # no `*``  
*Fix (trivial):* Insert `*` after the primary data args (after X, Y / Xs, Y) in solve_ridge_cv and solve_banded_ridge_cv to make the options keyword-only, matching cross_val_predict_ridge.

#### ✅ ~~F053 · MEDIUM~~ — implemented (`*` marker added to bootstrap/threshold/ttest/regions/standardize/plot; icc removed)
`nltools/data/braindata/__init__.py:948` — **Public methods with 3+ kwargs missing the required keyword-only '*' marker**  
CLAUDE.md requires a keyword-only '*' marker in any public method with 3+ kwargs. Several BrainData facade methods with well over three kwargs are fully positional: icc (line 948), bootstrap (546), threshold (1740), ttest (1839), regions (1540), standardize (1669), plot (1050). This lets fragile positional calls like bd.threshold(3, None, True) through and is inconsistent with align/fit/predict/mean/std which correctly use '*'. v0.6.0 is a breaking release, so this is the moment to add the markers.  
```python
def icc(self, n_subjects, n_sessions, method="icc2", parallel=None, n_jobs=-1, max_gpu_memory_gb=4.0):
    def bootstrap(self, stat, n_samples=5000, save_boots=False, percentiles=(2.5, 97.5), X_test=None, backend=None, ...):
    def threshold(self, upper=None, lower=None, binarize=False, coerce_nan=True, cluster_threshold=0):
```
*Fix (small):* Insert '*' after the primary/positional data args (e.g. def icc(self, n_subjects, n_sessions, *, method=..., ...); def threshold(self, *, upper=None, ...)). Apply consistently across the listed methods.

#### ✅ ~~F060 · MEDIUM~~ — implemented (`*` marker added to plot/plot_flatmap/resample_to)
`nltools/data/braindata/__init__.py:1050` — **Public plot()/plot_flatmap()/resample_to() facades lack the required keyword-only `*` marker**  
Project convention requires a keyword-only `*` marker in any public method with 3+ kwargs. plot() (15 kwargs), plot_flatmap() (16 kwargs), and resample_to() (3 args) accept all of them positionally. iplot() and plot_surf() correctly use `*`, so this is an inconsistency within the same facade. These wrap the in-scope plotting.py/io.py functions (plot_brain, plot_flatmap_brain, resample_to).  
*Evidence:* `def plot(self, method="glass", upper=None, lower=None, threshold=None, view="z", cut_coords=None, cmap=None, bg_img=None, ax=None, figsize=(8, 6), title=None, colorbar=True, save=None, stat="mean", limit=3, **kwargs):`  
*Fix (small):* Insert `*,` after `self` (and after the primary data arg for the module functions) so all optional kwargs are keyword-only, matching iplot/plot_surf.

#### ✅ ~~F034 · MEDIUM~~ — implemented (`metric`→`method` on cluster_summary facade + stats fn)
`nltools/data/adjacency/__init__.py:443` — **cluster_summary() uses reserved kwarg `metric` for a mean/median aggregation choice**  
Per the v0.6.0 convention, `metric` is reserved for a distance/similarity metric and is kept separate from `method` (the algorithm/variant choice). Here `metric='mean'|'median'|None` selects an aggregation, i.e. a variant choice, so it should be named `method`. Same misuse in the pure function `cluster_summary` in stats.py (line 484).  
*Evidence:* `def cluster_summary(self, clusters=None, metric="mean", summary="within"):`  
*Fix (small):* Rename `metric` -> `method` on both the facade and the stats.py function (v0.6.0 is a breaking release). `summary` ('within'|'between') is also a variant selector but is at least not colliding with a reserved name.

#### ✅ ~~F035 · MEDIUM~~ — implemented (reserved `metric` bool→`metric_mds` on plot_mds facade + plotting fn; added `*`)
`nltools/data/adjacency/__init__.py:595` — **plot_mds() uses reserved kwarg `metric` as a boolean metric/non-metric flag**  
`metric` is reserved for a distance/similarity metric string. In plot_mds it is a bool selecting metric vs non-metric MDS (`metric=True`), which collides confusingly with the reserved meaning used elsewhere (e.g. `distance(metric='correlation')`). Same in plotting.py `plot_mds` (line 58).  
*Evidence:* `def plot_mds(self, n_components=2, metric=True, labels=None, ...):`  
*Fix (small):* Rename to a non-reserved boolean name such as `metric_mds: bool` or route through `method='metric'|'nonmetric'`. Update plotting.py accordingly.

#### ✅ ~~F036 · MEDIUM~~ — implemented (`*` marker on Adjacency.similarity (moved up), plot_mds, plot_silhouette; ttest/threshold/cluster_summary already done)
`nltools/data/adjacency/__init__.py:670` — **Public methods with 3+ kwargs lack the required keyword-only `*` marker**  
The convention requires a `*` marker in any public method with 3+ kwargs. Several facade methods place all kwargs as positional-or-keyword: `similarity` (only `project` is keyword-only; plot/permutation_method/n_permute/metric/... are positional), `ttest` (permutation/n_permute/tail/return_null/n_jobs/random_state — no `*`), `threshold` (upper/lower/binarize — no `*`), `cluster_summary` (clusters/metric/summary — no `*`), `plot_mds`, and `plot_silhouette`. This lets callers pass these by position, which the convention explicitly forbids and which makes the API fragile to reordering.  
*Evidence:* `def ttest(self, permutation=False, n_permute=5000, tail=2, return_null=False, n_jobs=-1, random_state=None):  # no `*``  
*Fix (small):* Insert `*` after the primary data/positional arg in these signatures (and mirror in the stats/modeling functions where they are public-facing), forcing keyword use.

#### ✅ ~~F072 · MEDIUM~~ — implemented (isc_test `n_permute`→`n_samples`; dropped unused `permutation_method`)
`nltools/data/collection/inference.py:499-512` — **isc_test is a bootstrap but uses n_permute (should be n_samples) and carries an unused permutation_method**  
Per the API table, bootstrap sample count is `n_samples`, distinct from `n_permute` (permutation count). isc_test resamples subjects with replacement (a bootstrap) yet names the count `n_permute`. It also declares `permutation_method: str = 'bootstrap'` which is never read in the body — a dead kwarg that implies a choice the code doesn't offer.  
*Evidence:* `def isc_test(bc, *, method='loo', ..., n_permute=5000, permutation_method='bootstrap', ...):  # permutation_method never used`  
*Fix (small):* Rename `n_permute`→`n_samples` for the bootstrap, and either implement `permutation_method` (bootstrap vs sign-flip) or drop it. Fix the facade (core.py isc_test) to match.

#### ✅ ~~F073 · MEDIUM~~ — fully implemented (2026-07-17 cache= session). `scheme`→`spatial_scale` + `progress_bar` done earlier; `cache=` NOW WIRED — align persists its (joint-op) output collection to disk when caching, via a new `execution._persist_or_keep` helper mirroring `_apply`'s `'auto'` rule (cache iff source is path-backed). Implementing it surfaced TWO real bugs the skipped align tests had hidden: (1) **align fed `LocalAlignment` transposed data** — `BrainData.data` is `(n_samples, n_voxels)` but the aligner wants `(n_voxels, n_samples)`; crashed when `n_samples < n_voxels` (why every align behavior test was `@pytest.mark.skip` with a misdiagnosed "needs >8 timepoints" reason) and was silently-wrong otherwise. Fixed with `.T` on fit input + output rewrap. (2) **`to_nifti` inherited the mask's int8 dtype**, scale-quantizing all float writes (stat maps, betas, the whole collection disk cache) to ~1 LSB — fixed by pinning the on-disk dtype to the data's own, so the cache (and every user `bd.write`) is now lossless. Align behavior tests un-skipped; +io precision regression test.
`nltools/data/collection/inference.py:554-567` — **align exposes scheme= (banned) and silently ignores cache=/progress_bar=**  
The facade `align` takes `scheme='searchlight'`, but the convention bans `scheme` as an algorithm/variant name, and the searchlight-vs-parcellation axis is exactly a `spatial_scale` concept (whole_brain|roi|searchlight). The convention says the facade should translate the algorithm-layer legacy name (LocalAlignment.scheme) rather than surface it. Additionally `cache` and `progress_bar` are accepted but never referenced in the body (grep confirms), so they are inert.  
*Evidence:* `def align(bc, *, method='procrustes', scheme='searchlight', ..., progress_bar=False, cache='auto'):  # body never uses cache/progress_bar`  
*Fix (small):* Rename `scheme`→`spatial_scale` at the facade and translate to LocalAlignment.scheme internally; remove or implement cache/progress_bar.

#### ✅ ~~F071 · MEDIUM~~ — implemented (`algorithm`→`method` on predict; internal `_get_model` keeps `algorithm`, allowed)
`nltools/data/collection/pipeline.py:139-166` — **BrainCollectionPipeline.predict uses algorithm= instead of the canonical method=**  
The v0.6.0 convention mandates `method` for algorithm/variant choice and explicitly bans `algorithm`. This pipeline is user-reachable via `bc.cv(...).predict(y, algorithm='ridge')`, so it exposes a banned kwarg on the public surface. `_get_model(algorithm, ...)` propagates the same name. The rest of the facade (fit/predict) already standardized on `model=`/`method=`.  
*Evidence:* `def predict(self, y, algorithm: str = "ridge", **kwargs):`  
*Fix (small):* Rename to `model=` (to match BrainCollection.predict/fit) or `method=`; keep the internal `_get_model` name consistent with the choice.

#### ✅ ~~F079 · MEDIUM~~ — implemented (`verbose`→`progress_bar` on append/clean facades + append.py/diagnostics.py; clean default flipped True→False)
`nltools/data/designmatrix/__init__.py:458` — **`verbose` used as a print-gate in append()/clean() violates the reserved-name convention**  
CLAUDE.md states verbose is 'reserved for log-level only' and that any flag toggling status/progress output must be `progress_bar: bool = False`. Both `append(verbose=False)` (prints 'Separating columns across runs: ...') and `clean(verbose=True)` (prints dropped-column messages) use `verbose` purely to gate print() calls, which is exactly the banned usage.  
*Evidence:* `append(..., verbose: bool = False)  ->  print(f"Separating columns across runs: {sorted(cols_to_sep)}")   /   clean(..., verbose: bool = True) -> print(f"{col_i} and {col_j} correlated ...")`  
*Fix (small):* Rename to `progress_bar: bool = False` on both facades (and the underlying append.py / diagnostics.py functions), or route these through the logging layer if they are meant to be log-level. Note clean() also defaults to True (prints by default), unlike the convention's `False` default.

#### ✅ ~~F078 · MEDIUM~~ — implemented (`*` marker added to `clean()`)
`nltools/data/designmatrix/__init__.py:494-500` — **clean() has 4 kwargs but is missing the required keyword-only `*` marker**  
CLAUDE.md requires a keyword-only `*` marker in any public method with 3+ kwargs. `DesignMatrix.clean` exposes four (fill_na, exclude_confounds, thresh, verbose) with no `*`, so all four are callable positionally — e.g. `dm.clean(0, True, 0.9)` — which the convention forbids and which every other 3+-kwarg method here (plot, append, corr) obeys.  
*Evidence:* `def clean(self, fill_na=..., exclude_confounds=False, thresh=0.95, verbose=True) -> DesignMatrix:  # no `*``  
*Fix (trivial):* Insert `*` after the primary/first argument: `def clean(self, *, fill_na=0, exclude_confounds=False, thresh=0.95, verbose=True)` (and update the internal diagnostics.clean call site accordingly).

#### ✅ ~~F080 · MEDIUM~~ — implemented (explicit `method=` on downsample/upsample facade + transforms; `**kwargs` dropped)
`nltools/data/designmatrix/__init__.py:549` — **downsample()/upsample() forward `**kwargs` across an internal nltools->nltools boundary**  
CLAUDE.md permits `**kwargs` ONLY when forwarding to an external third-party API; internal delegation must use explicit signatures. The `downsample`/`upsample` facades take `**kwargs` and forward to nltools' own transforms.downsample/upsample. Worse, transforms.downsample has signature `downsample(dm, target, **kwargs)` and recovers `method` via `kwargs.pop('method','mean')` instead of an explicit `method` param — inconsistent with transforms.upsample which declares `method` explicitly.  
*Evidence:* `def downsample(self, target, method='mean', **kwargs) -> ...  # facade\ndef downsample(dm, target, **kwargs):  method = kwargs.pop('method', 'mean')  # transforms.py:100,123`  
*Fix (small):* Give both facade and internal functions explicit signatures (`method: str = 'mean'`/`'linear'`) and drop `**kwargs`. Add explicit params for any real option rather than passing an untyped bag between internal modules.

#### ✅ ~~F097 · MEDIUM~~ — DECISION (Eshin): implemented — Roc `plot_method`→`method` and `threshold_type`→`method`
`nltools/data/roc/__init__.py:233-240` — **plot(plot_method=...) and threshold_type= should be the canonical `method=` per v0.6.0**  
The v0.6.0 table mandates `method` for algorithm/variant choice (not scheme/kind/mode/type). `Roc.plot(plot_method='gaussian'|'observed')` is a variant selector that should be `method=`, and `threshold_type` in __init__/calculate is likewise a variant selector. Roc isn't one of the four core facades, but it's public API in the same release.  
*Evidence:* `def plot(self, plot_method="gaussian", balanced_acc=False, **kwargs):`  
*Fix (small):* Rename `plot_method` -> `method`; consider `threshold_type` -> `method` (or document the intentional exception). Keep keyword-only markers where 3+ kwargs.

#### ✅ ~~F096 · MEDIUM~~ — implemented (dropped unused `**kwargs` from Roc.__init__/plot; removed phantom `**kwargs` from calculate docstring)
`nltools/data/roc/__init__.py:35-42` — **Roc.__init__/calculate/plot accept **kwargs that are never used or forwarded**  
CLAUDE.md permits **kwargs only when forwarding to an external third-party API. Roc.__init__ accepts **kwargs but never uses it; plot() accepts **kwargs but never passes it to plot_roc; calculate()'s docstring documents **kwargs though the signature has none. These are dead/ignored kwargs that silently swallow typos.  
*Evidence:* `def __init__(self, input_values=None, binary_outcome=None, threshold_type="optimal_overall", forced_choice=None, **kwargs,):  # kwargs never referenced`  
*Fix (small):* Drop the unused **kwargs from __init__/plot (or forward plot's kwargs to plot_roc explicitly), and remove the phantom **kwargs from calculate's docstring.

#### ✅ ~~F105 · MEDIUM~~ — DECISION (Eshin): keep `n_iter` as a documented exception. It is the familiar sklearn RandomizedSearchCV name and no canonical name exists for a random-search count; renaming would invent vocabulary. To be documented in CLAUDE.md/migration guide.
`nltools/models/ridge.py:39` — **Ridge exposes 'n_iter', an explicitly banned kwarg alias, on a facade-forwarded path**  
The random-search iteration count is named n_iter. The v0.6.0 API table explicitly bans n_iter as an alias. Because BrainData.fit(model='ridge', ...) forwards **kwargs straight into Ridge(**ridge_kwargs) (see data/braindata/modeling.py:211), n_iter is a user-facing kwarg, not just an internal one, so the facade-translation escape hatch does not apply. There is no canonical name for random-search count, but the raw banned token should not surface.  
*Evidence:* `n_iter=100,  # __init__  ...  Ridge(alpha="auto", cv=3, n_iter=5, ...) in tests`  
*Fix (small):* Rename to an explicit non-banned name (e.g. n_search_iter or n_random_search) on the Ridge facade and translate to the solver's n_iter at the boundary; keep solvers.solve_banded_ridge_cv's internal n_iter as-is.

#### ✅ ~~F106 · MEDIUM~~ — implemented (`*` marker added to Ridge.__init__ and Glm.__init__)
`nltools/models/ridge.py:87` — **Ridge.__init__ and Glm.__init__ lack the required keyword-only '*' marker**  
Project rule: '*' marker required in __init__ after the primary data arg. Ridge.__init__ has 10 positional-or-keyword hyperparameters and Glm.__init__ has 5, none keyword-only. All of these are configuration params meant to be passed by name; allowing positional binding (e.g. Ridge(1.0, 5, [0.1,1.0], 100, ...)) is fragile across the breaking release where reordering is expected.  
*Evidence:* `def __init__(self, alpha=1.0, cv=None, alphas=None, n_iter=100, concentration=[0.1, 1.0], backend="numpy", ...)`  
*Fix (trivial):* Insert '*' after the first arg in both __init__ signatures (Ridge: after alpha or at the front; Glm: after t_r) to make all hyperparameters keyword-only.

#### ✅ ~~F117 · MEDIUM~~ — moot (AlignStep class removed entirely)
`nltools/pipelines/steps.py:465-534` — **AlignStep.__init__ uses **kwargs to forward to INTERNAL nltools algorithms (SRM/HyperAlignment)**  
Per CLAUDE.md, **kwargs is permitted only when forwarding to a third-party API; internal nltools->nltools delegation must use explicit signatures. AlignStep takes **kwargs and forwards a filtered subset to nltools.algorithms.SRM (`rand_seed`) and HyperAlignment (`auto_pad`) — both internal. It also silently drops any other kwarg (the dict comprehension keeps only whitelisted keys), so a typo'd or unsupported option vanishes with no error (ux hazard).  
*Evidence:* `model = SRM(n_iter=..., features=..., **{k: v for k, v in self.kwargs.items() if k in ["rand_seed"]})`  
*Fix (small):* Replace **kwargs with explicit params (e.g. rand_seed: int | None = None, auto_pad: bool = ...). Raise on unknown options instead of dropping them.

#### ✅ ~~F128 · MEDIUM~~ — implemented (plotting.plot_silhouette: explicit `colors`/`figsize` params + `*`, dropped `**kwargs`)
`nltools/plotting/adjacency.py:327-328` — **plot_silhouette pulls colors/figsize out of **kwargs instead of explicit params**  
The **kwargs rule permits **kwargs only when forwarding to an external third-party API. Here kwargs is never forwarded; it is mined with kwargs.get('colors') and kwargs.get('figsize'). These are genuine internal parameters and should be explicit keyword-only args. As written, any other kwarg (a typo, or a legit seaborn/matplotlib option) is silently swallowed and ignored, which is a confusing UX.  
```python
colors = kwargs.get("colors", sns.color_palette("hls", n_clusters))
    figsize = kwargs.get("figsize", (6, 4))
```
*Fix (small):* Make the signature explicit: `def plot_silhouette(distance, labels, *, ax=None, permutation_test=True, n_permute=5000, colors=None, figsize=(6, 4)):` and drop **kwargs (or forward it to a real third-party call).

#### ✅ ~~F141 · MEDIUM~~ — accepted (stats.align keeps `*args/**kwargs` forwarding to SRM/DetSRM with a `# nosemgrep` reason; deliberate boundary, not changed)
`nltools/stats/alignment.py:18` — **align() forwards to internal SRM/DetSRM via *args/**kwargs**  
The **kwargs rule permits pass-through only to external third-party APIs. Here `*args, **kwargs` are forwarded to nltools' own DetSRM/SRM classes (internal->internal delegation), which the rules require to use explicit signatures. The positional `*args` splat into `DetSRM(features=n_features, *args, **kwargs)` is also fragile.  
```python
def align(data, method="deterministic_srm", n_features=None, axis=0, *args, **kwargs):
    ...
    srm = DetSRM(features=n_features, *args, **kwargs)
```
*Fix (small):* Replace *args/**kwargs with the explicit SRM hyperparameters align actually supports (e.g. n_iter, etc.) and forward them by name.

#### ✅ ~~F140 · MEDIUM~~ — **RESOLVED by ICC strip (ledger #3732): compute_icc removed entirely**
`nltools/stats/correlation.py:279` — **compute_icc uses `icc_type=` (banned name)**  
`icc_type` is explicitly listed among the banned names for algorithm/variant choice (canonical = `method`). compute_icc is public (in nltools.stats.__all__) and called directly in tests, so users hit `icc_type=` directly. Facade translation is allowed for internal algorithm-layer APIs, but this is a re-exported public function.  
*Evidence:* `def compute_icc(Y, icc_type="icc2"):`  
*Fix (small):* Rename the kwarg to `method` (values 'icc1'|'icc2'|'icc3'). If backward-compat is needed, accept both and warn, but the canonical public name should be `method`.

#### ✅ ~~F142 · MEDIUM~~ — implemented (public permutation wrappers now use `device=`, translate to engine `parallel=`)
`nltools/stats/permutation.py:42` — **Public permutation API exposes banned `parallel=` kwarg alongside n_jobs**  
Every public permutation wrapper (one_sample/two_sample/correlation/timeseries/matrix) exposes both `parallel: str | None = 'cpu'` (values 'cpu'/'gpu'/None) and `n_jobs`. The v0.6.0 table bans `parallel=` (parallel execution = `n_jobs`) and reserves device/GPU selection to `device` on BrainCollection. These are user-facing, re-exported functions, not hidden internals, so the `parallel=` name leaks into the public surface.  
*Evidence:* `def one_sample_permutation_test(..., parallel: str | None = "cpu", n_jobs: int = -1, ...)`  
*Fix (medium):* Decide the canonical public spelling: fold backend selection into a `device`-style kwarg or drop `parallel` from the public wrappers and translate to the engine's `parallel=` at the boundary, keeping only `n_jobs` public.

#### ✅ ~~F180 · MEDIUM~~ — implemented (braindata bootstrap/standardize/etc already had `*`; Adjacency.similarity `*` moved up so all options are keyword-only)
`nltools/data/braindata/__init__.py:546-604` — **Keyword-only `*` marker missing on several public methods with 3+ kwargs (systemic)**  
The v0.6.0 rule requires a `*` marker in any public method with 3+ kwargs. It is applied on some methods (align, distance, mean, cv) but omitted on many others, so callers can pass core options positionally and the convention is only half-enforced. Confirmed offenders: BrainData.bootstrap (8 kwargs, no `*`), BrainData.icc (948), BrainData.standardize (1669), BrainData.resample_to (1599), BrainData.temporal_resample (1723), BrainData.filter (782), Adjacency.ttest (915), and Adjacency.similarity (670) which places `*` only before the final `project` param leaving 10 params positional.  
*Evidence:* `def bootstrap(self, stat, n_samples=5000, save_boots=False, percentiles=(2.5,97.5), X_test=None, backend=None, max_gpu_memory_gb=4.0, n_jobs=-1, random_state=None):  # no `*``  
*Fix (small):* Add `*` after the primary data/positional arg on each listed method so all options are keyword-only, matching align/distance.

#### ✅ ~~F174 · MEDIUM~~ — implemented (`scheme`→`spatial_scale` on BrainCollection.align, translated at boundary)
`nltools/data/collection/__init__.py:1062-1066` — **BrainCollection.align uses `scheme=` for spatial scope instead of the canonical `spatial_scale`**  
align accepts `scheme: str = "searchlight"` selecting the spatial scope (searchlight vs parcellation). The v0.6.0 convention reserves `spatial_scale` ('whole_brain'|'roi'|'searchlight') for spatial scope and forbids `scheme` as a public kwarg name (the internal algorithm layer may keep `scheme`, but the facade must translate). This is on the primary facade, so users hit it directly. (The undocumented state of the method compounds it.)  
*Evidence:* `def align(self, *, method: str = "procrustes", scheme: str = "searchlight", radius_mm: float = 10.0, ...)`  
*Fix (small):* Rename the facade kwarg to `spatial_scale` and translate to the algorithm layer's `scheme` at the boundary; document the accepted values.

#### ✅ ~~F195 · MEDIUM~~ — DECISION (Eshin): implemented — Roc.calculate/plot `threshold_type`/`plot_method`→`method`; `*` markers present
`nltools/data/roc/__init__.py:63` — **Roc.calculate/plot violate method-naming and keyword-only conventions**  
calculate() takes 6 keyword args with no keyword-only '*' marker (conventions require '*' for any public method with 3+ kwargs), and uses threshold_type= for a variant choice (should be method). Roc.plot uses plot_method= (should be method). This is a public data-class facade, exactly where the canonical rules apply.  
```python
def calculate(self, input_values=None, binary_outcome=None, criterion_values=None, threshold_type="optimal_overall", forced_choice=None, balanced_acc=False):
...
def plot(self, plot_method="gaussian", balanced_acc=False, **kwargs):
```
*Fix (small):* Add '*' after the primary data arg, rename threshold_type/plot_method to method (translating at the boundary), and add tests pinning the new signatures.

#### ✅ ~~F194 · MEDIUM~~ — **RESOLVED by ICC strip (ledger #3732): compute_icc removed entirely**
`nltools/stats/correlation.py:279` — **compute_icc uses banned kwarg name icc_type instead of canonical method**  
The v0.6.0 conventions explicitly list icc_type among the non-canonical names that should be method for algorithm/variant choice. compute_icc is public (re-exported by nltools.stats), not merely an internal-facade-translated name, so users call it directly with the deprecated spelling. Surfaced while auditing its (otherwise good, 52-ref) test coverage.  
*Evidence:* `def compute_icc(Y, icc_type="icc2"):`  
*Fix (small):* Rename to method= at the public boundary (keep icc_type as a translated alias if back-compat matters) and update the ~52 test references plus BrainData.icc facade translation.

### Low severity

#### ✅ ~~F019 · LOW~~ — accepted (algorithm-layer inference fns keep `parallel=`; the finding itself deems this acceptable for the internal layer)
`nltools/algorithms/inference/correlation.py:620-631` — **Public inference functions expose `parallel=` plus `n_jobs`, splitting device/parallel selection unlike the canonical scheme**  
Every top-level test exported in __init__.__all__ (correlation/one_sample/two_sample/timeseries/isc) takes `parallel: str|None = 'cpu'` (None/cpu/gpu) alongside `n_jobs`. The v0.6.0 convention reserves `parallel=` against and models device selection as `device='cpu'` + `n_jobs`. These are algorithm-layer functions and the data-class facades are allowed to translate, so this is not a facade violation -- but because they are also re-exported as the module's public API, direct callers see the non-canonical `parallel=` name and a device/parallelism split that differs from the rest of the library.  
*Evidence:* `def correlation_permutation_test(..., parallel: str | None = "cpu", n_jobs: int = -1, ...)`  
*Fix (medium):* Acceptable as internal algorithm layer, but consider documenting that `parallel=` is legacy/internal, or aligning the exported surface with `device=`/`n_jobs` so direct users get the canonical vocabulary.

#### ✅ ~~F028 · LOW~~ — implemented (ridge solvers now return the backend under the `backend` key, matching ridge_cv; consumers updated)
`nltools/algorithms/ridge/core.py:345` — **ridge_cv returns key 'backend' while the solvers return 'parallel' for the same value**  
core.ridge_cv returns the backend name under result['backend'], while solve_ridge_cv/solve_banded_ridge_cv/cross_val_predict_ridge return the same information under result['parallel']. Two names for one concept across the same package's result dicts is an avoidable inconsistency for downstream code. Relatedly, ridge_svd defaults parallel=None while ridge_cv and the solvers default parallel='cpu' (functionally equivalent but visibly inconsistent).  
```python
result = { ..., "backend": backend.name }  # core.py
result = { ..., "parallel": backend.name }  # solvers.py
```
*Fix (trivial):* Standardize on one key (e.g. 'backend') across all ridge result dicts, and align the parallel default (None vs 'cpu') between ridge_svd and the others.

#### ✅ ~~F056 · LOW~~ — implemented (standardize `verbose`→`suppress_warnings`, default False, logic inverted; facade + analysis fn)
`nltools/data/braindata/__init__.py:1669` — **standardize() uses verbose= to control warning suppression, but verbose is reserved for log-level**  
Per the v0.6.0 kwarg table, verbose is reserved for log-level only; here verbose=True/False toggles suppression of sklearn near-zero-variance warnings — a distinct concern. It also reads backwards (verbose=True means 'show warnings'). This overloads a reserved name.  
*Evidence:* `def standardize(self, axis=0, method="center", verbose=True):`  
*Fix (trivial):* Rename to a purpose-specific flag (e.g. suppress_warnings: bool = False) or fold into the standard logging/verbose semantics rather than warning gating.

#### ✅ ~~F041 · LOW~~ — implemented (Adjacency.plot_silhouette facade + stats fn now pass explicit `colors`/`figsize`; no more internal `**kwargs`)
`nltools/data/adjacency/__init__.py:640` — **plot_silhouette forwards **kwargs into an internal nltools function**  
The **kwargs rule permits **kwargs only when forwarding to a third-party API. Here `Adjacency.plot_silhouette` -> stats.plot_silhouette -> `nltools.plotting.plot_silhouette` is nltools->nltools internal delegation carrying **kwargs (stats.py lines 447-481), which the convention says must use explicit signatures. The kwargs may ultimately reach matplotlib, but the immediate delegation target is internal.  
*Evidence:* `def plot_silhouette(self, labels=None, ax=None, permutation_test=True, n_permute=5000, **kwargs):`  
*Fix (small):* Give the internal plot_silhouette an explicit signature for the parameters it actually consumes; reserve **kwargs for the final matplotlib/seaborn boundary only.

#### ✅ ~~F040 · LOW~~ — DECISION (Eshin): implemented — Adjacency.similarity `permutation_method`→`method` (facade + stats fn); `metric` stays the correlation type
`nltools/data/adjacency/__init__.py:674` — **similarity() `permutation_method` is close to the forbidden `perm_type`; convention prefers `method`**  
The canonical table forbids `perm_type` in favor of `method` for algorithm/variant choice. `permutation_method='1d'|'2d'|None` is exactly such a variant selector. It's a judgment call since `metric` already occupies the distance slot, but the current name sits in the forbidden family. Worth a deliberate decision before the breaking release rather than leaving an inconsistent near-miss.  
*Evidence:* `def similarity(self, data, plot=False, permutation_method="2d", n_permute=5000, metric="spearman", ...):`  
*Fix (small):* Either rename to `method` (documenting that `metric` = correlation type and `method` = permutation scheme), or explicitly accept `permutation_method` as an intentional exception noted in the migration guide.

#### ✅ ~~F077 · LOW~~ — implemented (`contrast_type`→`statistic`; banned `*_type` gone, deliberate output-selector name)
`nltools/data/collection/core.py:605-618` — **compute_contrasts uses contrast_type=, matching the banned *_type variant-naming pattern**  
The convention standardizes variant/statistic choice on `method` and explicitly bans `*_type` names (icc_type, extract_type, perm_type). `contrast_type` ('beta'|'t'|'z'|'p'|'se'|'all') is the same anti-pattern. Debatable since it selects an output statistic rather than an algorithm, but it reads inconsistently against the rest of the facade.  
*Evidence:* `def compute_contrasts(self, contrasts, *, contrast_type: str = "beta", ...):`  
*Fix (small):* Consider renaming to `method=` (or documenting an explicit exception for statistic-selection kwargs) for cross-facade consistency.

#### ✅ ~~F129 · LOW~~ — implemented (`*` marker added to plotting.plot_flatmap, matching plot_surf)
`nltools/plotting/brain.py:432` — **plot_flatmap lacks the keyword-only '*' marker that its sibling plot_surf has**  
plot_surf uses `def plot_surf(brain, *, ...)` per the convention (keyword-only after the primary data arg for methods/functions with 3+ kwargs). plot_flatmap has ~16 kwargs but no '*' marker, so all of them are positionally callable — inconsistent with plot_surf and with the stated convention.  
```python
def plot_flatmap(
    brain,
    threshold=None,
    cmap="RdBu_r",
    ...
```
*Fix (trivial):* Add `*,` after `brain` to make the options keyword-only, matching plot_surf.

#### ✅ ~~F148 · LOW~~ — DECISION (Eshin): implemented — compute_similarity `method`→`metric` + BrainData.similarity facade + analysis fn (metric-vocabulary compliance)
`nltools/stats/correlation.py:112` — **compute_similarity uses `method=` for what is a similarity metric**  
The canonical table keeps distance/similarity metric under `metric`, separate from `method` (algorithm choice). compute_similarity's `method` values ('correlation','spearman','cosine','dot_product') are all similarity metrics. The BrainData.similarity facade mirrors this with `method=`, so both layers diverge from the table. Noting for consideration since v0.6.0 is the window to canonicalize; low because it is at least internally consistent.  
*Evidence:* `def compute_similarity(data1, data2, method="correlation"):`  
*Fix (small):* Consider renaming to `metric=` here and at the facade for table compliance; if kept, document the deliberate exception.

#### ✅ ~~F146 · LOW~~ — DECISION (Eshin): implemented — isc `metric`→`summary` (stat) and `sim_metric`→`metric` (distance); trailing kwarg order fixed; translated to inference layer at boundary
`nltools/stats/intersubject.py:102` — **isc() inverts metric vocabulary and violates trailing kwarg order**  
`metric` is used for the summary statistic ('mean'/'median') while the actual pairwise distance metric is a separate `sim_metric` kwarg — inverting the canonical table where `metric` denotes the distance/similarity metric. Additionally `sim_metric` (a domain kwarg) is placed after `random_state`, violating the canonical trailing order (domain kwargs, then n_jobs, random_state, progress_bar). isc also hardcodes progress_bar=False rather than exposing it.  
*Evidence:* `def isc(data, n_samples=5000, metric="median", method="bootstrap", ..., n_jobs=-1, random_state=None, sim_metric="correlation"):`  
*Fix (small):* Rename the summary-stat kwarg (e.g. `summary='median'`) and use `metric` for the distance metric, or at minimum move `sim_metric` ahead of n_jobs/random_state and reconcile with the canonical vocabulary.

#### ✅ ~~F184 · LOW~~ — implemented (same fix as F056: standardize `verbose`→`suppress_warnings`; `*` already present)
`nltools/data/braindata/__init__.py:1669` — **standardize() uses `verbose=` for warning suppression, which the convention reserves for log-level**  
The v0.6.0 table states `verbose` is reserved for log-level only (progress/noise flags should use `progress_bar`). Here `verbose=True` toggles sklearn numerical-warning suppression in standardize (facade at 1669 and analysis.standardize at 844) — a noise flag, not a log level. Minor naming drift for a breaking release.  
*Evidence:* `def standardize(self, axis=0, method="center", verbose=True):  # verbose gates warnings.catch_warnings suppression`  
*Fix (small):* Rename to a purpose-specific flag (e.g. `suppress_warnings: bool` or fold into the standard noise convention) and add the `*` marker.

#### ✅ ~~F186 · LOW~~ — implemented (BrainData.distance `**kwargs` documented as scipy-cdist passthrough + `# nosemgrep`, matching backing analysis.distance)
`nltools/data/braindata/__init__.py:715-757` — **BrainData.distance forwards `**kwargs` across an internal nltools->nltools boundary**  
The `**kwargs` rule permits **kwargs only when forwarding to a third-party API. distance() forwards `**kwargs` to internal `analysis.distance`, which itself forwards to scipy cdist. The terminal is third-party so this is benign in practice, but it violates the letter of the rule (facade->internal hop should use an explicit signature) and lets arbitrary kwargs pass silently through two nltools layers.  
*Evidence:* `def distance(self, metric="euclidean", *, spatial_scale=..., roi_mask=None, radius_mm=10.0, **kwargs): return distance(self, ..., **kwargs)`  
*Fix (trivial):* Either document that **kwargs is a scipy-cdist passthrough (acceptable), or name the handful of cdist options explicitly at the facade boundary.

#### ✅ ~~F175 · LOW~~ — DECISION (Eshin): keep `model=` on BrainData.fit as a documented exception. It selects an estimator CLASS (Glm vs Ridge), reads naturally, and the v0.6 migration guide already documents `.fit(model='glm')`. To be documented in CLAUDE.md.
`nltools/data/braindata/__init__.py:843-845` — **BrainData.fit selects the algorithm via `model=` (positional) rather than the canonical `method=`**  
fit(self, model="glm", *, ...) chooses the estimator variant ('glm' vs 'ridge') through `model`, but the v0.6.0 convention says algorithm/variant choice is `method`. The primary data arg is not present here, so the algorithm selector is also positional rather than after the keyword-only marker. Borderline (one could argue `model` is its own concept), but it is inconsistent with `isc(method=...)`, `align(method=...)`, `distance(method=...)` elsewhere on the facades.  
```python
def fit(
        self,
        model="glm",
        *,
        X=None,
```
*Fix (small):* Consider renaming to `method=` for cross-facade consistency (with a deprecation alias), or explicitly document why `model` is intentionally distinct.

#### ✅ ~~F197 · LOW~~ — implemented (`mode`→`method` in `regress()`; only 'ols' supported)
`nltools/stats/regression.py:18` — **regress() uses mode= for algorithm choice and mode is vestigial (only 'ols')**  
Conventions list mode among banned variant-selector names (canonical: method). Here mode only accepts 'ols' and raises NotImplementedError otherwise, so it is both non-canonical and effectively dead — it selects nothing. The companion stats= param is also an unusual selector name.  
```python
def regress(X, Y, mode: str = "ols", stats: str = "full"):
    if mode != "ols":
        raise NotImplementedError(...)
```
*Fix (trivial):* Either drop mode entirely (only OLS is supported) or rename to method for consistency; whichever is chosen, pin it in the new test_regression.py.

## Documentation & docstrings (44)

_Auto-generated by griffe2md → these render as the shipped API reference._

### High severity

#### ✅ ~~F091 · HIGH~~
`nltools/data/fitresults/__init__.py:282-352` — **Predict docstring documents dispatch as method='whole_brain'/'roi'/'searchlight' but the API uses spatial_scale**  
BrainData.predict (nltools/data/braindata/prediction.py) takes `spatial_scale: str = 'whole_brain'` with values whole_brain|roi|searchlight — exactly the v0.6.0 canonical `spatial_scale` kwarg. The Predict class docstring instead labels every dispatch section `method='whole_brain'`, `method='roi'`, `method='searchlight'`, which contradicts the shipped API and the CLAUDE.md convention (method = algorithm choice, distinct from spatial_scale). Users copying these will pass the wrong kwarg.  
*Evidence:* `based on the dispatch path (``method``, ``y`` vs ``X``, ``refit``) ... **method='whole_brain'** (with ``y``): ... **method='roi'** ... **method='searchlight'**`  
*Fix (trivial):* Replace `method=` with `spatial_scale=` throughout the Predict docstring dispatch descriptions.

#### ✅ ~~F092 · HIGH~~
`nltools/data/simulator/__init__.py:47-52` — **Simulator.create_data example uses y=/n_reps= kwargs that don't exist in the signature**  
The class docstring example calls `sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)`, but create_data's signature is `create_data(self, levels, sigma, radius=5, center=None, reps=1, output_dir=None)`. There is no `y` or `n_reps` parameter — the example raises TypeError as written. Stale docstring.  
*Evidence:* `>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)`  
*Fix (trivial):* Update the example to `sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)`.

#### ✅ ~~F093 · HIGH~~
`nltools/data/simulator/__init__.py:534-539` — **SimulateGrid example calls fit(n_permute=1000) but fit() takes no arguments**  
The class docstring example `sim.fit(n_permute=1000)` does not match `def fit(self)` — fit accepts no parameters and only runs a t-test. The example raises TypeError. This also signals the never-wired permutation path (see dead-code finding on _run_permutation).  
*Evidence:* `>>> sim.fit(n_permute=1000)`  
*Fix (trivial):* Fix the example to `sim.fit()`, or wire permutation support into fit() if intended.

#### ✅ ~~F165 · HIGH~~
`nltools/data/collection/__init__.py:189,210,244,255,261,267,293,298,388,404,422,438,460,925,928,931,934,937,940,943,946,949,952,960,966,986,1012,1034,1062,1229` — **~30 public methods/properties on the BrainCollection facade have no docstring**  
BrainCollection is one of the four primary public facades, yet a large, coherent block of its public API is undocumented, so those entries render empty in the auto-generated griffe2md API reference. This is inconsistent even within the file (read, cleanup, shape, is_loaded, n_voxels DO have docstrings). Undocumented: classmethods from_glob (189), from_paths (210); properties n_subjects (244), mask (255), metadata (261), designs (267), cache_root (293); methods memory_estimate (298), smooth (388), standardize (404), detrend (422), threshold (438), resample (460), concat (925), mean (928), std (931), var (934), median (937), sum (940), min (943), max (946), ttest (949), ttest2 (952), anova (960), permutation_test (966), permutation_test2 (986), isc (1012), isc_test (1034), align (1062), write (1229).  
```python
def smooth(self, fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache=...) -> BrainCollection:
        return self.apply("smooth", fwhm=fwhm, ...)   # <-- no docstring
```
*Fix (medium):* Add a one-line Google-style summary (plus Args/Returns for the non-trivial ones like threshold, isc, isc_test, align, permutation_test) to every public method/property. Even delegating one-liners need a summary since griffe does not inherit the delegate's docstring.

#### ✅ ~~F166 · HIGH~~
`nltools/data/fitresults/__init__.py:8-9,97-98,130-131,137-138,182-183,221-222,250-251,347` — **fitresults uses NumPy-style section headers (Parameters/Attributes/Returns/Notes/Examples with `----` underlines) instead of Google style**  
The project mandates Google-style docstrings (Args:/Returns:/Raises:/Examples:/Note:). griffe2md is configured for the Google parser, so these numpydoc `Header\n----` sections are not recognized as sections; the `----` underlines render as Markdown horizontal rules / literal text and the parameter tables do not render. Fit is a public class exported for standalone use, so this degrades a user-facing reference page. Affects the module docstring (Examples, line 8) and the Fit class body (Attributes 97, Returns 130, Examples 137/221/250, Notes 182).  
```python
Attributes
    ----------
    fitted_values : ndarray
        Fitted values or predictions, always present
    ...
    Methods
    -------
    available() : list
```
*Fix (small):* Convert all numpydoc sections to Google style: `Attributes:` / `Returns:` / `Examples:` / `Note:` with `name (type): description` indented entries; drop the `----` underlines.

### Medium severity

#### ✅ ~~F003 · MEDIUM~~
`nltools/algorithms/alignment/local.py:310-360` — **LocalAlignment docstrings are numpydoc-style (name : type) not Google-style, and will mis-render under griffe2md**  
CLAUDE.md mandates Google-style Markdown docstrings (`name (type): description`). The LocalAlignment class docstring and the fit/transform/fit_transform docstrings all use numpydoc field-list style with the parameter name and type on their own line and the description indented beneath (e.g. `scheme : str, default='searchlight'` / `    Spatial scheme: ...`, and `data : List[np.ndarray]`). griffe's Google parser will not parse these as parameters, so the Args/Attributes tables will render incorrectly. The `Examples:` block also references an undefined `mask` (`>>> la.fit(data, mask)`).  
```python
Args:
scheme : str, default='searchlight'
    Spatial scheme: 'searchlight' (overlapping spheres) or 'piecewise'
```
*Fix (small):* Rewrite all four docstrings in Google style: `scheme (str): Spatial scheme ...`, `data (list[np.ndarray]): List of subject arrays ...`, indented one level under `Args:`/`Attributes:`/`Returns:`. Define `mask` in the example.

#### ✅ ~~F069 · MEDIUM~~
`nltools/data/collection/core.py:736-795` — **predict(y=...) is documented/annotated as returning BrainData but returns a Predict dataclass**  
`_predict_group` returns `bd.predict(y=...)` which, with the default `inplace=False`, returns a `Predict` dataclass (see braindata/prediction.py predict_mvpa). Yet `BrainCollection.predict`'s return annotation comment says `# BrainData | BrainCollection`, its docstring says 'y= only → group MVPA → BrainData', and `_predict_group`'s docstring says '→ BrainData with CV attrs'. All three are wrong. The only test asserting BrainData here is xfail(strict=False), so the mismatch is uncaught.  
```python
):  # BrainData | BrainCollection
        """...\n          ``y=`` only    → group MVPA (subjects as samples) → ``BrainData``"""
...
        return bd.predict(y=y_arr, ...)  # -> Predict dataclass
```
*Fix (small):* Update the annotation/docstrings to reflect the `Predict` return (or wrap the result into a BrainData if that was the intended contract), and de-xfail the behavior test.

#### ✅ ~~F081 · MEDIUM~~
`nltools/data/designmatrix/transforms.py:26` — **zscore/standardize/convolve docstrings say 'non-polynomial columns' but the code excludes ALL confounds**  
The default-column selection uses `get_data_columns(dm, exclude_confounds=True)`, which drops every confound (polynomial drift, DCT cosines, AND motion/physio/compcor). The docstrings claim only 'non-polynomial' columns are affected, so a user reading them expects motion regressors to be z-scored/convolved when in fact they are silently skipped. The facade convolve docstring was already corrected to 'non-confound' (__init__.py:531), leaving the others inconsistent and wrong.  
*Evidence:* `transforms.zscore: 'standardize all non-polynomial columns.'  vs  columns_to_zscore = get_data_columns(dm, exclude_confounds=True)`  
*Fix (trivial):* Change 'non-polynomial' to 'non-confound' in transforms.zscore (l26), transforms.standardize (l65), regressors.convolve (l32), and __init__ standardize/zscore docstrings (l702, l858) to match actual behavior.

#### ✅ ~~F099 · MEDIUM~~
`nltools/data/fitresults/__init__.py:8-55` — **fitresults docstrings use NumPy/RST section underlines and stale mode= example, not Google-style Markdown**  
CLAUDE.md requires Google-style Markdown (Args:/Returns:/Examples:), no RST. The module and Fit class docstrings use NumPy-style underlined sections (`Attributes\n----------`, `Examples\n--------`, `Methods\n-------`) which griffe2md's Google parser does not render as sections — they leak as literal text. Additionally the example `brain.fit(X=design_matrix, mode="ridge", ...)` uses the non-canonical `mode=` kwarg (should be `method=`).  
```python
Examples
--------
**Using with BrainData workflow:**
>>> fit = brain.fit(X=design_matrix, mode="ridge", cv_dict={"type": "kfold", "n_splits": 5})
```
*Fix (medium):* Convert underlined sections to Google-style headers; move the Attributes prose into an `Attributes:` Google block or a table; fix `mode=` -> `method=` in the example.

#### ✅ ~~F104 · MEDIUM~~
`nltools/models/glm.py:45` — **Docstrings reference nonexistent class name 'GLMModel' (should be 'Glm')**  
The class is named Glm and exported as Glm in __init__.py, but the docstrings refer to it as GLMModel in four places, including a copy-paste-runnable example 'from nltools.models import GLMModel' and 'model = GLMModel(...)' which would raise ImportError. This is user-facing and misleading in the generated API docs.  
*Evidence:* `>>> from nltools.models import GLMModel  ...  >>> model = GLMModel(t_r=2.0, noise_model='ar1')  ...  'Unlike Ridge ... GLMModel'  ...  'GLMModel: Fitted model instance'`  
*Fix (trivial):* Replace all four 'GLMModel' occurrences (lines 45, 66, 79, 149) with 'Glm'.

#### ✅ ~~F144 · MEDIUM~~
`nltools/stats/correlation.py:28` — **fisher_r_to_z Note claims clipping that the code does not perform**  
The docstring Note says 'Clips r values to (-1, 1) range to avoid invalid arctanh inputs' and an inline comment repeats it, but the implementation only wraps `np.arctanh(r)` in errstate(invalid='ignore'). No clipping occurs: arctanh(±1) returns ±inf and arctanh(|r|>1) returns nan (now silently, because the warning is suppressed). The docstring misrepresents behavior and the warning suppression hides genuinely invalid inputs.  
```python
# Clip r to valid range for arctanh to avoid invalid value warnings
with np.errstate(invalid="ignore"):
    return np.arctanh(r)
```
*Fix (trivial):* Either actually clip (e.g. `np.arctanh(np.clip(r, -1 + eps, 1 - eps))`) and keep the Note, or drop the false Note/comment and reconsider suppressing the invalid-value warning.

#### ✅ ~~F143 · MEDIUM~~
`nltools/stats/correlation.py:36` — **fisher_z_to_r docstring describes the wrong direction and lacks Args/Returns**  
The one-line summary is copy-pasted from fisher_r_to_z: it says 'convert correlation to z score', but this function converts z back to r (np.tanh). The summary is the griffe table one-liner, so the API reference will show the inverse function's description. It also has no Args/Returns sections.  
```python
def fisher_z_to_r(z):
    """Use Fisher transformation to convert correlation to z score."""
    return np.tanh(z)
```
*Fix (trivial):* Rewrite: 'Convert Fisher z back to a correlation coefficient.' plus Args (z) and Returns (r).

#### ✅ ~~F155 · MEDIUM~~
`nltools/mask.py:25-34` — **create_sphere docstring documents a nonexistent 'centers' arg and omits Returns**  
Signature is `create_sphere(coordinates, radius=5, mask=None)`, but Args documents `radius` and `centers` — `centers` is not a parameter, while `coordinates` and `mask` are undocumented. No Returns section (it returns a Nifti1Image). The param-name mismatch leaks into generated API docs.  
```python
Args:
    radius: vector of radius.  ...
    centers: a vector of sphere centers of the form [px, py, pz] ...
```
*Fix (trivial):* Rename `centers` to `coordinates`, document `mask`, add a Returns section.

#### ✅ ~~F178 · MEDIUM~~
`nltools/data/braindata/__init__.py:428-432` — **align() docstring says spatial_scale='roi' is 'not yet implemented' but the code DOES implement it**  
The docstring states "'roi' / 'searchlight' are not yet implemented" and describes `roi_mask` / `radius_mm` as "Reserved", but the method body dispatches `spatial_scale=='roi'` to `align_per_roi` (analysis.py:167), which is fully implemented and covered by tests (test_braindata_spatial_scale.py). Only 'searchlight' actually raises. The stale docstring will tell users a working feature is unavailable.  
*Evidence:* `"'roi' / 'searchlight' are not\n yet implemented ..."  ... roi_mask: Reserved for ``spatial_scale='roi'``.   # but: if spatial_scale == 'roi': return align_per_roi(...)`  
*Fix (trivial):* Rewrite the docstring: 'roi' is supported (requires roi_mask); only 'searchlight' is unimplemented. Change roi_mask from 'Reserved' to 'Atlas image used when spatial_scale="roi"'.

#### ✅ ~~F167 · MEDIUM~~
`nltools/data/fitresults/__init__.py:14` — **Module-docstring example calls BrainData.fit with kwargs that no longer exist (mode=, cv_dict=)**  
The Examples block shows `brain.fit(X=design_matrix, mode="ridge", cv_dict={"type": "kfold", "n_splits": 5})`, but the current signature is `fit(model="glm", *, X=None, cv=None, ...)`. There is no `mode=` param (it is `model=`, and `mode` is not even the v0.6.0 canonical `method`) and no `cv_dict=` param (it is `cv=`). Copy-pasting this example fails at runtime. The `ridge_cv(..., cv_dict=...)` example just below is likely stale for the same reason.  
*Evidence:* `>>> fit = brain.fit(X=design_matrix, mode="ridge", cv_dict={"type": "kfold", "n_splits": 5})`  
*Fix (small):* Update the examples to the real signature (`brain.fit(model="ridge", X=design_matrix, cv=...)`) and verify the ridge_cv example against its current signature.

#### ✅ ~~F169 · MEDIUM~~
`nltools/data/simulator/__init__.py:253-268` — **create_cov_data docstring documents nonexistent params (radius, center, **kwargs)**  
Signature is `create_cov_data(self, cor, cov, sigma, mask=None, reps=1, n_sub=1, output_dir=None)`, but the Args section documents `radius`, `center`, and `**kwargs: Additional keyword arguments to pass to the prediction algorithm` — none of which are parameters. Its first-line summary is also identical to create_ncov_data's ('create continuous simulated data with covariance'), so tables show two indistinguishable entries.  
```python
radius: vector of radius.  Will create multiple spheres if len(radius) > 1
            center: center(s) of sphere(s) ...
            **kwargs: Additional keyword arguments to pass to the prediction algorithm
```
*Fix (trivial):* Delete the radius/center/**kwargs entries and differentiate the first-line summary from create_ncov_data.

#### ✅ ~~F168 · MEDIUM~~
`nltools/data/simulator/__init__.py:366-380` — **create_ncov_data docstring documents params that don't match the signature (mask vs masks, phantom **kwargs)**  
Signature is `create_ncov_data(self, cor, cov, sigma, masks=None, reps=1, n_sub=1, output_dir=None)` but the Args section documents `mask:` (the arg is `masks`) and a `**kwargs: Additional keyword arguments to pass to the prediction algorithm` that does not exist in the signature and is nonsensical for a data simulator (copy-paste from a prediction function). Stale param names mislead users and render a phantom entry.  
```python
mask: region(s) where we will have activations (list if more than one)
            ...
            **kwargs: Additional keyword arguments to pass to the prediction algorithm
```
*Fix (trivial):* Rename the `mask` entry to `masks` and delete the bogus `**kwargs` line.

### Low severity

#### ✅ ~~F008 · LOW~~
`nltools/algorithms/alignment/srm.py:528` — **Typos in user-facing error messages and docstrings**  
transform_subject raises "The number of timepoints(TRs) does not match theone in the model." ("theone") in both SRM (line 528) and DetSRM (line 1013), and the method summary line reads "The subject is assumed to have recieved equivalent stimulation" ("recieved") — also the summary spills onto a second physical line, so griffe's first-line summary will be malformed.  
*Evidence:* `"The number of timepoints(TRs) does not match theone in the model."`  
*Fix (trivial):* Fix the typos ("the one", "received") and make the transform_subject docstring first line a single standalone sentence ending in a period.

#### ✅ ~~F018 · LOW~~
`nltools/algorithms/inference/bootstrap.py:179-226` — **OnlineBootstrapStats.get_results docstring has non-standard sections and non-runnable examples**  
The Returns/Examples block uses bold pseudo-headers ('**Basic usage:**', '**Usage:**') that are not Google-style sections, and the second example references undefined names (`bootstrap_samples`) and comments like '# Returns: {...}' as if doctest output. griffe will render this as a long, partly-broken example. It also duplicates the class-level example verbatim.  
```python
**Basic usage:**
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
...
**Usage:**
>>> ... for sample in bootstrap_samples:  # Iterate over samples
```
*Fix (trivial):* Trim to a single runnable Examples block; drop the bold sub-headers and the pseudo-code referencing undefined variables.

#### ✅ ~~F030 · LOW~~
`nltools/algorithms/ridge/core.py:85, 220` — **Docstrings claim GPU falls back to CPU 'if GPU unavailable' but torch-missing raises ImportError**  
ridge_svd/ridge_cv (and the solvers) document parallel='gpu' as 'GPU acceleration via PyTorch (falls back to CPU if GPU unavailable)'. resolve_backend('gpu') constructs Backend('torch'), which raises ImportError when torch is not installed (only parallel='auto' falls back). So on a torch-less machine the promised graceful fallback does not happen — the call errors. The docstring overstates robustness.  
*Evidence:* `"gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable)  # but Backend('torch') raises if torch absent`  
*Fix (trivial):* Clarify that fallback only covers 'torch present but no GPU device' (→ torch-cpu), and that torch must be installed for 'gpu'; suggest 'auto' for torch-optional fallback.

#### ✅ ~~F029 · LOW~~
`nltools/algorithms/ridge/utils.py:124-130, 206-212, 291-293` — **Helper docstrings omit the required `backend` argument (and _decompose_ridge default None crashes)**  
_decompose_ridge, _select_best_alphas, and _r2_score all take a `backend` argument that is effectively required (they call backend.svd / backend.xp), but none of their Args sections document `backend`. Worse, _decompose_ridge declares `backend: Backend | None = None`; called with the default it raises AttributeError on `backend.svd` (line 178) rather than a clear error. The docstring param list therefore mismatches the true signature.  
*Evidence:* `def _decompose_ridge(Xtrain, alphas, n_alphas_batch=None, method='svd', backend=None):  # Args: lists no `backend`; None default -> None.svd`  
*Fix (trivial):* Document `backend` in each Args section, and either make it a required positional (drop the None default) or raise an explicit ValueError('backend is required') when None.

#### ✅ ~~F055 · LOW~~
`nltools/data/braindata/__init__.py:1646-1647` — **BrainData.similarity docstring understates supported methods**  
The facade docstring lists method options as ['correlation','dot_product','cosine'], but the underlying analysis.similarity accepts ['correlation','pearson','rank_correlation','spearman','dot_product','cosine']. Users can't discover rank_correlation/spearman/pearson from the public docstring.  
```python
method: (str) Type of similarity
                ['correlation','dot_product','cosine']
```
*Fix (trivial):* List the full supported set to match analysis.similarity's supported_metrics.

#### ✅ ~~F064 · LOW~~
`nltools/data/braindata/cache.py:66-70` — **hash_mask docstring example shows a non-hexadecimal hash**  
The docstring advertises a 16-character hexadecimal hash (sha256[:16]) but the example output contains 'g' and 'h', which are not hex digits, so it cannot be a real output of this function.  
```python
>>> hash_mask(mask)
'a1b2c3d4e5f6g7h8'
```
*Fix (trivial):* Replace with a plausible hex string, e.g. 'a1b2c3d4e5f60789'.

#### ✅ ~~F065 · LOW~~
`nltools/data/braindata/io.py:773-798` — **upload_neurovault marks img_type/img_modality as Required but they are never validated**  
Both io.py and the facade docstring label img_type and img_modality as 'Required', yet only access_token is validated. If left None they are silently forwarded to pynv as map_type=None/modality=None, producing an opaque third-party API error rather than the clear ValueError users get for a missing token.  
```python
if access_token is None:
    raise ValueError("You must supply a valid neurovault access token")
# no check for img_type / img_modality despite 'Required' in docstring
```
*Fix (trivial):* Add explicit None checks for img_type and img_modality mirroring the access_token guard, with actionable messages.

#### ✅ ~~F039 · LOW~~
`nltools/data/adjacency/stats.py:50-53` — **similarity() Returns section omits the BrainData return when project=True**  
The docstring documents the return as 'dict or list', but when `project=True` and the Adjacency has a spatial_scale, the function returns a `BrainData` (line 196-198). The facade docstring in __init__.py (lines 686-724) has no Returns section at all. Users relying on the documented type will be surprised.  
```python
Returns:
    dict or list: Correlation result dict with keys 'r' and 'p', or a list of
        such dicts when adj contains multiple matrices.
```
*Fix (trivial):* Document the third return branch: 'BrainData when project=True (per-matrix correlations projected via spatial_scale)'. Also add a Returns section to the facade method docstring.

#### ✅ ~~F044 · LOW~~
`nltools/data/atlases/labeling.py:30-37` — **_xyz_to_ijk docstring claims it clamps out-of-bounds voxels, but it only rounds/casts**  
The docstring says 'Out-of-bounds voxels get clamped to the origin (treated as background).' but `_xyz_to_ijk` only solves the affine, rounds, and casts to int. The clamping actually happens in the separate `_clip_to_box` helper. The stale claim is misleading for anyone editing the labeling pipeline (e.g. they might assume clamping is already done and drop the `_clip_to_box` call).  
```python
"""Map MNI mm -> integer voxel ijk via the inverse affine.

    Out-of-bounds voxels get clamped to the origin (treated as background).
    """
    ...
    return np.round(ijk).astype(int)
```
*Fix (trivial):* Drop the clamping sentence from `_xyz_to_ijk` (it does not clamp) and keep that note on `_clip_to_box`, which actually implements it.

#### ✅ ~~F075 · LOW~~
`nltools/data/collection/pipeline.py:36-42` — **BrainCollectionPipeline class docstring example uses stale cv(scheme=) and algorithm= API**  
The example shows `.cv(scheme='loso')` and `.predict(labels, algorithm='svm')`, but `BrainCollection.cv` takes `method=` (translated to CVScheme.scheme internally), so `cv(scheme=...)` would be a TypeError, and `algorithm=` conflicts with the canonical kwarg. A copy-pasted example will fail.  
```python
>>> result = (bc
...     .cv(scheme='loso')
...     .standardize()
...     .predict(labels, algorithm='svm'))
```
*Fix (trivial):* Update the example to `.cv(method='loso')` and the canonical predict kwarg.

#### ✅ ~~F074 · LOW~~
`nltools/data/collection/pipeline.py:569-583` — **pipeline.py uses numpydoc/RST docstring sections that won't render under griffe Google-style**  
`_apply_contrast` uses RST/numpydoc `Parameters\n----------` / `Returns\n-------` underline sections, and the class/`pool` docstrings use `Examples\n--------` underlines. Project convention is Google-style Markdown (Args:/Returns:/Examples:) with no RST; these underline sections leak unrendered into the docs.  
```python
Parameters
        ----------
        data : np.ndarray
            Shape (n_subjects, n_conditions, n_voxels).
```
*Fix (small):* Convert all numpydoc-underline sections in pipeline.py to Google-style `Args:`/`Returns:`/`Examples:`.

#### ✅ ~~F100 · LOW~~
`nltools/data/roc/__init__.py:1-6` — **Module docstrings use RST `====` heading underlines that leak in mystmd docs**  
roc/__init__.py and simulator/__init__.py module docstrings use RST section-underline headings ('NeuroLearn Analysis Tools\n=========='). Under the Google-style Markdown convention these underlines render as literal text/leak in griffe2md output. Also the Roc class docstring summary section is 'Roc Class' RST-title-cased rather than a one-line sentence summary.  
```python
"""
NeuroLearn Analysis Tools
=========================
These tools provide the ability to quickly run
machine-learning analyses on imaging data
"""
```
*Fix (trivial):* Replace RST underlines with a plain one-line summary sentence + Markdown prose.

#### ✅ ~~F110 · LOW~~
`nltools/models/glm.py:78` — **Uses non-canonical 'Notes:' section instead of 'Note:'**  
CLAUDE.md lists 'Note:' as the canonical Google-style admonition section. The Glm class docstring uses both 'Note:' (line 41) and 'Notes:' (line 78), and fit/predict/score use 'Notes:'. The plural form may not render as an admonition under griffe2md, leaking as plain text and reading inconsistently.  
*Evidence:* `Note:  (line 41)  ...  Notes:  (line 78, 151, 229, 262)`  
*Fix (trivial):* Standardize on 'Note:' across the module (or verify griffe2md renders 'Notes:'; otherwise convert all plural occurrences).

#### ✅ ~~F122 · LOW~~
`nltools/pipelines/cv.py:33` — **CVScheme `n` docstring says 'bootstrap iterations' but n also drives permutation count**  
The Args entry documents `n` as 'Number of bootstrap iterations (for bootstrap scheme)', but n is equally the permutation count used by _permutation_split and returned by n_splits() for scheme='permutation'. The docstring is stale/incomplete for the permutation path. (Separately, per API conventions permutation count should be n_permute and bootstrap count n_samples — this single overloaded `n` also leaks through the BrainCollection.cv() facade.)  
*Evidence:* `n: Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000.`  
*Fix (trivial):* Reword to 'Number of resampling iterations (bootstrap draws or permutations)', and consider splitting into n_samples/n_permute to match the canonical kwarg table.

#### ✅ ~~F131 · LOW~~
`nltools/plotting/brain.py:44-50` — **plot_interactive_brain docstring: wrong threshold default and false Returns claim**  
The threshold arg doc says 'default 0' but the actual default is 1e-6. The Returns section says it returns an 'interactive brain viewer widget', but the function calls ipywidgets.interact(...) and returns None (interact renders as a side effect). Both statements mislead users.  
```python
threshold (float/str): threshold to initialize the visualization, maybe be a percentile string; default 0
    ...
    Returns:
        interactive brain viewer widget
```
*Fix (trivial):* State 'default 1e-6' and change Returns to 'None (renders widgets inline)'. Also fix typo 'maybe be' -> 'may be'.

#### ✅ ~~F132 · LOW~~
`nltools/plotting/decomposition.py:21-27` — **component_viewer summary line is ungrammatical and missing a Returns note**  
First physical line 'This a function to interactively view the results of a decomposition analysis.' is missing a verb ('This is a...') and reads awkwardly as the one-line table summary griffe extracts. The nested component_inspector docstring repeats the same 'This a function' typo. There is no Returns note (function returns None / renders widgets).  
*Evidence:* `"""This a function to interactively view the results of a decomposition analysis.`  
*Fix (trivial):* Rewrite the summary as a clean standalone sentence, e.g. 'Interactively view the results of a BrainData.decompose() run.' and note it renders inline and returns None.

#### ✅ ~~F130 · LOW~~
`nltools/plotting/prediction.py:18-21` — **prediction docstrings mislabel input as 'a pandas file' and promise a returned figure that is never returned**  
All four functions describe stats_output as 'a pandas file' (it is a DataFrame) and document `Returns: fig: Will return a seaborn/matplotlib plot`, but each function ends with bare `return` (None). The Returns sections are inaccurate for every function in the module.  
```python
Args:
        stats_output: a pandas file with prediction output
    Returns:
        fig: Will return a seaborn plot of distance from hyperplane
```
*Fix (trivial):* Fix wording to 'pandas DataFrame' and, once the functions actually return their figure/grid, make the Returns line accurate.

#### ✅ ~~F145 · LOW~~
`nltools/stats/alignment.py:242-266` — **procrustes docstring uses numpydoc field-list style, not Google-style**  
Args and Returns are written numpydoc-style (`data1 : array_like` on its own line with an indented description block) rather than the Google-style `name: description` the project mandates for griffe2md/mystmd. This tends to render incorrectly (params not parsed into the params table).  
```python
data1 : array_like
            Matrix, n rows represent points in k (columns) space `data1` is the
            reference data ...
```
*Fix (small):* Convert Args/Returns to Google-style `data1: ...` / `mtx1: ...` single-line-lead entries.

#### ✅ ~~F162 · LOW~~
`nltools/__init__.py:1` — **Top-level package has no module docstring**  
`nltools/__init__.py` opens directly with `__all__`; there is no module docstring. The package front door shows empty in `help(nltools)` and the generated API index. Convention flags missing module docstrings.  
```python
__all__ = [
    "SRM",
    "Adjacency",
```
*Fix (trivial):* Add a one-paragraph module docstring before `__all__`.

#### ✅ ~~F161 · LOW~~
`nltools/cross_validation.py:26-72` — **KFoldStratified docstrings use numpydoc field-list style, not project Google-style**  
Args are numpydoc-style (`n_splits: int, default=3`; `X : array-like, shape (n_samples, n_features)`) and Returns lists train/test as fields. Project mandates Google-style Args/Returns with Markdown; this renders awkwardly under griffe2md.  
```python
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
```
*Fix (small):* Reformat to Google-style `Args:`/`Returns:` with `name: description` entries.

#### ✅ ~~F163 · LOW~~
`nltools/mask.py:274` — **roi_to_brain_from_atlas docstring describes values as 1-D but code supports 2-D**  
The summary and `values` Args entry say '1-D array of per-parcel scalars' with 'len(values) must match len(roi_labels)', but the implementation explicitly accepts 2-D `(n_images, n_parcels)` (validated l.309, handled l.362-368). The docstring understates supported shapes.  
```python
values: 1-D array of per-parcel scalars; ``len(values)`` must match
    ``len(roi_labels)`` ...
```
*Fix (trivial):* Document both the 1-D `(n_parcels,)` and 2-D `(n_images, n_parcels)` cases.

#### ✅ ~~F176 · LOW~~
`nltools/__init__.py:1` — **Top-level package has no module docstring**  
nltools/__init__.py starts directly with `__all__ = [...]` and has no top-of-file docstring, so the package's landing entry in the API reference has no description.  
```python
__all__ = [
    "SRM",
    "Adjacency",
```
*Fix (trivial):* Add a concise package-level docstring summarizing nltools.

#### ✅ ~~F172 · LOW~~
`nltools/data/braindata/io.py:330` — **RST role `:func:` leaks in the _mask_images_fast docstring**  
The one confirmed RST-role leak in the package. Although the function is private (hidden from the API ref by default), the project convention bans RST roles in any docstring because they render literally if surfaced.  
*Evidence:* `out of the per-image loop. Split out so the fallback in :func:`mask_images``  
*Fix (trivial):* Replace with a plain Markdown code span: `mask_images`.

#### ✅ ~~F173 · LOW~~
`nltools/data/collection/pipeline.py:572-582` — **collection/pipeline.py mixes numpydoc sections into otherwise Google-style docstrings**  
BrainCollectionPipeline and FittedBrainCollection use Google-style Examples elsewhere, but methods like _apply_contrast use NumPy-style `Parameters\n----------` / `Returns\n-------`. Public methods in this file (413, 503) also carry the `>>>` Examples style, and any public one using numpydoc sections will render broken under the Google parser.  
```python
Parameters
        ----------
        data : np.ndarray
            Shape (n_subjects, n_conditions, n_voxels).
```
*Fix (trivial):* Standardize on Google-style Args:/Returns: throughout this module.

#### ✅ ~~F171 · LOW~~
`nltools/data/roc/__init__.py:1-6,17,63,233` — **Roc module docstring is inaccurate and uses an RST title underline; class/method summaries lack terminal periods**  
The module docstring reads 'NeuroLearn Analysis Tools\n=========================\nThese tools provide the ability to quickly run machine-learning analyses' — a generic, inaccurate description of a ROC-only module, with an RST title underline that renders literally and no terminal period. The Roc class summary ('Roc Class'), calculate, and plot summaries also lack a terminal period.  
```python
"""
NeuroLearn Analysis Tools
=========================
These tools provide the ability to quickly run
machine-learning analyses on imaging data
"""
```
*Fix (trivial):* Rewrite as a one-line Google-style module summary describing ROC analysis; drop the RST underline; add terminal periods to the class/method summaries.

#### ✅ ~~F170 · LOW~~
`nltools/data/simulator/__init__.py:1,72,102,124,142,157,191,253,366,582,637,677,750,789` — **Simulator module docstring uses RST title underline; ~14 method summaries are lowercase and lack a terminal period**  
The module docstring uses an RST section-title underline ('==========================' under 'NeuroLearn Simulator Tools') which renders literally. Many method first-line summaries (gaussian, sphere, normal_noise, to_nifti, n_spheres, create_data, create_cov_data, create_ncov_data, add_signal, fit, etc.) start lowercase and do not end with a period, violating the 'first physical line = complete standalone sentence ending with a period' rule that feeds griffe's one-line summary tables.  
*Evidence:* `"""create a 3D gaussian signal normalized to a given intensity  (no leading capital, no period)`  
*Fix (small):* Drop the RST underline from the module docstring; capitalize each method summary and end it with a period.

#### ✅ ~~F196 · LOW~~
`nltools/data/roc/__init__.py:72` — **Roc.calculate docstring documents a **kwargs param the signature lacks and has a wrapped first line**  
The docstring's Args lists '**kwargs: Additional keyword arguments to pass to the prediction algorithm', but calculate()'s signature ends at balanced_acc=False with no **kwargs — a stale param that will render a phantom argument in the API docs. Also the first physical line ('Calculate Receiver Operating Characteristic plot (ROC) for') wraps mid-phrase instead of being a complete standalone one-sentence summary, and 'input_values: nibabel data instance' misdescribes the argument.  
```python
"""Calculate Receiver Operating Characteristic plot (ROC) for
        single-interval classification.
...
            **kwargs: Additional keyword arguments to pass to the prediction algorithm
```
*Fix (trivial):* Remove the **kwargs entry, make the first line a single ≤120-char sentence ending in a period, and correct the input_values description.

## Dead code & deferred gaps (23)

> **✅ BUCKET CLOSED — 2026-07-17 (dead-code session).** All 23 findings addressed.
> - **Already resolved by the earlier Layer-B pipelines strip** (base.py no longer defines the CVScheme Protocol or Terminal; steps.py has no AlignStep/FittedAlign): **F114, F179, F115, F116**. `from nltools.pipelines import CVScheme` now correctly resolves to the concrete `cv.CVScheme`.
> - **Fixed this session:** F007, F010, F015, F023 (both `batch_or_skip` copies deleted), F024, F054, F062, F070 (FittedBrainCollection ~290 LOC removed), F094 (+ misleading `correction='permutation'` branch), F095, F109, F121/F181 (StatResult.to_nifti stub + its test), F133, F138 (dead ISC trio ~180 LOC), F147, F185 (regress/predict_multi shims removed entirely — Adjacency.regress untouched).
> - **Already fixed before this session (no-op):** F084 (`upsample` no longer had `**kwargs`).
> - **F022 — WIRED (complete).** `max_gpu_memory_gb` now drives GPU target-batch sizing everywhere it appears, via `ridge/utils._auto_n_targets_batch(max_gpu_memory_gb, elements_per_target, n_targets)` (GPU path only; CPU/None unchanged; mirrors the removed ICC batcher). Wired into the three CV solvers (`solve_ridge_cv`, `solve_banded_ridge_cv`, `cross_val_predict_ridge`) and into `core.ridge_svd`'s torch branch (batches over target columns; `core.ridge_cv` forwards the budget). Verified on torch-MPS: forcing a 0.001 GB budget triggers multi-batch execution and matches the unbatched numpy result to 8e-7 (float32 noise).
> - Gate: `poe lint` + `lint-api` (0 semgrep / 0 check_kwonly), targeted suites all green (~900 tests across ridge/simulator/isc/alignment/pipeline/collection/braindata/stats/plotting).


### Medium severity

#### F022 · MEDIUM
`nltools/algorithms/ridge/solvers.py:51, 637, 912` — **max_gpu_memory_gb is accepted across all solvers (and ridge_svd/ridge_cv) but never referenced**  
max_gpu_memory_gb is declared in ridge_svd, ridge_cv, solve_ridge_cv, solve_banded_ridge_cv, and cross_val_predict_ridge, and documented as 'GPU memory budget in GB', but it is never read in any of these bodies (memory control is done manually via n_targets_batch/n_alphas_batch). Unlike inference/icc.py which actually uses max_gpu_memory_gb, here it is dead. Users expecting it to bound GPU allocation will be surprised by OOMs.  
*Evidence:* `max_gpu_memory_gb: float = 4.0,  # never used in ridge solvers/core bodies`  
*Fix (small):* Remove the parameter from the ridge solvers/core functions (v0.6.0 is breaking), or wire it into the batch-size auto-selection so it actually caps GPU memory.

#### F023 · MEDIUM
`nltools/algorithms/ridge/utils.py:87-121` — **_batch_or_skip is dead code and duplicates batch_or_skip in shape_utils.py**  
_batch_or_skip in ridge/utils.py and batch_or_skip in algorithms/shape_utils.py are near-identical implementations of the same himalaya pattern. Neither has any call site anywhere in the codebase (grep finds only their definitions and their own docstring examples). This is both duplicated logic and dead code — a single-source-of-truth violation plus an unused function shipped in a release.  
```python
def _batch_or_skip(array, batch, axis):  # utils.py, no callers
def batch_or_skip(array, batch, axis):    # shape_utils.py, no callers
```
*Fix (trivial):* Delete both if genuinely unused, or keep exactly one (in shape_utils.py) and import it where the batching pattern is intended to be applied. Do not ship two copies of an uncalled helper.

#### F070 · MEDIUM
`nltools/data/collection/pipeline.py:393-686` — **FittedBrainCollection is dead code — never instantiated or exported**  
FittedBrainCollection (~290 lines) is defined but never constructed anywhere in nltools (grep finds no `FittedBrainCollection(` instantiation; it's absent from collection/__init__.py __all__). `bc.fit()` returns a plain BrainCollection via `_clone`, not this class. It depends on `nltools.pipelines.pool.PooledData` and `.base.FittedStack` and a `_design_columns`/`return_stats` protocol that the current fit path doesn't produce. For a breaking v0.6.0 this superseded orchestration should be removed.  
```python
class FittedBrainCollection:
    """Wrapper for fitted BrainCollection enabling pool() chaining. ..."""
```
*Fix (small):* Delete FittedBrainCollection (and its pipelines.pool dependency if now orphaned), or wire it in and cover it; do not ship dead orchestration in the breaking release.

#### F095 · MEDIUM
`nltools/data/simulator/__init__.py:288` — **Leftover debug print() calls dump full arrays during simulation**  
create_cov_data and create_ncov_data contain numerous stray debug prints — `print(mv_sim)` (288), `print(mv_sim_l)` (443), and progress/shape prints at 413, 439, 453, 464-465, 478, 482-484. `print(mv_sim)`/`print(mv_sim_l)` dump entire simulated arrays to stdout on every call. These are development leftovers, not user-facing output.  
```python
mv_sim = self.random_state.multivariate_normal(...)
print(mv_sim)
```
*Fix (small):* Remove the debug prints (or gate real progress messages behind a progress_bar/verbose flag per API conventions).

#### F094 · MEDIUM
`nltools/data/simulator/__init__.py:622-635` — **_run_permutation is never called; permutation 'correction' is not actually implemented**  
`_run_permutation` computes one_sample_permutation_test per pixel but no method ever calls it. fit() only calls _run_ttest, and run_multiple_simulations always uses _run_ttest. Yet _threshold_simulation/threshold_simulation accept correction='permutation' (validated against threshold_type=='p'), giving the false impression permutation-based thresholding is supported — it actually thresholds parametric t-test p-values. Dead code plus a misleading capability.  
```python
def _run_permutation(self, data):
    ... one_sample_permutation_test(flattened[i, :]) ...  # no caller
```
*Fix (medium):* Either wire _run_permutation into fit()/run_multiple_simulations for correction='permutation', or remove the method and the 'permutation' correction branch.

#### F114 · MEDIUM
`nltools/pipelines/base.py:105-124` — **CVScheme is defined twice (Protocol in base.py, concrete dataclass in cv.py); __init__ exports the unused Protocol, shadowing the real class**  
base.py defines `class CVScheme(Protocol)` and __init__.py exports THAT one (`from .base import CVScheme`). The concrete, actually-used config class is cv.py's `CVScheme` dataclass, which every real caller imports directly (`from ...pipelines.cv import CVScheme`, tests likewise). So `from nltools.pipelines import CVScheme` hands users the empty Protocol, not the config object — a name collision that is confusing and error-prone. The base.py Protocol has no implementer that relies on it and its split() signature (no `groups`) doesn't even match the concrete class.  
```python
# base.py
@runtime_checkable
class CVScheme(Protocol): ...
# __init__.py
from .base import (CVScheme, FittedStack, ...)
```
*Fix (small):* Delete the base.py CVScheme Protocol and export the concrete cv.CVScheme from __init__ (or rename the protocol, e.g. CVSplitter). One name, one meaning.

#### F115 · MEDIUM
`nltools/pipelines/base.py:127-155` — **Terminal protocol has no implementer and is never consumed**  
The `Terminal` protocol (fit_evaluate(...)) is exported but nothing implements or consumes it. BrainCollectionPipeline.predict inlines model fit/score in _execute_loso / _execute_pooled_cv and never references Terminal or FittedStack.inverse_transform. This is leftover abstraction from the removed standalone Pipeline orchestrator. Same applies to FittedStack.inverse_transform / is_fully_invertible, which are appended to but never invoked by any live caller.  
```python
class Terminal(Protocol):
    def fit_evaluate(self, train_data, test_data, train_idx, test_idx, fitted_stack: FittedStack) -> Any: ...
```
*Fix (small):* Remove Terminal (and the unused FittedStack inversion API, or wire it in) now that the fluent Pipeline orchestrator is gone. Dead protocols mislead readers about how the module is used.

#### F116 · MEDIUM
`nltools/pipelines/steps.py:437-538` — **AlignStep / FittedAlign are orphaned — no BrainCollectionPipeline builder adds an align step**  
BrainCollectionPipeline exposes standardize()/reduce()/pipe()/predict() but has no .align() method, so AlignStep and FittedAlign are unreachable through the public facade; they are only exercised by test_pipeline_align.py. After the standalone Pipeline removal, this alignment machinery (a substantial ~100-line class plus transform_new_subject/inverse_transform) is dead relative to the shipping API.  
*Evidence:* `rg 'align|AlignStep' nltools/data/collection/pipeline.py  -> (no matches)`  
*Fix (medium):* Either add a `.align(method=..., ...)` builder to BrainCollectionPipeline so LOSO alignment is actually usable, or drop AlignStep/FittedAlign from the release. Do not ship it half-wired.

#### F138 · MEDIUM
`nltools/stats/intersubject.py:39` — **Dead ISC helper cluster (_bootstrap_isc, _compute_isc_group, _permute_isc_group)**  
`isc` and `isc_group` now delegate to nltools.algorithms.inference (isc_permutation_test / isc_group_permutation_test). `_bootstrap_isc` (line 39) has no callers; `_permute_isc_group` (line 243) has no callers; `_compute_isc_group` (line 181) is called only by the unused `_permute_isc_group`. This ~130-line block is superseded and duplicates logic maintained/tested in inference/isc.py. It also carries latent bugs (e.g. _permute_isc_group indexes the permuted matrix with the original `group` mask, only correct for contiguous group arrays; line 285 has a `permute_group = permute_group = ...` double-assignment), which is extra reason to remove rather than keep.  
```python
def _bootstrap_isc(...)  # no callers
def _permute_isc_group(...)  # no callers; calls _compute_isc_group
permute_group = permute_group = random_state.permutation(group)
```
*Fix (small):* Delete the three private helpers (single source of truth is inference/isc.py). Confirm no external imports rely on them.

#### F179 · MEDIUM
`nltools/pipelines/base.py:106` — **CVScheme name collision: the public export is an unused Protocol, the concrete class users need is not exported**  
`pipelines/base.py` defines `class CVScheme(Protocol)` and `pipelines/__init__.py` exports THAT in `__all__`. But the real, instantiable `CVScheme` lives in `pipelines/cv.py` (used by collection/__init__.py:1111 and cv() facade). The Protocol is never used as a type annotation anywhere (grep shows only docstring mentions) — so `from nltools.pipelines import CVScheme` hands users the empty Protocol, not the class that takes `scheme=`, `k=`, etc. Two classes, same name, wrong one public.  
*Evidence:* `base.py:106 `class CVScheme(Protocol):` exported in __init__ __all__; cv.py:18 `class CVScheme:` (the real one) NOT exported; collection imports `from ...pipelines.cv import CVScheme`.`  
*Fix (small):* Rename the Protocol (e.g. `CVSchemeProtocol` or `Splitter`) or drop it from the public `__all__`, and export the concrete cv.CVScheme instead so the public name resolves to the usable class.

#### F181 · MEDIUM
`nltools/pipelines/pool.py:412` — **PooledData.to_nifti is a public method that only raises NotImplementedError**  
`StatResult.to_nifti(path, mask)` is a public method on an exported class but its entire body is a placeholder that raises unconditionally ('to_nifti requires mask integration'). It accepts `path` and `mask` args that are never used. Either a half-implemented feature to finish or dead surface to remove.  
```python
def to_nifti(self, path: str, mask=None) -> None:
    # Placeholder - would need mask to reconstruct 3D
    raise NotImplementedError("to_nifti requires mask integration")
```
*Fix (medium):* Implement using the stored mask/back-projection path, or remove the method until it can be supported so the public API doesn't advertise a dead entry point.

### Low severity

#### F010 · LOW
`nltools/algorithms/alignment/local.py:70` — **Unused scale return and unused voxel_to_parcel field**  
`_orthogonal_procrustes_backend` returns `scale` documented as "not used, kept for compatibility," and its only caller `_fit_local_procrustes` discards it (`R, _ = ...`). Separately, PiecewiseNeighborhoods.voxel_to_parcel is computed in _compute_piecewise_neighborhoods (line 164) but never read anywhere (transform iterates parcel_to_voxels). Both are minor dead surface area.  
```python
R = Vt.T @ U.T
scale = s.sum()
return R, scale   # scale never consumed
```
*Fix (trivial):* Drop the scale return (return only R) or document why it must stay; remove voxel_to_parcel or document a concrete use.

#### F007 · LOW
`nltools/algorithms/alignment/srm.py:649` — **Redundant double reduction: np.sum(...).sum()**  
`np.sum(w_new * a_subject)` already returns a 0-d scalar; the trailing `.sum()` is a no-op. Same pattern in the single-threaded branch at line 697. Harmless but confusing and suggests a copy-paste leftover.  
*Evidence:* `rho2_new += -2 * np.sum(w_new * a_subject).sum()  # line 649; line 697 identical pattern`  
*Fix (trivial):* Drop the redundant `.sum()`: `rho2_new += -2 * np.sum(w_new * a_subject)`.

#### F015 · LOW
`nltools/algorithms/inference/correlation.py:487-492` — **Unused `batch_idx = torch.arange(...)` shadows the batch loop variable**  
Inside `for batch_idx in range(n_batches)` (line 455), the Pearson branch reassigns `batch_idx = torch.arange(current_batch_size, ...)[:, None]` at line 487 but never uses that tensor (advanced indexing at line 490 uses `batch_indices_device`, not `batch_idx`). It shadows the loop counter and is confusing; harmless only because Python's range iterator holds its own state.  
```python
batch_idx = torch.arange(
    current_batch_size, device=batch_indices_device.device
)[:, None]  # (current_batch_size, 1)
perm_data1 = data1_device[batch_indices_device]
```
*Fix (trivial):* Delete the unused assignment.

#### F024 · LOW
`nltools/algorithms/ridge/solvers.py:527` — **_refit_banded_ridge's `alphas` parameter is never used**  
_refit_banded_ridge takes `alphas` but the body only uses `best_alphas` and `unique(best_alphas)`; `alphas` is never read. cross_val_predict_ridge even passes it with the comment 'unused internally; uses unique(best_alphas)'. It is a confusing dead parameter that forces callers to pass a redundant argument.  
*Evidence:* `def _refit_banded_ridge(... best_alphas, alphas, ...):  # `alphas` never referenced in body`  
*Fix (trivial):* Drop the `alphas` parameter and update the two call sites (solve_ridge_cv, cross_val_predict_ridge).

#### F054 · LOW
`nltools/data/braindata/__init__.py:28-29` — **Module-level nx and MAX_INT in braindata/__init__.py are unused**  
nx = attempt_to_import('networkx', 'nx') and MAX_INT = np.iinfo(np.int32).max are defined at module scope but referenced nowhere in the 1979-line file, and nothing imports them from here (other modules define their own local MAX_INT). Dead imports/constants left over from the pre-facade class.  
```python
nx = attempt_to_import("networkx", "nx")
MAX_INT = np.iinfo(np.int32).max
```
*Fix (trivial):* Delete both lines (and the now-unused attempt_to_import import if nothing else in the file uses it).

#### F062 · LOW
`nltools/data/braindata/io.py:205` — **detect_space's `bd` parameter is unused**  
detect_space(bd, mask) never references bd; the docstring even says '(unused, kept for API consistency)'. It is always called as detect_space(bd, bd.mask). Dead argument that adds noise and invites confusion about whether it mutates bd.  
```python
def detect_space(bd, mask):
    """...bd: BrainData instance (unused, kept for API consistency)..."""
```
*Fix (small):* Drop the `bd` parameter (this is an internal helper, not a public API), updating the ~4 call sites.

#### F084 · LOW
`nltools/data/designmatrix/transforms.py:189` — **upsample()'s `**kwargs` ('Reserved for future extensions') is accepted then silently swallowed**  
upsample declares `**kwargs` documented as 'Reserved for future extensions' but never reads or forwards it. Any typo'd or unsupported keyword (e.g. `dm.upsample(2.0, methdo='nearest')`) is silently accepted and ignored, hiding user error. Combined with the facade forwarding, this is unused internal kwargs surface.  
*Evidence:* `def upsample(dm, target, method='linear', **kwargs):  # kwargs never referenced`  
*Fix (trivial):* Remove `**kwargs` from both the internal upsample and the facade; add explicit parameters if/when a real option is needed.

#### F109 · LOW
`nltools/models/ridge.py:238` — **Redundant if/elif branches produce identical alpha_ squeeze result**  
The 'if y_was_1d and ...' branch and the 'elif ...' branch have identical bodies (float(self.alpha_[0])) and identical guard conditions except the y_was_1d flag, which is irrelevant when alpha_ is a length-1 1D array. Both cases do the same thing, so the y_was_1d condition is dead/pointless and just obscures intent.  
*Evidence:* `if (y_was_1d and isinstance(self.alpha_, np.ndarray) and self.alpha_.ndim == 1 and self.alpha_.shape[0] == 1): self.alpha_ = float(self.alpha_[0])  elif (isinstance(...) ndim==1 shape[0]==1): self.alpha_ = float(self.alpha_[0])`  
*Fix (trivial):* Collapse to a single condition: if isinstance(self.alpha_, np.ndarray) and self.alpha_.ndim == 1 and self.alpha_.shape[0] == 1: self.alpha_ = float(self.alpha_[0]).

#### F121 · LOW
`nltools/pipelines/pool.py:405-413` — **StatResult.to_nifti is a permanent NotImplementedError stub**  
to_nifti unconditionally raises NotImplementedError('to_nifti requires mask integration'), and the test locks that in (test_pipeline_pool.py:229 asserts it raises). PooledData already carries a `mask` nibabel image (pool.py:52) that could be threaded through, but StatResult has no mask. Shipping a public method that only ever raises is a poor UX for a release.  
*Evidence:* `raise NotImplementedError("to_nifti requires mask integration")`  
*Fix (small):* Implement it by carrying the mask onto StatResult, or remove the method until it works rather than shipping a raising stub.

#### F133 · LOW
`nltools/plotting/adjacency.py:177-189` — **plot_between_label_distance carries an unused fontsize parameter**  
The fontsize parameter is immediately discarded (`del fontsize`) and documented as 'Reserved for future use; currently unused.' It is dead API surface on a public function heading into a breaking release where signatures can be cleaned up.  
```python
fontsize=18,
    ...
    del fontsize  # kept for API parity, not used
```
*Fix (trivial):* Drop fontsize from the signature (this is a breaking release) or actually apply it to the axis title/labels.

#### F147 · LOW
`nltools/stats/corrections.py:30-34` — **fdr validates p-value range twice**  
The same range check is performed with Python `any()` then again with `np.any()` back-to-back, raising different messages for the identical condition. The first (`any(p < 0)`) also relies on Python's builtin any over a numpy array. Redundant.  
```python
if any(p < 0) or any(p > 1):
    raise ValueError("array contains p-values that are outside the range 0-1")
if np.any(p > 1) or np.any(p < 0):
    raise ValueError("Does not include valid p-values.")
```
*Fix (trivial):* Keep a single np.any-based check with one clear message.

#### F185 · LOW
`nltools/data/braindata/modeling.py:773` — **Deprecated raise-only shims (regress, predict_multi) retained with dead kwargs in a breaking release**  
`modeling.regress(bd, design_matrix, method, mode)` and `BrainData.regress`/`BrainData.predict_multi` exist only to raise NotImplementedError with a migration message; their kwargs (design_matrix, method, mode='mode') are never read (the facade at 1570 still forwards all three). In a v0.6.0 breaking release these could be deleted (migration guide already covers them); `mode=` also violates the canonical-kwarg rule. Keeping them gives friendlier errors, so this is a judgment call.  
*Evidence:* `def regress(bd, design_matrix=None, method="ols", mode=None): ... raise NotImplementedError("The regress() method has been removed in v0.6.0...")`  
*Fix (trivial):* Decide explicitly: either remove the shims for the breaking release (rely on the migration guide) or keep them but drop the unused typed kwargs so the signature doesn't advertise a working API.

## Refactor / architecture (8)

### Medium severity

#### ✅ ~~F118 · MEDIUM~~ — **RESOLVED by removal (2026-07-17 discussion session)**
> **`pool.py` removed entirely.** Decision (Eshin): the divergent parser was on dead surface, not two live code paths. `PooledData`/`StatResult`/`ResultDict` had **zero producers** — the only `.pool()` producer (`FittedBrainCollection.pool()`) was deleted in `36bf2695`, and no `.pool()` method exists anywhere in the library; the classes were referenced only by their own unit tests and the internal `pipelines/__init__.py` export. Deleted `nltools/pipelines/pool.py` (~485 LOC) + `nltools/tests/core/test_pipeline_pool.py`, cleaned the package `__init__` exports/docstring, regenerated API docs (pool entries gone). This disposes of F118 (divergent parser), **F111** (broken `repool`), and the F121/F181 `StatResult`/`PooledData.to_nifti` stubs in one move. Two-stage GLM pooling, if wanted later, needs a fresh design against the current `BrainCollection`. Rest of `pipelines/` (`cv`/`base`/`steps`) is live — it backs `BrainCollectionPipeline`. Gate green (`lint` + `lint-api` clean; core/collection/analysis suites 717 passed).

`nltools/pipelines/pool.py:195-225` — **Contrast parsing/applying is duplicated across pool.py and collection/pipeline.py with divergent behavior**  
PooledData._parse_contrast/_apply_contrast (pool.py) and FittedBrainCollection._parse_contrast/_apply_contrast (collection/pipeline.py:605) both parse the same user-facing contrast syntax but with two different regex implementations. pool.py's `([+-])(\w+)` cannot handle scalar coefficients like '2*A', while pipeline.py's parser explicitly supports '2*A-B'. Same documented syntax, two behaviors depending on which entry point the user hits — a single-source-of-truth violation and a latent correctness discrepancy.  
```python
pool.py: pattern = r"([+-])(\w+)"  # no coefficient support
pipeline.py: match = re.match(r"(\d*\.?\d*)\*?(.+)", part)  # supports 2*A
```
*Fix (small):* Extract one pure `parse_contrast(spec, condition_names) -> weights` helper (stats.py/utils.py) and have both PooledData and FittedBrainCollection call it.

### Low severity

#### F006 · LOW · ✅ DONE (2026-07-17)
> `LocalAlignment` is a plain `@dataclass`; replaced the misleading `object.__setattr__(self, "aggregation", "all")` with plain `self.aggregation = "all"`.

`nltools/algorithms/alignment/local.py:399-401` — **object.__setattr__ used to mutate a non-frozen dataclass in __post_init__**  
LocalAlignment is a plain `@dataclass` (not frozen), yet __post_init__ auto-switches aggregation via `object.__setattr__(self, "aggregation", "all")`. On a non-frozen dataclass this is just an obfuscated `self.aggregation = "all"`. object.__setattr__ is the frozen-dataclass idiom and its presence here misleads readers into thinking the class is immutable. CLAUDE.md also prefers frozen dataclasses for state containers; this class holds a lot of mutable fitted state (transforms_, template_, etc.), so either make config frozen and fitted-state separate, or use plain attribute assignment.  
*Evidence:* `object.__setattr__(self, "aggregation", "all")`  
*Fix (small):* Replace with `self.aggregation = "all"` (class is not frozen), or restructure config vs. fitted state so the config portion can be frozen.

#### F014 · LOW · ✅ MOOT (verified 2026-07-17)
> `icc.py` was deleted entirely by the ICC strip — the triplicated code no longer exists. Do not re-open.

`nltools/algorithms/inference/icc.py:234-248` — **ICC formula block duplicated three times (single source of truth violation)**  
The identical icc1/icc2/icc3 mean-square-to-ICC formula (including the invalid icc1 branch above) is copy-pasted in _compute_icc_vectorized (234-248), _compute_icc_gpu_batch (345-359), and _compute_single_icc (393-407). Per the project 'single source of truth' rule this should be one helper taking MSR/MSC/MSE (which broadcast identically for numpy scalars, numpy arrays, and torch tensors). The triplication is exactly why the icc1 statistical error has to be fixed in three places.  
```python
if icc_type == "icc1":
    ICC = (MSR - MSE) / (MSR + (n_sessions - 1) * MSE + EPSILON)
elif icc_type == "icc2": ...
```
*Fix (small):* Extract `_icc_from_meansquares(MSR, MSC, MSE, n_subjects, n_sessions, icc_type)` and call it from all three paths.

#### F061 · LOW · ✅ DONE (2026-07-17)
> `@dataclass(frozen=True)`. Verified safe: the `mean_size`/`min_size`/`max_size` properties compute on the fly (no self-mutation/caching), so freezing breaks nothing. (The audit's "caches derived properties on them" was inaccurate.)

`nltools/data/braindata/neighborhoods.py:41` — **SphereNeighborhoods is a mutable dataclass but holds immutable precomputed state**  
Convention: 'Frozen dataclasses for immutable state containers.' SphereNeighborhoods bundles precomputed adjacency/hash/radius/n_voxels that must not change after construction (it caches derived properties on them), yet it is declared with a bare @dataclass, leaving all fields mutable.  
```python
@dataclass
class SphereNeighborhoods:
    adjacency: sparse.csr_matrix
    mask_hash: str
    radius_mm: float
    n_voxels: int
```
*Fix (trivial):* Use @dataclass(frozen=True). (Sparse matrix field is fine as a frozen reference.)

#### F076 · LOW · ✅ MOOT (verified 2026-07-17)
> `collection/pipeline.py` (`FittedBrainCollection`) was deleted in `36bf2695` — this parser no longer exists. Do not re-open. (The *other* parser divergence is now tracked under the re-framed F118.)

`nltools/data/collection/pipeline.py:605-636` — **Contrast-string parsing duplicated between pipeline.py and execution.py**  
pipeline.py `_parse_contrast` / `_apply_contrast` reimplement the 'A-B'/'2*A-B' parser that already exists as `execution._parse_contrast_string` / `_coerce_contrast`. Two divergent parsers for the same syntax violate single-source-of-truth and will drift (e.g. the execution version validates against regressor_names; this one does not).  
*Evidence:* `parts = re.split(r"(\+|-)", contrast)  # pipeline.py, parallel to execution._parse_contrast_string`  
*Fix (small):* Have pipeline.py import and reuse the execution-layer contrast parser (or a shared helper), especially since FittedBrainCollection is a removal candidate.

#### F154 · LOW · ✅ DONE (2026-07-17)
> Extracted `detect_resolution(affine) -> tuple[float, bool]` (round-to-3-decimals isotropy check; mean fallback when anisotropic) and routed `match_resolution`, `is_standard_space`, `get_bg_image` through it. `is_standard_space` still recomputes the raw `res_array` locally only for its non-isotropic zooms message. Docs regenerated (the new helper appears in `templates.md`).

`nltools/templates/matching.py:57` — **Affine-to-resolution detection duplicated across three functions with divergent rounding**  
`match_resolution` (l.57), `is_standard_space` (l.134), and `get_bg_image` (l.182) each independently recompute `np.abs(np.diag(affine[:3,:3]))` + `np.unique` + isotropy/resolution logic, with the int()-vs-round inconsistency noted above. Exactly the duplicated core logic the functional-core rule says to extract.  
```python
res_array = np.abs(np.diag(affine[:3, :3]))
voxel_dims = np.unique(res_array)   # repeated in 3 functions
```
*Fix (small):* Extract a pure `detect_resolution(affine) -> tuple[float, bool]` helper and call it from all three.

#### F183 · LOW · ✅ DONE (2026-07-17)
> Extracted `_select_corr_func(metric)` and called it from both the CPU-parallel (~191) and observed-correlation (~764) sites.

`nltools/algorithms/inference/correlation.py:191` — **Duplicated metric->corr_func dispatch block appears twice in the same file**  
The identical 4-branch dispatch (pearson/spearman/kendall else NotImplementedError selecting `corr_func`) is written verbatim at lines 191-197 and again at 764-770, violating single-source-of-truth. If a new metric is added it must be edited in two places.  
*Evidence:* `Both sites: `if metric=='pearson': corr_func=_pearson_correlation elif 'spearman'... elif 'kendall'... else raise NotImplementedError(f"Metric '{metric}' not yet implemented")``  
*Fix (trivial):* Extract a `_select_corr_func(metric)` helper and call it from both locations.

#### F192 · LOW · ✅ DONE (2026-07-17)
> Collapsed the 11 methods into one `test_export_is_importable` parametrized over the 10 export names × both public modules (`nltools.stats`, `nltools.stats.permutation`) via `importlib` — same coverage (now asserts each name in BOTH modules), 20 tiny cases instead of 11 hand-written methods.

`nltools/tests/core/test_stats/test_permutation.py:14-56` — **Eleven import-only test methods duplicate a single import check**  
TestStatsPermutationImports has 10 one-line test methods each importing a single name from nltools.stats, plus test_import_from_permutation_submodule which already imports all ten. These 11 near-identical no-behavior tests add collection/reporting overhead with no marginal value.  
```python
def test_import_one_sample(self):
    from nltools.stats import one_sample_permutation_test  # noqa: F401
# ...x10 more identical-shape methods
```
*Fix (trivial):* Collapse to one parametrized test (@pytest.mark.parametrize over the ten names using importlib) or just keep test_import_from_permutation_submodule and delete the singletons.

## Test coverage & streamlining (8)

### High severity

#### F187 · HIGH · ✅ DONE (verified 2026-07-17)
> Already resolved: `nltools/tests/data/roc/test_roc.py` now exists (5 tests). Do not re-open.

`nltools/data/roc/__init__.py:17-394` — **Roc classification class is entirely untested (empty tests/data/roc/)**  
The Roc class (calculate/plot/summary, ~394 LOC) computes sensitivity, specificity, AUC, accuracy and permutation p-values for single-interval and forced-choice classification — the core of MVPA accuracy reporting. There is no test anywhere: tests/data/roc/ contains only __init__.py, and the 'roc' matches in other test files are all lowercase substring hits (correlation, etc.), not Roc usage. All of calculate()'s branches (threshold_type='optimal_overall'|'optimal_balanced'|'minimum_sdt_bias', forced_choice, balanced_acc) are unverified.  
```python
class Roc:
    def calculate(self, input_values=None, binary_outcome=None, criterion_values=None, threshold_type="optimal_overall", forced_choice=None, balanced_acc=False):
```
*Fix (medium):* Add nltools/tests/data/roc/test_roc.py: fit Roc on a separable and a chance-level binary problem, assert AUC≈1.0 / ≈0.5, sensitivity+specificity, accuracy and its permutation p-value; cover each threshold_type and the forced_choice path.

#### F188 · HIGH · ✅ DONE (2026-07-17)
> Added `nltools/tests/stats/test_regression.py` (8 tests): b/se/t/p/df pinned against `scipy.stats.linregress` (independent ref; statsmodels not a dep), 1D→scalar squeeze, 2D multi-target vs column-wise fits, `stats='betas'`/`'tstats'` early returns, near-zero-se t-mask (perfect fit → t held at 0), `method='robust'`→NotImplementedError, invalid `stats`→ValueError. NB the audit evidence shows `mode=`; current code already uses canonical `method=`.

`nltools/stats/regression.py:18` — **Public stats.regress() OLS helper has no direct unit test**  
regress is re-exported from nltools.stats (in __all__) and is the tutorial-facing OLS helper returning (b, se, t, p, df, res). No test imports it directly — the two 'regress' hits in braindata/adjacency tests assert the .regress() facade was *removed*. Its numeric correctness (t-stats, two-tailed p from t_dist, se via pinv(X'X) diag, df, 1D-vs-2D Y squeeze, stats='betas'/'tstats' early returns, the se>1e-6 t-mask edge, and the NotImplementedError for mode!='ols') is completely unverified.  
*Evidence:* `def regress(X, Y, mode: str = "ols", stats: str = "full"):  # __all__ = ["regress"], re-exported by nltools.stats`  
*Fix (small):* Add nltools/tests/stats/test_regression.py comparing b/se/t/p against a known statsmodels/scipy fit, plus cases for 2D multi-target Y, stats='betas'/'tstats', a rank-deficient X (mask branch), and mode='robust' raising NotImplementedError.

### Medium severity

#### F037 · MEDIUM · ✅ DONE (verified 2026-07-17)
> Already resolved: `test_list_of_adjacency_preserves_y_and_labels` covers this (credited F032). Do not re-open.

`nltools/tests/data/adjacency/test_adjacency_core.py:126-129` — **No test that list-based Adjacency construction preserves Y/labels**  
The Y/metadata tests only exercise `adj.append(adj)` (line 126, which does preserve Y) and slicing. There is no test constructing `Adjacency([adj1, adj2])` with non-empty Y/labels and asserting they survive — exactly the path that the constructor bug (finding F001) silently breaks. Existing `Adjacency([...])` usages in tests all build from raw matrices without Y, so the regression is invisible.  
```python
combined = adj.append(adj)
assert combined.Y.shape[0] == 2 * n   # only covers append(), not Adjacency([adj, adj])
```
*Fix (trivial):* Add a test: give two single-matrix Adjacency objects Y frames + labels, build `Adjacency([a, b])`, assert combined.Y has both rows and labels are retained.

#### F134 · MEDIUM · ✅ DONE (verified 2026-07-17)
> Already resolved: `nltools/tests/plotting/test_f123_prediction.py` covers all 4 funcs (plot_roc/plot_scatter/plot_probability/plot_dist_from_hyperplane). Do not re-open.

`nltools/plotting/prediction.py:1-14` — **prediction.py has zero test coverage, which is why the seaborn-0.13.2 breakage went unnoticed**  
No test references plot_roc/plot_scatter/plot_probability/plot_dist_from_hyperplane (grep of nltools/tests returns nothing). Two of the four are currently broken on the pinned seaborn. A single Agg-backend smoke test per function would have caught both critical failures.  
*Evidence:* `rg -ln 'plot_roc|plot_scatter|plot_probability|plot_dist_from_hyperplane' nltools/tests/  -> (no matches)`  
*Fix (small):* Add a plotting/test_prediction.py with matplotlib.use('Agg') that calls each function with a tiny DataFrame and asserts a figure/grid is returned.

#### F189 · MEDIUM · ✅ DONE (2026-07-17)
> Added 4 more fast tests: `test_sphere_builds_binary_region`, `test_gaussian_normalizes_to_total_intensity`, `test_create_cov_data_single_subject`, `test_create_cov_data_multi_subject_concats`. **Coverage surfaced a real bug**: `create_cov_data` was dead-on-arrival for its documented/default single-3-D-mask usage — `apply_mask(mask, brain_mask)` returns a 1-D vector, but the body indexes `flat_sphere.shape[1]` / `np.where(...)[1]` assuming 2-D (`create_ncov_data` avoids this by wrapping its masks in a list → 2-D result). Fixed with `np.atleast_2d(apply_mask(...))` (one line). The two create_cov tests carry a `filterwarnings` for the inherent non-PSD-covariance RuntimeWarning (the function doesn't constrain the covariance matrix; out of scope here).

`nltools/data/simulator/__init__.py:191-510` — **Simulator has near-zero coverage and all its tests are slow-marked (default run exercises nothing)**  
Only Simulator.create_data is touched by a test; create_cov_data, create_ncov_data, n_spheres, gaussian, and sphere are untested. Worse, every test in test_simulator.py is @pytest.mark.slow, and the default suite runs '-m not slow', so the entire 839-LOC simulator module (used to generate ground-truth signal for other tests) has no coverage in normal CI iteration.  
```python
@pytest.mark.slow
def test_simulator(tmpdir): ...  # + test_simulategrid_fpr, test_simulategrid_fdr all slow; only create_data covered
```
*Fix (small):* Add a fast (non-slow) smoke test for create_cov_data/create_ncov_data/n_spheres asserting output shapes and the injected covariance/signal structure, so the module is exercised by the default run.

#### F190 · MEDIUM · ✅ DONE (2026-07-17)
> Added `TestTrim` (3 tests) to `test_outliers.py`: quantile trim nulls both extremes (2 nulls, incl. the 1053 outlier) and crucially leaves every surviving value UNCHANGED (the null-vs-clamp distinction from winsorize), std trim nulls only 1053, and a Series input returns a Series.

`nltools/stats/outliers.py:70` — **stats.trim() untested — its null-replacement branch is never exercised**  
trim() is public (re-exported by nltools.stats) but has no direct test; test_outliers.py only covers winsorize and zscore. The trim-specific branch in _transform_outliers (method=='trim' replacing out-of-cutoff values with null/NaN rather than clamping) is therefore never executed by any test, including for both the DataFrame and Series return paths and std vs quantile cutoffs.  
```python
def trim(data, cutoff=None):
    return _transform_outliers(data, cutoff, replace_with_cutoff=None, method="trim")  # 0 tests reference trim
```
*Fix (trivial):* Add trim tests in test_outliers.py mirroring the winsorize cases: assert out-of-cutoff entries become null (not clamped) for both std and quantile cutoffs, and that a Series input returns a Series.

#### F191 · MEDIUM · ✅ DONE (2026-07-17)
> Removed all 3 wall-clock assertions and replaced them with structural checks. `test_method_chaining_efficiency` (default-run) now asserts shared mask + independent data buffer. The two slow-marked timing-ratio tests became fast structural tests: `test_comparison_with_deepcopy`→`test_shallow_vs_deepcopy_sharing` (shallow shares data buffer, deepcopy duplicates it; mask is a shared immutable resource in both — deepcopy does NOT clone it), and `test_transform_methods_efficient` (shared mask, independent buffer, source unmutated). Dropped the now-unused `time` import.

`nltools/tests/support/test_efficient_copy.py:63-72` — **Wall-clock timing assertions are flaky; one runs in the default suite**  
test_method_chaining_efficiency (not slow-marked, runs by default) asserts wall-clock 'elapsed < 1.0' on tiny data — spuriously fails under a loaded machine or heavy pytest -n auto parallelism. test_comparison_with_deepcopy (line 129: shallow_time < deep_time*0.5) and test_transform_methods_efficient (line 235: transform_time < deep_copy_time*0.75) assert relative timing ratios on small arrays where timing is noise-dominated and can invert. These test performance via clocks rather than behavior.  
```python
start = time.time()
result = sim_brain_data.scale(100.0).standardize()
elapsed = time.time() - start
assert elapsed < 1.0, ...
```
*Fix (small):* Drop the wall-clock assertions; verify efficiency structurally instead (e.g. assert result.mask is original.mask / result.data is not original.data, and that no deepcopy of data occurred), or gate the ratio checks behind a much larger array with a wide safety margin.

### Low severity

#### F193 · LOW · ✅ DONE (2026-07-17)
> `make_cosine_basis`: added column-count-vs-filter-length (fl=128→2, fl=32→8 for n=128), near-orthogonality (off-diagonal Gram <1e-8), and drop-removes-leading-columns (drop=2 → shape−2, equals full[:, 2:]) tests. `calc_bpm`: added a not-hardcoded test asserting 60·sf/interval across (1000,1000)→60, (500,1000)→120, (2,1)→30.

`nltools/tests/stats/test_timeseries.py:68-77` — **make_cosine_basis / calc_bpm tested for shape only, not numeric correctness**  
make_cosine_basis is verified only via basis.shape[1] >= 1 — it never checks the number of basis functions for a given filter_length, DCT orthogonality, or the drop/unit_scale parameters. calc_bpm has a single 72-BPM case. These are the only two tests for each function, so their actual math is essentially unverified.  
```python
basis = make_cosine_basis(n_timepoints, sampling_freq=1, filter_length=128, drop=0)
assert basis.shape[0] == n_timepoints
assert basis.shape[1] >= 1
```
*Fix (small):* Assert the expected basis column count for a known filter_length, near-orthogonality of columns, and that drop>0 removes the expected low-frequency columns; add a second calc_bpm interval to confirm the mapping is not hard-coded.

## Type safety (5)

### Medium severity

#### F107 · MEDIUM · ✅ DONE (2026-07-17)
> Annotated base.py/glm.py/ridge.py: `__init__` params, the four data-facing methods (fit/predict/score/predict), helpers, and properties. `from __future__ import annotations` in all three; TYPE_CHECKING imports for `Backend` (ridge), `pd`/`DesignMatrix` (glm). **LSP note**: Glm (Nifti) and Ridge (2-D array) operate on different modalities, so no single tight param type is Liskov-substitutable — `BaseModel`'s abstract fit/predict/score keep params **unannotated** (comment added) with return types broadened (`predict -> np.ndarray | list`, `score -> float | np.ndarray`) to cover both. `Glm.predict` DataFrame coercion rewritten via `getattr(X, "to_numpy", None)` so ty's callable check stays off the typed ndarray branch.

`nltools/models/base.py:1` — **Entire models layer has no type annotations**  
None of base.py, glm.py, ridge.py annotate any parameters or return types (types live only in docstrings), while the algorithms layer they wrap (solvers.py) is fully annotated with modern | unions. CLAUDE.md mandates modern Python with type hints. This is a consistency and maintainability gap on the public model API surface.  
*Evidence:* `def fit(self, X, y):  /  def predict(self, X=None):  /  def __init__(self, alpha=1.0, cv=None, ...):  — no annotations anywhere`  
*Fix (medium):* Add type hints to public signatures (e.g. fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> 'Ridge'), at minimum on the four data-facing methods and the __init__ params.

### Low severity

#### F009 · LOW · ✅ DONE (2026-07-17)
> `parcellation` and `mask_` retyped `nib.Nifti1Image | None` (nib already imported under TYPE_CHECKING; `from __future__ import annotations` in effect). Dropped the now-unused `Any` import.

`nltools/algorithms/alignment/local.py:366` — **parcellation/mask typed as Any despite nib.Nifti1Image being known**  
`parcellation: Any | None` (line 366) and `mask_: Any | None` (line 384) are annotated Any with `# Nifti1Image` comments, even though the module already imports `nibabel as nib` under TYPE_CHECKING (line 20). Using Any loses type checking on the most error-prone inputs.  
*Evidence:* `parcellation: Any | None = None  # Nifti1Image`  
*Fix (trivial):* Annotate as `nib.Nifti1Image | None` (works because the annotations are string/deferred via `from __future__ import annotations`).

#### F017 · LOW · ✅ DONE (2026-07-17)
> Widened hint to `tail: int | str = 2`, added an upfront `validate_tail_parameter(tail)` call (matching one_sample/two_sample/matrix) so an invalid value fails immediately instead of after every permutation, and documented all three int/str forms + one-tailed direction in the docstring.

`nltools/algorithms/inference/timeseries.py:595` — **timeseries test declares tail: int but no upfront validation; str/-1 only fail deep inside**  
`timeseries_correlation_permutation_test` hints `tail: int = 2` (docstring: '1=one-tailed, 2=two-tailed') and, unlike the sibling tests, performs no upfront `validate_tail_parameter`. Valid string forms accepted elsewhere ('upper'/'lower') actually work here because they reach `_compute_pvalue`, but an invalid value only errors after all permutations run, and the type hint understates the accepted inputs. Direction of the one-tailed test is also undocumented.  
```python
tail: int = 2,
...
# (no validate_tail_parameter call before the permutation loop)
```
*Fix (trivial):* Widen the hint to `tail: int | str = 2`, call `validate_tail_parameter(tail)` up front like one_sample/two_sample/matrix, and document the one-tailed direction.

#### F046 · LOW · ✅ DONE (2026-07-17)
> Added `from matplotlib.figure import Figure` under TYPE_CHECKING; `ClusterReport.plot` return + local `figures` list retyped to `list[tuple[str, "Figure"]]`, matplotlib import stays lazy. Dropped the unused `Any` import.

`nltools/data/atlases/reporting.py:58` — **ClusterReport.plot uses Any for what is known to be matplotlib Figure objects**  
The return type is `list[tuple[str, Any]] | None`, and the docstring explicitly documents the second element as `matplotlib.figure.Figure`. `Any` erases that known type from the public signature. matplotlib is imported lazily inside the method (intentional), but the type is still knowable.  
*Evidence:* `def plot(self, *, output_dir: str | Path | None = None) -> list[tuple[str, Any]] | None:`  
*Fix (trivial):* Import Figure under `if TYPE_CHECKING:` (`from matplotlib.figure import Figure`) and annotate as `list[tuple[str, "Figure"]] | None`, keeping the runtime import lazy.

#### F085 · LOW · ✅ DONE (2026-07-17)
> Added `-> "Figure"` to `DesignMatrix.plot` (Figure imported under TYPE_CHECKING); widened `append`'s `fill_na` to `int | float | None` matching the impl + updated its docstring to mention None preserves nulls.

`nltools/data/designmatrix/__init__.py:587` — **plot() facade lacks a return type annotation; append() fill_na hint too narrow**  
`DesignMatrix.plot(...)` has no `->` return annotation despite its docstring promising a matplotlib Figure, unlike sibling methods which annotate `-> DesignMatrix`. Separately, the `append` facade types `fill_na: int | float = 0` while the underlying append() and its docstring accept `None` to preserve nulls — the facade hint wrongly rejects a documented, working value.  
*Evidence:* `def plot(self, method='matrix', *, ...):  # no -> annotation   |   def append(self, ..., fill_na: int | float = 0, ...)`  
*Fix (trivial):* Add `-> matplotlib.figure.Figure` (or `-> 'plt.Figure'` under TYPE_CHECKING) to plot; widen append's fill_na hint to `int | float | None` to match the implementation and docstring.

## Performance (1)

### Low severity

#### F045 · LOW · ✅ DONE (2026-07-17)
> Hoisted `world_xyz = nb_affines.apply_affine(affine, ijk)` above the `for atlas in atlas_objs` loop in `_build_clusters_dataframe` and reused it in each `_xyz_to_ijk`. Pure perf hoist (atlas-independent transform now computed once per cluster instead of once per atlas); reporting suites green (no behavior change).

`nltools/data/atlases/reporting.py:258-263` — **Cluster voxel ijk->world mapping recomputed once per atlas inside the atlas loop**  
In `_build_clusters_dataframe`, `nb_affines.apply_affine(affine, ijk)` maps a cluster's voxel coordinates into world (mm) space, which is independent of the atlas. It is recomputed inside the `for atlas in atlas_objs` loop, so for the default trio of atlases the same (n_vox, 3) transform is done 3x per cluster.  
```python
for atlas in atlas_objs:
    atlas_ijk = _xyz_to_ijk(
        nb_affines.apply_affine(affine, ijk), atlas.image.affine
    )
```
*Fix (trivial):* Hoist `world_xyz = nb_affines.apply_affine(affine, ijk)` above the atlas loop and reuse it in each `_xyz_to_ijk` call.

## UX & surprising behavior (7)

> **Partially closed — 2026-07-17 (signature-lies session).** ✅ **F068** (roi_mask implemented; radius_mm/device/n_jobs/progress_bar removed — was a silent correctness trap, misfiled as UX), ✅ **F021** (n_jobs removed from the 3 ridge solvers), ✅ **F182** (docstring made honest + Raises:; X kept for the BaseModel contract; real new-design prediction deferred, needs Eshin's call). Commits `d2a77e27` + `9c96c1da`.
> Still open here: **F098** (Roc `if not any(binary_outcome)` with a "may not be boolean" message), **F031** (ridge `n_iter` overloaded int-or-array; rejects np.int64), **F159** (h5 `to_h5` raises TypeError for a bad `obj_type` *value* → should be ValueError), **F157** (`collapse_mask` undocumented `auto_label`, silent None returns, "collapased" typo).

### High severity

#### F068 · HIGH · ✅ FIXED (2026-07-17, `d2a77e27`)
> `roi_mask` implemented (scopes the computation; output maps carry the ROI mask); `radius_mm`/`device`/`n_jobs`/`progress_bar` removed. Correctness trap — was returning whole-brain maps for `roi_mask=atlas` with no warning.

`nltools/data/collection/inference.py:443-496` — **isc and isc_test silently ignore roi_mask, radius_mm (and device/n_jobs/progress_bar)**  
Both functions accept `roi_mask` and `radius_mm` in their signatures (and the facade forwards them), but the bodies never reference them — grep confirms none of roi_mask/radius_mm/device/n_jobs/progress_bar appear in the isc/isc_test bodies. A user who passes `roi_mask=atlas` expecting ROI-restricted or searchlight ISC silently gets whole-brain ISC with no warning, a correctness trap in published analyses. The isc docstring doesn't even mention roi_mask/radius_mm.  
*Evidence:* `def isc(bc, *, method='loo', roi_mask=None, radius_mm=6.0, metric='median', device='cpu', n_jobs=-1, progress_bar=False):  # body uses only method/metric`  
*Fix (medium):* Either implement the ROI/searchlight scoping (preferred) or raise NotImplementedError when roi_mask/radius_mm are non-default, so the surface doesn't lie. Remove or wire up device/n_jobs/progress_bar likewise.

### Medium severity

#### F021 · MEDIUM · ✅ FIXED (2026-07-17, `d2a77e27`)
> `n_jobs` removed from all 3 solvers (signatures + docstrings). Removed rather than implemented — ridge acceleration is the vectorized/GPU backend, and the CV/gamma loops are serial by design. No caller ever passed it.

`nltools/algorithms/ridge/solvers.py:50, 636, 911` — **n_jobs is accepted and documented as CPU parallelization but is never used**  
solve_banded_ridge_cv, solve_ridge_cv, and cross_val_predict_ridge all declare `n_jobs: int = -1` and document it as 'Number of CPU cores for parallelization (-1 = all cores)'. grep shows n_jobs is referenced only in the signatures and docstrings — the CV/gamma loops are fully serial and no joblib/threadpool is ever invoked. The facade (models/ridge.py) never forwards n_jobs either. A user setting n_jobs to tune parallelism gets a silent no-op, contradicting the documented behavior.  
*Evidence:* `n_jobs: int = -1,  # signature only; body never reads n_jobs`  
*Fix (small):* Either implement parallelization over folds/gammas using n_jobs, or drop the parameter and remove the misleading docstring lines. If kept for signature symmetry, document it explicitly as 'currently unused' like random_state is.

#### F098 · MEDIUM · ✅ DONE (2026-07-17)
> Coerce `binary_outcome` to bool once up front, then require BOTH classes present (`.all() or not .any()` → raise) with an accurate message ("must contain both positive and negative cases"). Reuses the coerced array for `self.binary_outcome`.

`nltools/data/roc/__init__.py:48-49` — **Misleading validation: `if not any(binary_outcome)` with message 'may not be boolean'**  
This raises when binary_outcome has no truthy element (all zeros/False), but the message says 'binary_outcome may not be boolean'. The check neither validates booleanness nor detects a non-boolean dtype; it just requires at least one positive case. A user passing a valid all-negative vector gets a confusing error, and a genuinely non-boolean array (e.g. floats 0.3/0.7) passes silently and then breaks at boolean indexing.  
```python
if not any(binary_outcome):
    raise ValueError("Data Problem: binary_outcome may not be boolean")
```
*Fix (small):* Validate dtype explicitly (coerce to bool, require both classes present) with an accurate message, e.g. 'binary_outcome must contain both True and False'.

#### F182 · MEDIUM · ✅ DONE (2026-07-17 functional-GLM session, `99140652` + `3f8e809f` + `7fb76b4b`; groundwork `103d95bf`)
> **IMPLEMENTED — new-design GLM prediction now works.** The scoping below is retained as the design record; every "what to build" item was executed. Summary of the landed solution:
> - `Glm.coef_` — `(n_regressors, n_voxels)` betas assembled from `run_glm`'s `labels_`/`results_` `theta` (NOT via per-regressor identity contrasts; theta is the direct GLS estimate). No unmask.
> - `Glm.predict(X)` → `X @ coef_`, a 2-D ndarray, Ridge-parity. `predict()` (no arg) still returns nilearn fitted-values. Multi-run `predict(X)` raises with a per-run-fit message (the "single X ambiguous across runs" concern).
> - `nltools/data/braindata/prediction.py` — the GLM `NotImplementedError` fork is gone; GLM new-X now routes through `model_.predict(X)` exactly like Ridge, returning a `BrainData`.
> - `nltools/tests/models/test_f182_glm_predict.py` — rewritten: the raise-pin is gone, replaced by coef_/predict parity + report tests.
> - Enabling groundwork (`103d95bf`): `Glm` sets `signal_scaling=False` so scaling is explicit, decoupling betas from nilearn's implicit per-voxel scaling. (See the preprocessing-redesign session note near the top.)
> - AR handled: contrasts/predict stay correct in-memory via nilearn's per-voxel covariance; `BrainCollection` refuses AR (OLS-only closed form) rather than approximate.
>
> _Original surface fix (superseded, `d2a77e27`):_ Docstring made honest (states X is unsupported + Raises: section + points to working alternatives); the loud raise stays. **X is KEPT** — `BaseModel.predict(X)` is abstract with X required, so dropping it breaks the contract (unlike F185). Fails loudly so it cannot corrupt an analysis.
>
> **Eshin's call (2026-07-17): this is a real implementation gap, not a wontfix.** `BrainData` fits both `model='glm'` and `model='ridge'`; the end-user `.predict()` experience should feel the same regardless of which model was fit. Ridge already supports new-design prediction; GLM does not — that asymmetry is the gap to close. Scoping below; not implemented this session (deferred FEATURE, needs a design decision on multi-run semantics). **Recommend filing as a Linear issue in project `nltools`.**
>
> **Scoping note — GLM vs Ridge `.predict()` parity**
>
> _Two distinct predict surfaces — don't conflate them:_
> - `BrainData.predict(...)` (`data/braindata/__init__.py:1344`) defaults to **MVPA decoding** (`model='svm'`, dispatched on `y=`). The encoding/"run the fitted model" flow is the **timeseries path**: `predict(X=...)` with no `y` → `prediction.predict` → `prediction.predict_timeseries` → `model_.predict(...)`.
> - The model-layer `Glm.predict()` / `Ridge.predict()` are what actually diverge.
>
> _Where the asymmetry is enforced (two hard sites + one pinning test):_
> - `nltools/models/glm.py:246` — `Glm.predict(X)` raises `NotImplementedError` for any non-None X; `predict()` (no arg) returns `self._glm.predicted` (list of `Nifti1Image` fitted values).
> - `nltools/data/braindata/prediction.py:127-136` — `predict_timeseries` forks on `isinstance(bd.model_, Glm)`: GLM → fitted-values-only + `NotImplementedError` for new X; Ridge → `bd.model_.predict(X)` unconditionally.
> - `nltools/tests/models/test_f182_glm_predict.py` — deliberately pins the current gap (asserts the raise + docstring wording); **must be rewritten when implemented.**
>
> _Why Ridge works (the reference UX):_ `Ridge` persists `coef_` (+ `intercept_`); `predict(X)` = `X @ coef_ + intercept_` (`models/ridge.py:290-314`), returns an ndarray, X required — matches the `BaseModel.predict(X)` contract literally.
>
> _Why GLM doesn't:_ backend is nilearn `FirstLevelModel` (via `self._glm`), which **exposes no betas** and returns fitted values as a list of Nifti images. `Glm` stores no `coef_`.
>
> _What to build for parity (per the code's own comments):_
> 1. **Recover a β matrix** `(n_regressors, n_voxels)` — nilearn holds none, but per-regressor identity contrasts recover them. **The machinery already exists**: `modeling.fit_glm` (`data/braindata/modeling.py:508-540`) already loops design columns building one-hot contrasts via `compute_contrast(..., output_type='all')`. Cache the β matrix at fit time, mirroring `ridge.coef_`.
> 2. **Define `Glm.predict(X)` = `X @ betas`** and settle the return container (list of `Nifti1Image` to match the training path, or a 2D array to match Ridge — a UX decision).
> 3. **Resolve multi-run ambiguity** — `Glm.fit` accepts a list of per-run design matrices; a single new X is ambiguous across runs. Need a per-run API or a documented single-run restriction.
> 4. **Reconcile the GLM scaling story** — BrainData grand-mean-scales before GLM fit (`modeling.py:203-206`) and GLM designs usually carry an explicit constant/drift column, so "predicted" for a new X must be defined relative to that scaling.
> 5. Feature-width validation for the GLM branch already exists (`prediction.py:119-125`). On implementation, remove both `NotImplementedError` sites and rewrite `test_f182_glm_predict.py`.
>
> _Coverage gap worth noting:_ the encoding `BrainData.predict(X=new_design)` path is **untested for both models** (Ridge new-X is only tested at the model layer, not through the facade; GLM is only negatively pinned).

`nltools/models/glm.py:214-247` — **Glm.predict(X=...) documents and accepts an X design matrix but raises NotImplementedError for it**  
The `predict` method accepts `X` and its docstring describes 'If DataFrame: generates predictions using new design matrix', but any non-None X raises NotImplementedError. A documented, accepted kwarg that always errors is a deferred gap that will surprise users who follow the docstring.  
```python
def predict(self, X=None): ... if X is None: return self._glm.predicted
        raise NotImplementedError("Prediction with new design matrix not yet implemented.")
```
*Fix (medium):* Either implement `betas @ X.T` prediction (the code comment already sketches it), or make the docstring unambiguous that X is not yet accepted and consider dropping the param until it works.

### Low severity

#### F031 · LOW · ✅ DONE (2026-07-17)
> Kept the `n_iter` name (documented F105 exception — random-search iteration count, no rename in this breaking release), but fixed the numpy-int rejection: `isinstance(n_iter, int)` → `isinstance(n_iter, numbers.Integral)` (accepts np.int64), hint widened to `int | np.integer | np.ndarray`, and a comment documents the dual int-count/gamma-array meaning. Did NOT split into two params (that would fight the F105 decision).

`nltools/algorithms/ridge/solvers.py:31` — **n_iter parameter is overloaded (int count OR gamma array) and its name conflicts with API naming prefs**  
solve_banded_ridge_cv's `n_iter` accepts either an int (number of random-search samples) or a 2D ndarray of explicit gamma weights. Passing an array to a parameter named 'n_iter' is surprising, and the type check `isinstance(n_iter, int)` also rejects numpy integer scalars (np.int64) which then fall through to 'Unknown parameter n_iter'. The name also sits awkwardly against the project's canonical count names (n_permute/n_samples).  
*Evidence:* `if isinstance(n_iter, int): ... elif isinstance(n_iter, np.ndarray) and n_iter.ndim == 2: gammas = n_iter ... else: raise ValueError(f"Unknown parameter n_iter={n_iter!r}")`  
*Fix (small):* Split into two explicit params (e.g. `n_samples`/`n_gammas` for the count and a separate `gammas=` array), or at least accept numpy integers (use numbers.Integral) and document the dual meaning prominently.

#### F159 · LOW · ✅ DONE (2026-07-17)
> `TypeError` → `ValueError` (bad string *value*, not a type error). Updated the pinning test `test_invalid_obj_type_raises` to expect ValueError.

`nltools/io/h5.py:96-97` — **to_h5 raises TypeError for a bad obj_type string value**  
`obj_type` is a string value, so an invalid value is a value error, not a type error. Raising TypeError is semantically misleading for callers catching by exception type.  
*Evidence:* `raise TypeError("obj_type must be one of 'brain_data' or 'adjacency'")`  
*Fix (trivial):* Raise ValueError instead.

#### F157 · LOW · ✅ DONE (2026-07-17)
> Documented `auto_label` (+ tightened `mask`/`custom_mask` Args, added Raises:), fixed the "collapased" typo (removed with the warning line), and replaced BOTH silent `None` returns (single-mask warn+None AND the multi-dim-but-single-image no-warning None) with one upfront `ValueError` when there are fewer than 2 masks — the function now always returns a BrainData or raises, matching its documented Returns. Dropped the now-unused `warnings` import. No callers/tests depended on the None/warning behavior.

`nltools/mask.py:126-187` — **collapse_mask has undocumented auto_label param, silent None returns, and a typo**  
`auto_label` is a parameter not in the docstring Args (only mask/custom_mask are). The function returns BrainData in the multi-mask case but implicitly returns None both for a single mask (with warning) and when `len(mask.shape) > 1` but `len(mask) <= 1` (no else) — inconsistent/silent return type. Warning text misspells 'collapased'.  
```python
warnings.warn("Doesn't need to be collapased")
...
return None
```
*Fix (small):* Document `auto_label`, fix the typo, and return a consistent type (or raise) in the degenerate branches.

## Per-subsystem health notes

- **algorithms-alignment** (10): The three algorithms are mathematically sound (Procrustes/SVD updates, SRM EM and DetSRM BCD look correct and are adapted faithfully from brainiak), and the parallelization plumbing works. The most important issue is HyperAlignment's `auto_pad`, which silently truncates voxels to the smallest subject while its docstring promises zero-padding — a real data-loss/behavior-mismatch risk. Secondary concerns are API-convention drift: transform methods ignore their own `n_jobs` argument, LocalAlignment forces tqdm on with no `progress_bar` control (making the facade's progress_bar argument dead), and public estimator methods omit the mandated keyword-only `*` marker. local.py's docstrings are numpydoc-style and will mis-render under the Google-style griffe pipeline. No critical correctness bug in the core linear algebra, but the auto_pad truncation and the dead n_jobs/progress_bar parameters should be fixed before a breaking release.
- **algorithms-inference** (9): The inference package is generally solid, well-documented, and internally consistent in its CPU/GPU/parallel dispatch pattern. Two findings matter for release: (1) correlation_permutation_test hand-rolls `if tail not in [1,2]` and thus rejects the very tail values ('two'/'upper'/'lower'/-1) its own docstring recommends -- a real crash and the only module not using the shared validate_tail_parameter; and (2) ICC1 is computed with the ICC3 error term (MSE) rather than the pooled within-subject mean square, so it silently returns ICC3 values despite the cited Shrout & Fleiss one-way model (this is tested-as-intended but statistically wrong and worth reconciling). A secondary correctness issue: isc_group_permutation_test's bootstrap 'ci' is computed from the observed-centered null, so it brackets 0 instead of the estimate, unlike the correct isc_permutation_test. The rest are low-severity polish: triplicated ICC formula (single-source-of-truth), a dead shadowing assignment, circle_shift's zero-shift/1D-vs-2D inconsistency, and some docstring/type-hint tidy-ups.
- **algorithms-ridge-misc** (12): The ridge package is well-documented and mostly Google-style compliant, but carries several release-blocking rough edges. The most serious is a latent NaN-corruption bug in solve_banded_ridge_cv's in-place feature scaling: when a Dirichlet gamma component is exactly 0 (which np.random.dirichlet with low concentration can produce), the restore step divides X by sqrt(0), poisoning X for every subsequent random-search iteration. A cluster of advertised-but-dead parameters (n_jobs, max_gpu_memory_gb) mislead users into expecting parallelism/memory control that the code never implements. There is duplicated dead code (batch_or_skip in two places, both uncalled) and minor API-consistency drift (missing keyword-only markers, inconsistent return-dict keys, inconsistent parallel defaults). Docstrings are strong overall with only small param-list gaps. Backend abstraction and the random/shape/hrf helpers are clean.
- **braindata-core** (10): The BrainData facade + core modeling/prediction/bootstrap layers are largely well-structured: logic lives in module-level functions and the class delegates cleanly, docstrings are Google-style with no RST leakage, and result containers (Fit/Predict/BrainData) are consistent. The most serious issue is a real crash: filter_data double-passes detrend/standardize to nilearn.signal.clean, so the documented filter(detrend=True) usage raises TypeError. Beyond that, the main weaknesses are API-consistency drift against the v0.6.0 conventions (icc exposes internal parallel= with a no-op n_jobs default and diverges from bootstrap's backend= vocabulary; several public methods with 3+ kwargs lack the required '*' marker; predict hardcodes random_state=42 with no override) and two silent-correctness edges (fit(inplace=False) still mutates bd.model_/X_ contradicting its docstring; searchsorted on model.alphas breaks for unsorted alpha grids). None of the correctness edges affect default-path results except the filter crash, which should block release until fixed.
- **braindata-io-plot** (9): The IO/plotting/viewer/cache/utils/validation layer is generally well-structured and adheres to the functional-core/imperative-shell split, with clear docstrings largely free of RST leakage and good keyword-only discipline in the newer surfaces (iplot, plot_surf, build_viewer). The most material issue is a real crash path in upload_neurovault: a failed create_collection leaves `collection` unbound and triggers an UnboundLocalError. Secondary correctness concerns cluster around ad-hoc temp-dir handling in io.py (a leak in load_from_url plus a collision-prone os.times()-based naming scheme). Remaining items are polish: missing keyword-only markers on the plot()/plot_flatmap()/resample_to() facades, a non-frozen state dataclass, an unused parameter, and minor docstring/security nits. No data-corruption or wrong-math bugs found in the core masking/resampling paths.
- **data-adjacency** (10): The Adjacency facade and submodules are largely functional-core compliant (logic lives in stats.py/modeling.py/utils.py; the class delegates), and canonical names like n_permute/n_samples/include_diag/n_jobs are used correctly in most places. The biggest risk is a silent data-loss bug in the constructor: building an Adjacency from a list of Adjacency objects concatenates their data but then overwrites the concatenated Y (and never carries labels), so metadata is dropped without error — and there is no test covering this path. Secondary issues are canonical-kwarg violations around the reserved name `metric` (misused for mean/median aggregation in cluster_summary and for a bool in plot_mds), a zero-value falsy bug in threshold, and systematically missing keyword-only `*` markers on 3+-kwarg public methods.
- **data-atlases** (5): The atlas subsystem is generally clean, well-typed, and follows the functional-core/imperative-shell convention: registry.py/loading.py/labeling.py are pure and cohesive, module docstrings are present and Google-style, and public signatures respect the keyword-only and kwarg conventions. The serious risk is in reporting.py: `_build_peaks_dataframe` crashes on any cluster that has sub-peaks (nilearn emits empty-string cluster sizes for sub-peak rows), which is the common case in real stat maps and is completely missed by the synthetic single-peak tests. A second, quieter correctness/UX hazard is that the `peaks` and `clusters` DataFrames use two unrelated `cluster_id` spaces (nilearn peak-stat order + string dtype vs. size order + int dtype), so users cannot reliably relate the two tables. Remaining items are minor (a stale internal docstring, a small redundant recompute).
- **data-collection** (12): The BrainCollection facade + core/execution/io layer is generally well-structured and adheres to the functional-core/imperative-shell split, with pure reductions in inference.py and clean worker plumbing in execution.py. The most serious issues are correctness: isc_test computes a statistically invalid p-value (bootstrap null is centered on the observed ISC, not 0), and BrainCollectionPipeline.n_subjects calls a nonexistent BrainCollection.n_images (crashes on property/repr access). A recurring hygiene problem is accepted-but-ignored kwargs on the inference surface — isc/isc_test silently drop roi_mask/radius_mm, and align ignores cache/progress_bar — which mislead users about what actually ran. pipeline.py is the weakest file: it carries dead code (FittedBrainCollection is never instantiated), numpydoc/RST docstrings, stale API examples, and a banned algorithm= kwarg; it reads like legacy that should be pruned for the breaking v0.6.0 release. execution.py, io.py, and core.py are the cleanest of the set.
- **data-designmatrix** (8): The DesignMatrix module is well-structured and cleanly follows the functional-core/imperative-shell pattern — the facade delegates to pure standalone functions in append/diagnostics/io/plotting/regressors/transforms, metadata propagation via copy_with is careful, and docstrings are largely Google-style (no RST leakage found). The main weaknesses are API-convention drift rather than correctness bugs: a missing keyword-only marker on clean(), `verbose` used as a print-gate where the convention reserves it for log-level, and internal `**kwargs` forwarding across the downsample/upsample nltools->nltools boundary. Real correctness risks are minor and edge-case (non-integer downsample ratios, incomplete dtype-compatibility checking). The most user-visible issue is the stale 'non-polynomial columns' docstring wording, which understates that all confounds (motion/physio included) are excluded from default z-score/standardize/convolve.
- **data-misc** (17): Roc and Simulator are legacy imperative classes carrying real defects, while FitResults (Fit/Predict) is a clean frozen-dataclass pair whose main issues are stale/RST docstrings. The most serious problems: Simulator.__init__ uses `~isinstance(...)` (bitwise, always-truthy) so passing a valid nibabel mask always raises — a critical, untested regression; Roc.accuracy_se computes p*p/n instead of p*(1-p)/n (wrong proportion SE); and create_ncov_data has a copy-paste `type(cor)` check that should be `type(cov)`. Beyond correctness, both files violate v0.6.0 conventions: unused/unforwarded **kwargs, non-canonical variant kwargs (plot_method/threshold_type vs method), leftover array-dumping debug prints, a never-wired _run_permutation path, and multiple broken docstring examples (create_data y=/n_reps=, SimulateGrid.fit(n_permute=...), Predict's method= vs the shipped spatial_scale). None of the three files is functional-core clean, but Roc/Simulator predate the refactor and should be the audit's priority before release.
- **models** (8): The model layer is a thin, sensibly-structured sklearn-style facade over the algorithms layer, and the core numeric logic (intercept centering, R2, banded/CV dispatch) is correct and well-commented. The main pre-release gaps are API-hygiene rather than math bugs: Glm.fit and both __init__ methods omit the mandated keyword-only '*' marker (Glm.fit's omission is a real footgun that lets a design matrix bind to the unused y arg), Ridge surfaces the explicitly-banned n_iter kwarg through the facade's **kwargs forwarding, and the whole layer lacks type annotations despite the code it wraps being fully typed. Docstrings reference a nonexistent 'GLMModel' class in four places (including a broken import example). No critical correctness or data-loss issues found.
- **pipelines** (12): The pipelines module is NOT dead code — cv.CVScheme, FittedStack, the transform steps (Normalize/Reduce/Pipe), and pool (PooledData/StatResult/ResultDict) are all still wired in through BrainCollection.cv() and FittedBrainCollection.pool(). However, the removal of the standalone Pipeline orchestrator left real orphans and bugs: the Terminal protocol and a duplicate base.CVScheme Protocol are dead abstractions, AlignStep/FittedAlign have no facade builder and are only reachable in tests, and PooledData.repool() plus StatResult.to_nifti are non-functional (repool can never match the fitted_state shape the facade actually produces; to_nifti is a raising stub). The most serious correctness issue is the permutation CV scheme, which abuses the (train_idx, test_idx) contract and yields an invalid null when driven through the facade. Contrast parsing is duplicated with divergent behavior between pool.py and the collection facade. Docstrings are mostly Google-style-clean within scope; the main risks are correctness (repool, permutation, bootstrap n_splits) and the orphaned/stubbed surface area that should be cut or finished before a v0.6.0 release.
- **plotting** (12): The plotting layer is uneven. brain.py (plot_surf/plot_flatmap) is modern, well-documented, and correct, with proper keyword-only markers and canonical kwargs. adjacency.py is a clean polars-native rewrite but has two real correctness bugs (a swapped-triangle assignment tied to the normalize flag, and a stray blank figure) and an api-consistency violation (colors/figsize smuggled through **kwargs). prediction.py is the biggest risk: it is legacy, untested, and two of its four public functions (plot_probability, plot_dist_from_hyperplane) are outright BROKEN under the pinned seaborn 0.13.2 (positional x/y args raise TypeError), while all four claim to return a figure but return None — the latter silently breaks Roc.plot, whose own docstring promises a figure. No tests cover prediction.py at all, which is why the seaborn break was never caught.
- **stats** (14): The stats functional core is mostly sound numerically, but the pre-release audit surfaced two real correctness bugs (holm_bonf ignores its `alpha` argument; procrustes_distance compares an observed disparity against a null distribution of similarities and mislabels disparity as "similarity"), a dead validation check in align(), and a ~130-line cluster of dead ISC helpers superseded by the inference module. API-consistency debt is the biggest theme: the public re-exported surface still uses several kwarg names the v0.6.0 table explicitly bans (`mode=` in regress, `icc_type=` in compute_icc, `parallel=` across permutation.py, `method=`/`sim_metric=` for what are metrics), and align() forwards to internal SRM classes via `*args/**kwargs`. A handful of docstrings are stale or numpydoc-style (fisher_z_to_r, fisher_r_to_z, procrustes). Fixing the two correctness bugs and the banned-kwarg names before release is the priority.
- **templates-io-toplevel** (15): The templates/ subpackage is the strongest part of this scope: clean frozen-dataclass config, a well-factored registry, and mostly compliant Google-style docstrings. The two most serious issues are latent correctness bugs in mask.py and cross_validation.py: expand_mask silently returns empty masks for any non-contiguous-label atlas (iterates nonzero indices instead of label values), and KFoldStratified completely ignores its documented shuffle/random_state arguments. Secondary risks are a background-fill inconsistency in roi_to_brain (2-D fills with 1.0), an int()-vs-round voxel-size mismatch across the three matching.py detection functions (which also duplicate that logic), and a package __init__ whose __all__ advertises submodules (datasets, cross_validation) that aren't importable as attributes. The remaining findings are low-severity docstring/robustness polish. The h5 I/O layer is solid and defensively written.
- **xcut-api-consistency** (1): Test summary.
- **xcut-deadcode-gaps** (10): The codebase largely honors the "functional core / imperative shell" architecture: the four facades (BrainData, Adjacency, DesignMatrix, BrainCollection) delegate cleanly to pure functions in stats/, algorithms/, and per-facade submodules, and ruff finds no unused imports/vars/redefinitions. The pipelines/ module is NOT orphaned — it is still actively imported by collection/pipeline.py, collection/__init__.py (CVScheme), and its tests — so it should be kept. The real weaknesses are API-convention drift (the v0.6.0 keyword-only `*` rule and the `parallel=`/`n_jobs=` translation rule are violated in several public methods), a genuine `CVScheme` name collision between a public Protocol export and the concrete class actually used, a few stale/deferred docstrings that claim features are unimplemented when they are (align roi) or document kwargs that only raise (Glm.predict X), and minor duplicated metric-dispatch logic. No critical correctness bugs surfaced.
- **xcut-docstrings** (12): Docstring hygiene is generally strong: RST-role leakage is nearly absent (a single `:func:` in a private helper), module-docstring coverage is near-complete, and the algorithm/stats layers use clean Google-style docstrings. The two real weak spots are (1) the BrainCollection facade — the primary public entry point — where ~30 public methods and properties delegate to submodules with no docstring at all, so they render as empty API-reference entries, and (2) two legacy modules (fitresults, simulator) plus collection/pipeline.py that still use NumPy-style `Header\n----` sections which do not render under the Google-configured griffe2md parser, alongside several stale Args lists (create_cov_data/create_ncov_data phantom params, the fitresults `mode=`/`cv_dict=` example). Biggest risk: users landing on BrainCollection's reference page find blank method docs. A handful of minor api-consistency drifts (align's `scheme=`, fit's `model=`) are worth a look but are secondary to the docstring gaps.
- **xcut-tests** (11): The nltools test suite (~1,848 tests across 112 files) is broadly healthy: the correctness-critical inference/algorithms layer (ISC, permutation, correlation, ICC, ridge, SRM) has strong edge-case coverage (invalid/mismatched shapes, tied/constant data, single-vs-multi-feature, seed reproducibility), markers are registered in pyproject and largely applied consistently, and the four data facades each have dedicated suites. The biggest gaps are whole modules with zero coverage — the Roc classification class (394 LOC, empty tests/data/roc/) and the standalone stats.regress OLS helper — plus the Simulator module whose only tests are all slow-marked, so the default run exercises none of it. Secondary issues: a handful of wall-clock-timing tests that are inherently flaky (one runs by default), low-value import-only ceremony tests, and several source-side v0.6.0 convention violations (icc_type/mode/threshold_type kwargs that should be method, a missing keyword-only marker, and a stale **kwargs docstring) surfaced while cross-referencing tests to source.

## Dismissed by adversarial verification (false positives)

_These correctness findings were refuted by the codex cross-check; recorded so they aren't re-raised._

- `nltools/algorithms/ridge/utils.py:255, 257, 286` — _select_best_alphas indexes backend (torch) arrays with numpy np.arange, breaking on CUDA  
  *Why not a bug:* The finding assumes CUDA advanced indexing requires every index tensor to reside on CUDA. PyTorch deliberately supports CPU indices for CUDA tensors; NumPy integer arrays are accepted as CPU indices. Therefore mixing the CUDA `alphas_argmax` index with the NumPy/CPU `arange` index is supported, not 

- `nltools/algorithms/ridge/solvers.py:32-33, 623` — Mutable list default arguments for concentration/alphas  
  *Why not a bug:* This is a legitimate style/maintainability smell, but not a correctness bug in the traced implementation. The finding itself acknowledges that the defaults are never mutated; a hypothetical future edit is not current misbehavior.

- `nltools/data/braindata/analysis.py:1236-1269` — align() returns 'transformed' as a raw ndarray for SRM but a BrainData for procrustes  
  *Why not a bug:* The finding treats a method-dependent representation as an API correctness violation, overlooking that SRM output is latent common-feature data and may not be representable by the source BrainData mask. The proposed uniform BrainData wrapper would be incorrect for reduced-dimensional SRM targets.

- `nltools/data/roc/__init__.py:282-296` — Gaussian ROC (single-interval) uses uncorrected z_true/z_false, ignoring pooled_sd normalization asymmetry  
  *Why not a bug:* The finding treats `z_true` and `z_false` as though they were evaluated against a fixed zero-referenced threshold. They are instead evaluated over an `x` grid that shifts with both means. The common additive offset cancels inside each Gaussian CDF, leaving the ROC coordinate pairs dependent only on 
