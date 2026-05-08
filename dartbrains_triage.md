# dartbrains × nltools v0.6.0 — Triage

Living status doc covering both nltools-side fixes and dartbrains course-side
migration to the v0.6.0 API. Last full pass: 2026-05-08 (predict-rewrite
follow-ups landed: `a3fe4da7` brain-space `Predict` fields + ROI per-fold
scores, `80c882b8` `.convolve()` idempotent fix, `05b02baf` drop
`final_weight_map`, `c3b9f6e3` ROI voxel-space weight maps).

---

## Landed

| ID | Side | Item | Resolution |
|---|---|---|---|
| **A** | course | Class renames `Brain_Data → BrainData`, `Design_Matrix → DesignMatrix` | sd-applied across 11 + 5 files; zero stragglers |
| **A** | course | Import-path updates: `glover_hrf`, `SimulateGrid`, `get_anatomical → load_mni152_template`, `one_sample_permutation_test`, `regress`/`onsets_to_dm` dropped | 9 files |
| **A** | course | `data.X = dm; data.regress()` → `data.fit(model='glm', X=dm)` + `.glm_betas`/`.glm_residual` | 4 real call sites + narrative |
| **A** | course | `dm.heatmap()` → `dm.plot()` | 7 call sites + narrative |
| **A** | course | `ttest(threshold_dict={…})` → `ttest()` + `.threshold(upper=, lower=)` on z-map; FDR via `fdr()` + `norm.isf()` | `Thresholding_Group_Analyses.py` × 4 cells |
| **C1** | nltools | `convolve()` `_c{i}` suffix consistency between data + `.convolved` metadata | `942c9b35`: always-suffix; `.convolved` post-suffix |
| **C2** | nltools + course | nilearn 0.13 `cut_coords=range(...)` rejection | nltools coerces `range → list` (`e0a8d92a`); course `nilearn<0.13` pin dropped |
| **C3** | nltools (smoke) + course | `onsets_to_dm` column-mangling workaround in `Connectivity.load_bids_events` | Smoke-tested with mixed-case/hyphen `trial_type` values; `Connectivity.py` rewritten to `DesignMatrix(events_path, run_length=N, TR=tr).convolve()` + `.append([…], axis=1)` chain |
| **C4** | course | `BrainData([BrainData(f) for f in paths])` 0.5.1 flatten workaround | Removed in `RSA.py`, `Group_Analysis.py`, `Thresholding_Group_Analyses.py` |
| **C5.2** | nltools | `DesignMatrix(other_dm)` should copy-construct | `1c167d31`: accepted; carries `.data`/`.convolved`/`.confounds`/`.sampling_freq`/`.multi` |
| **C5.3** | nltools | `dm.convolved`/`dm.confounds` direct mutation footgun | `1c167d31`: read-only properties; `.convolve()`/`.append()` are the only managed paths |
| **C7** | nltools | `decompose()` had loose `*args, **kwargs` | `a0cfd67c`: dropped dead `*args`, kwarg-only marker, sklearn passthrough audited and documented; also fixed stale `algorithm='ica'` in `plotting/decomposition.py` docstring |
| **C2 (course)** | course | `dartbrains/pyproject.toml` deps | `nilearn<0.13` pin dropped; `nltools` repointed to local `path = "../"` (editable) |

---

## TODO — nltools side

### Migration-guide gaps (B-bucket)

- [x] ~~**B1 — `BrainData.iplot()` removal not documented.**~~ Resolved by **rebuilding the method** rather than documenting its removal. New `iplot()` is an `anywidget`-backed viewer (no `ipywidgets` extra) that renders identically in Jupyter, marimo, VS Code, and JB v2 / mystmd built sites via the standard widget mimebundle. API: `iplot(view='ortho'\|'surface', threshold=None, bg_img=None, ...)`. 4D BrainData auto-grows a volume-step slider alongside the threshold slider. `surface=True` → `view='surface'`, `anatomical=` → `bg_img=` (canonical with nilearn). `anywidget` added to `[project.dependencies]`. 11 tests added in `nltools/tests/data/braindata/test_iplot.py` covering 3D/4D dispatch, slider triggers, view-kwarg routing, threshold range from data, mimebundle for JB v2 builds. Course-side: 22 dartbrains `.iplot()` call sites continue to work as-is (signature compatible for the no-arg form). Migration guide gained Pattern 0 + a row in the Removed Methods table reframed as "rebuilt."
- [x] ~~**B2 — `decompose(algorithm=) → decompose(method=)` not called out by name.**~~ Added explicit `decompose` mentions in the migration guide at five surfaces: the methods/alternatives table (line 590), the algorithm-rename row's Scope column (line 1163), an OLD/NEW example pair (lines 1177/1186), the per-component change table (line 1362), the backward-compat status table (line 1522), and the must-fix checklist (line 1618). Readers grepping for `decompose` will now hit the rename in any of those places.
- [x] ~~**B4 — `DesignMatrix(events_path)` produces boxcar, not HRF-convolved.**~~ Resolved by **changing the default** rather than just documenting harder. Constructor now takes `hrf_model='glover'` (matching nilearn's `make_first_level_design_matrix`) and HRF-convolves events files at construction time; pass `hrf_model=None` to opt into boxcar (PPI / FIR / pedagogy). Tests added in `test_file_reader.py::TestDesignMatrixFromEventsFile` covering default-convolved, opt-out boxcar, unknown-model rejection, and tabular-file ignore. Migration guide rewritten in three places (rename table line 83, full example block at lines 95-115, file-paths section at lines 314-335, must-fix checklist at line 1605). Course-side: `Group_Analysis.py` dropped its trailing `.convolve()`; `Connectivity.py` PPI cell + `GLM_Single_Subject_Model.py` pedagogy cell got explicit `hrf_model=None` opt-outs with comments explaining why.
- [x] ~~**B5 — `DesignMatrix.reset_index()` removal.**~~ Resolved without action. Pandas-era artifact; Polars has no row-index concept so no shim is warranted. The only `dm.reset_index(drop=True)` call in dartbrains was inside the `Connectivity.load_bids_events` workaround already removed during the C3 sweep. Remaining `reset_index()` mentions in `Introduction_to_Pandas.py` are pandas-tutorial cells operating on real `pd.DataFrame` instances (teaching pandas itself) — correctly untouched.
- [x] ~~**C5.1 — Confound-stacking worked example.**~~ Resolved via three small API additions + a worked-example block. Stepping back from the v0.5.1 idioms surfaced two real interop gaps that would have made any "documented" workaround fragile: (1) `DesignMatrix.with_columns(**named_exprs)` — Polars-native chainable column add/replace, accepts `pl.Expr`/`pl.Series`/arrays/scalars, returns new DM with metadata preserved; (2) `__setitem__` accepts `pl.Expr` — procedural alternative to `with_columns`; (3) `BrainData.find_spikes()` returns a `DesignMatrix` with spike columns pre-marked as confounds and no `TR` index column (pandas-era artifact dropped — row position is the time axis), with `TR=`/`sampling_freq=` kwargs to set the returned DM's sampling rate. Migration guide gained a "Worked example: PPI design" subsection in the file-paths section showing all three idioms composed end-to-end. Course-side: `Connectivity.py` PPI cell + seed-FC cell (×2) + `GLM_Single_Subject_Model.py` spikes cell + `Group_Analysis.py` narrative all rewritten using the cleaner surface; the rewrite also fixed two latent bugs (course used `dm.drop(axis=1)` and `dm.loc[:, ...]` which never existed on the v0.6.0 Polars-backed DM, plus `dm.convolved = [...]` mutation that became read-only in `1c167d31`).
- [x] ~~**C6 — URL → nibabel idiom.**~~ Resolved by leaning on existing infra rather than recommending a new external call. The 4 atlases used by dartbrains tutorials (Desikan-Killiany, Shen 268, Glasser 360, FSL bilateral amygdala) were uploaded to the existing `nltools/niftis` HF dataset under `masks/`. Added `nltools.templates.list_resources(prefix=None)` for discoverability — single HF API hit per session, cached. `fetch_resource(relpath)` is now the canonical entry point: returns a path string that drops into both nilearn (`plot_roi(fetch_resource(...))`) and BrainData (`BrainData(fetch_resource(...))`) without round-tripping through `.to_nifti()`. Course-side: 4 Pattern A sites + 3 Pattern B sites rewritten. Migration guide gained a "Loading canonical brain images" subsection.

### New issue surfaced during course migration

- [x] ~~**NEW: `predict()` return shape collapsed to a single accuracy `BrainData`.**~~ Resolved by **rewriting `BrainData.predict()` around a `Predict` dataclass** (mirrors the `Fit` story for `bd.fit()`). The old v0.5.1 dict (`weight_map`, `mcr_all`, `yfit_all`, `dist_from_hyperplane_all`) maps cleanly onto `Predict.weight_map`, `Predict.scores` / `.mean_score`, `Predict.predictions`, with `fold_weight_maps` (per-fold coefs) and `final_weight_map` (full-data refit) added on top. Signature collapsed: `model='svm'` (mirrors `bd.fit(model=)`, replaces `estimator=` and the legacy `algorithm=`), `scoring='auto'` (resolves to `'accuracy'`/`'r2'` via `is_classifier`), `standardize=True`, `reduce='pca'`/`n_components=` for optional per-fold PCA (with weight-map back-projection), `refit=True` for the full-data fit. The fluent API on `BrainData` (`.cv()`, `.normalize()`, `.reduce()`, `.pipe()`) was deleted in the same change — every step folded into a kwarg, and custom preprocessing chains use the standard sklearn idiom (`model=make_pipeline(StandardScaler(), MyXform(), SVC())`). `BrainCollection`'s pipeline stays — multi-subject hyperalignment / SRM is where chaining still earns its keep. Course-side: `Multivariate_Prediction.py` is now mechanically migratable — `result['weight_map']` → `result.weight_map`, `algorithm=` → `model=`, `cv_dict=` → `cv=`. Tests: 17 in `test_braindata_prediction.py` (+ 13 in `test_predict_results.py`) cover whole-brain × searchlight × ROI × classification/regression, model passthrough, `scoring='auto'`, PCA back-projection, `refit`, `inplace=True/False`, and non-linear-model warning.

---

## TODO — course side

### Open regressions from the v0.6.0 mechanical pass

- [ ] **`GLM_Single_Subject_Model.py` cell at lines 105-120 still uses `onsets_to_dm`.** When I dropped the import in the A-bucket sweep I missed the corresponding cell that wraps it in a local `load_bids_events(subject)` helper. The cell signature still lists `onsets_to_dm` in its captures (line 88) and `def _(get_file, get_tr, load_events, nib, onsets_to_dm)` in its signature (line 106). Same C3 treatment as `Connectivity.py` — replace with `DesignMatrix(get_file('S01', 'raw', 'events', '.tsv'), run_length=n_tr, TR=tr)` (boxcar). Also drop `onsets_to_dm` from the imports cell return tuple. **Fix before committing.**
- [ ] **Audit other cells for stale capture-tuple entries** after the import drops in the A pass. Marimo cells declare both inputs (signature) and outputs (return tuple). I dropped imports without sweeping the corresponding return tuples / downstream signatures in every file. Affected names to grep: `regress`, `onsets_to_dm`, `get_anatomical`, `one_sample_permutation` (without `_test`).

### Deferred chapter

- [ ] **`Multivariate_Prediction.py`** — unblocked by the nltools-side `Predict` rewrite (see resolved item above), then **further simplified** by four follow-up commits (`a3fe4da7`, `80c882b8`, `05b02baf`, `c3b9f6e3`). Migration is now even more mechanical than the original triage anticipated:
  - Kwarg rewrite: `algorithm=` → `model=`, `cv_dict={...}` → `cv=int_or_splitter`. Note `'ridge'` is regression-only — for classification use `'ridge_classifier'`.
  - Result access: `result['weight_map']` → `result.weight_map`; `result['mcr_all'].mean()` → `result.mean_score`; per-fold scores → `result.scores`. Spatial fields (`weight_map`, `fold_weight_maps`, `accuracy_map`) are now `BrainData` objects — `result.weight_map.plot()` works directly with no wrapping.
  - For the published "single weight map": **don't pass anything special** — `result.weight_map` is the all-data refit by default (commit `05b02baf` dropped `refit=` / `final_weight_map`). The per-fold CV-mean is no longer surfaced as a top-level field; it's `result.fold_weight_maps.data.mean(axis=0)` if anyone needs it.
  - To apply the trained model to new data: `result.estimator.predict(new_X)` (whole_brain) or `result.estimator[label].predict(...)` (ROI, dict-keyed by parcel label).
  - Custom preprocessing pipelines: `model=make_pipeline(StandardScaler(), SelectKBest(...), LinearSVC())`. Don't pass `standardize=False` — `predict()` auto-detects the Pipeline and flips it for you with a one-shot warning (commit `a3fe4da7`). The fluent `data.cv(k=5).predict(...)` API is gone.
  - **New capability available**: ROI dispatch (`method='roi', roi_mask=atlas`) now produces voxel-space `weight_map` / `fold_weight_maps` plus a per-parcel `estimator` dict (commit `c3b9f6e3`). If the chapter wants a "where in the brain is the pattern most diagnostic" analysis to complement whole-brain decoding, this is a single extra `predict()` call.
  - **Reference tutorial**: `docs/tutorials/workflows/05_decoding.md` walks through the whole flow on Haxby (face vs house) — whole-brain weight map + interpretation caveats, fold stability map, ROI dispatch with the bundled k50 atlas, and a Pipeline-as-model aside. Use as the migration target shape for `Multivariate_Prediction.py`.
  - C4 cleanup of `BrainData([BrainData(f) for f in paths])` at lines 168-169 still needed.

### iplot() cleanup (depends on B1 nltools-side decision)

- [ ] Replace 22 `.iplot()` call sites once B1 documents the canonical replacement. Files: `Group_Analysis.py`, `Thresholding_Group_Analyses.py`, `Introduction_to_Neuroimaging_Data.py`, `Connectivity.py`, `GLM_Single_Subject_Model.py`, `Multivariate_Prediction.py`.

### Lower-priority C-bucket cleanups

- [ ] **`Connectivity.py:285-290`** — earlier seed-based DM cell still uses the C5 `pd.concat([vmpfc_1, csf, mc_cov, spikes…], axis=1)` + manual `dm.convolved = ['vmpfc']` pattern. Now that `1c167d31` made `.convolved` read-only, this either crashes or needs the same `dm.append([…], axis=1)` rewrite I applied to the PPI cell at line ~335. Same pattern, separate cell — covered by the same idiom but left untouched in this pass.
- [ ] **Glossary.md** — narrative content beyond the class renames. The bulk renames swept ~40 occurrences cleanly, but glossary entries describing the old `.regress()` / `.iplot()` / `predict()` dict-return API likely need narrative editing once those topics settle.

### Pre-commit verification

- [ ] Run each rewritten notebook through `marimo edit` (or `marimo export ipynb --include-outputs`) to catch the cell signature/return-tuple mismatches the AST parse won't flag. The mechanical pass got us to "files parse"; this gets us to "cells execute".
- [ ] Visual smoke check on the figure-rendering cells per the dartbrains CLAUDE.md rules around `plt.gcf()` as terminal expression.
