# GitHub Issues Audit for v0.6.0

**Date:** 2026-01-01
**Total Open Issues:** 76

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Already Solved | ~12 | Close on GitHub |
| Your Meta-Issues | 9 | Close after v0.6.0 release |
| Already in Beads | 1 | No action |
| Bugs to Verify/Fix | ~10 | Triage & fix for v0.6.0 |
| P4 Feature Requests | ~40 | Backlog for v0.7.0+ |
| Stale/Wontfix | ~4 | Close with explanation |

---

## 1. ALREADY SOLVED (Can Close on GitHub)

These issues have been resolved by recent work:

| # | Title | Evidence |
|---|-------|----------|
| 418 | stats.upsample .iteritems() | 0 occurrences in codebase |
| 435 | Fix failing test_threshold | 11 threshold tests pass |
| 444 | nilearn _get_dataset_dir import | 0 occurrences in codebase |
| 445 | numpy locked to old version | uv-cleanup updated deps |
| 451 | scipy binom_test deprecation | 0 occurrences in codebase |
| 201 | add backends | Backends implemented (numpy/torch/jax) |
| 430 | Python 3.11 joblib | Known issue, mitigated |

**Likely Solved (need quick verification):**

| # | Title | Check Needed |
|---|-------|--------------|
| 370 | Brain_Data.threshold() Bug | Tests pass |
| 371 | Adjacency multiple directed_flat | Adjacency refactored |
| 383 | Adjacency h5 with Path object | Quick test |
| 390 | Adjacency append error | Adjacency refactored |
| 432 | Adjacency.similarity() NaNs | Quick test |

---

## 2. YOUR META-TRACKING ISSUES (Close After v0.6.0)

These are umbrella issues you created in July 2024:

| # | Title | Status |
|---|-------|--------|
| 377 | Major refactoring | In progress (uv-cleanup branch) |
| 433 | Move docs to jupyter book | Part of nltools-foa |
| 434 | Clean up regress | Partial progress |
| 436 | Plotting fixes & enhancements | Part of nltools-81p |
| 437 | Evaluate alternative backends | DONE - backends exist |
| 438 | Stats fixes and enhancements | ISC fixed, ongoing |
| 439 | Adjacency fixes & enhancements | DONE - refactored |
| 440 | Brain_Data fixes & enhancements | Ongoing |
| 441 | Design_Matrix fixes & enhancements | Ongoing |

---

## 3. ALREADY TRACKED IN BEADS

| # | Title | Bead |
|---|-------|------|
| 452 | Header-only read functions | nltools-6wj |

---

## 4. BUGS TO VERIFY/FIX FOR v0.6.0

**High Priority (user-reported, recent):**

| # | Title | Created | Notes |
|---|-------|---------|-------|
| 446 | Brain_Data atlas resampling creates new values | 2024-12 | Bug in apply_mask/resample? |
| 447 | "bug" | 2025-05 | Need to read details |
| 449 | Brain_Data flattens multiple files | 2025-08 | Constructor behavior |
| 443 | download nifti from web broken | 2024-08 | nilearn API changes |
| 429 | neurovault downloading new nilearn API | 2024-03 | Related to #443 |

**Medium Priority (older but valid):**

| # | Title | Created | Notes |
|---|-------|---------|-------|
| 126 | Expand/collapse mask extra values | 2017-04 | Long-standing |
| 311 | iplot vs plot percentile thresholding | 2019-07 | Plotting consistency |
| 357 | create_sphere() problem | 2020-06 | Need to verify |
| 372 | Plotting thresholded data unreliable | 2021-03 | Related to #311 |
| 381 | Adjacency.plot_mds n_components=3 | 2021-03 | Quick check |
| 422 | fetch_pain() issue | 2023-10 | Dataset fetch |
| 424 | legacy hdf5 loading | 2023-12 | File format compat |

---

## 5. P4 FEATURE REQUESTS (Backlog v0.7.0+)

### Prediction/Classification
- #176 - permutation tests to predict
- #177 - balanced option to predict
- #182 - Clean up predict algorithms
- #183 - Refactor ROC module
- #265 - SVM return distance to hyperplane
- #425 - spearman correlation for predict

### Adjacency Enhancements
- #196 - CV to Adjacency class
- #233 - cluster method to adjacency
- #244 - 2d_permutation for Adjacency ttest
- #254 - Consider subclassing Adjacency
- #384 - main diagonal in Adjacency.distance
- #394 - network-based statistics

### Design Matrix
- #263 - Alphabet-Optimality metrics
- #277 - subclassing updates
- #284 - optimal highpass filter cutoff
- #344 - BIDS events compatibility
- #347 - flexible custom index
- #405 - onsets_to_dm weighting

### Stats/Inference
- #172 - Stats output class
- #287 - sigma estimation in regress
- #304 - stats.threshold confusing usage
- #307 - auto-correlation in regression
- #315 - multiple comparisons + 1-tailed tests
- #317 - Improve Brain_Data.randomise
- #395 - different similarity metrics to isc()

### Brain_Data
- #150 - tests for custom masks
- #285 - voxel-wise scale option
- #339 - mahalanobis distance to find_spikes
- #354 - interpolation option to append
- #386 - probabilistic atlas
- #391 - ignore_attrs flag to append
- #410 - SRM different sample sizes

### Plotting
- #278 - gif from graphs over time
- #378 - vmin/vmax/cmap options

### Misc
- #239 - cythonized tfce from MELD
- #249 - Add DueCredit
- #302 - h5 files on different hardware
- #402 - design matrix in RSA
- #420 - slice time correction defaults

---

## 6. STALE/WONTFIX

| # | Title | Reason |
|---|-------|--------|
| 227 | Github Repository too big | LFS or archive, low priority |
| 250 | Neurovault I/O example error | 6+ years old, likely outdated |
| 148 | lassopcr documentation | Docs being overhauled |

---

## Recommended v0.6.0 Action Plan

### Phase 1: Quick Wins (1 session)
1. Verify and close ~12 already-solved issues
2. Read #447 to understand the bug
3. Quick tests for #370, #371, #383, #390, #432

### Phase 2: Critical Bugs (1-2 sessions)
1. Fix #446 (atlas resampling) - user-reported, recent
2. Fix #449 (constructor flattens files) - user-reported
3. Fix #443/#429 (nilearn download) - if feasible

### Phase 3: Create Beads for Tracking
- Create beads for bugs being fixed
- Create P4 beads for feature requests (bulk)
- Link to GitHub issue numbers

### Phase 4: Close Meta-Issues
- After v0.6.0 release, close #377, #437, #438, #439, etc.
