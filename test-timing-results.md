# nltools Test Timing Analysis Results

**Date**: 2025-10-29
**Total Tests**: 385 (381 passed, 4 skipped)
**Total Runtime**: 390.60s (6:30)
**Baseline Runtime** (no timing overhead): ~255s (4:15)

---

## Executive Summary

**Top 10 slowest tests consume 241s (62% of total runtime):**
1. test_hyperalignment: 53.16s (14%)
2. test_find_spikes: 33.62s (9%)
3. test_temporal_resample: 32.11s (8%)
4. test_threshold_cluster_with_upper_only: 28.55s (7%)
5. test_align_without_isc: 24.40s (6%)
6. test_threshold_cluster_with_lower_only: 14.49s (4%)
7. test_threshold_cluster_with_binarize: 14.28s (4%)
8. test_threshold_cluster_basic: 13.81s (4%)
9. test_compute_contrasts_numeric_vector: 9.83s (3%)
10. test_compute_contrasts_string_parsing: 8.96s (2%)

**Key Finding**: Just 10 tests (2.6% of suite) take 241s (62% of runtime)!

---

## Timing Distribution

```
>20s:   5 tests  = 172s total (44% of runtime) - ULTRA SLOW
10-20s: 3 tests  = 43s total  (11% of runtime) - VERY SLOW
5-10s:  11 tests = 79s total  (20% of runtime) - SLOW
3-5s:   12 tests = 45s total  (12% of runtime) - MODERATE
<3s:    ~350 tests = ~51s total (13% of runtime) - FAST
```

**Average test time**:
- Top 50 slowest: 7.0s average
- Remaining ~330 tests: 0.15s average (47× faster!)

---

## Detailed Timing Breakdown (Top 50 Slowest Tests)

### TIER 2A: Ultra Slow (>20s) - 5 tests, 172s total

| Time | Test | Category |
|------|------|----------|
| 53.16s | test_hyperalignment | Brain_Data.hyperalignment() |
| 33.62s | test_find_spikes | stats.find_spikes() |
| 32.11s | test_temporal_resample | Brain_Data.temporal_resample() |
| 28.55s | test_threshold_cluster_with_upper_only | Brain_Data.threshold() clustering |
| 24.40s | test_align_without_isc | stats.align() without ISC |

**Impact**: These 5 tests alone = 44% of total suite runtime

### TIER 2B: Very Slow (10-20s) - 3 tests, 43s total

| Time | Test | Category |
|------|------|----------|
| 14.49s | test_threshold_cluster_with_lower_only | Brain_Data.threshold() clustering |
| 14.28s | test_threshold_cluster_with_binarize | Brain_Data.threshold() clustering |
| 13.81s | test_threshold_cluster_basic | Brain_Data.threshold() clustering |

**Pattern**: All threshold clustering tests (uses nilearn connected components)

### TIER 2C: Slow (5-10s) - 11 tests, 79s total

| Time | Test | Category |
|------|------|----------|
| 9.83s | test_compute_contrasts_numeric_vector | Brain_Data.compute_contrasts() |
| 8.96s | test_compute_contrasts_string_parsing | Brain_Data.compute_contrasts() |
| 8.84s | test_regress_glm_parameters | Brain_Data.regress() with GLM |
| 8.81s | test_compute_contrasts_invalid_length | Brain_Data.compute_contrasts() |
| 8.72s | test_compute_contrasts_multiple_dict | Brain_Data.compute_contrasts() |
| 8.63s | test_extract_roi | Brain_Data.extract_roi() |
| 6.74s | test_regress | Brain_Data.regress() base test |
| 6.31s | test_decompose | Brain_Data.decompose() |
| 5.53s | test_glm_fit_matches_current_regress | Brain_Data.fit(model='glm') |
| 5.15s | test_predict_multi | Brain_Data.predict_multi() |

**Pattern**: GLM operations, decomposition, ROI extraction

### TIER 2D: Moderate (3-5s) - 12 tests, 45s total

| Time | Test | Category |
|------|------|----------|
| 4.57s | test_no_accidental_deep_copies | Performance validation |
| 4.49s | test_permutation | stats.permutation() |
| 4.43s | test_transform_methods_efficient | Performance validation |
| 4.34s | test_load | Brain_Data I/O |
| 4.27s | test_threshold_cluster_realistic_neuroimaging | Threshold integration |
| 3.74s | test_fit_passes_kwargs_to_model | fit() integration |
| 3.23s | test_regress_uses_glm_model | regress() GLM test |
| 3.17s | test_regress_attributes_match_glm | regress() validation |
| 3.11s | test_threshold | Brain_Data.threshold() base |
| 3.09s | test_auto_backend_selection | Backend selection |
| 3.06s | test_fit_predict_glm_workflow | fit/predict integration |

---

## Tier Classification Recommendation

### TIER 1 (Fast Core Tests): <3s per test
- **Count**: ~330 tests
- **Total time**: ~51s (13% of runtime)
- **Target**: <2 minutes total
- **Run frequency**: Every development iteration

**Criteria**:
- API contracts (parameters, return types, shapes)
- Mathematical correctness with minimal data
- Fast numpy operations
- Unit tests (single function, mocked dependencies)

**Examples**:
- All initialization tests
- Property getters/setters
- Math operations (add, subtract, mean, std)
- Append/concatenate
- Most test_models.py (except GLM tests)
- Most test_srm.py (property-based tests)
- Most test_design_matrix_new.py (Polars operations)

### TIER 2 (Comprehensive Tests): >3s per test
- **Count**: ~50 tests
- **Total time**: ~339s (87% of runtime!)
- **Target**: <15 minutes total
- **Run frequency**: Before commits, on PR merge, nightly CI

**Criteria**:
- Expensive nilearn operations (threshold clustering, GLM)
- Integration tests (full pipelines)
- Performance benchmarks
- I/O operations with large data
- Resampling methods (permutation)
- Full brain data operations

**Specific tests to mark as tier2**:

**From test_brain_data.py** (31 tests):
```python
# Ultra Slow (>20s)
test_hyperalignment  # 53s
test_temporal_resample  # 32s
test_threshold_cluster_with_upper_only  # 29s

# Very Slow (10-20s)
test_threshold_cluster_with_lower_only  # 14s
test_threshold_cluster_with_binarize  # 14s
test_threshold_cluster_basic  # 14s

# Slow (5-10s)
test_compute_contrasts_numeric_vector  # 10s
test_compute_contrasts_string_parsing  # 9s
test_regress_glm_parameters  # 9s
test_compute_contrasts_invalid_length  # 9s
test_compute_contrasts_multiple_dict  # 9s
test_extract_roi  # 9s
test_regress  # 7s
test_decompose  # 6s
test_glm_fit_matches_current_regress  # 6s
test_predict_multi  # 5s

# Moderate (3-5s)
test_load  # 4s
test_threshold_cluster_realistic_neuroimaging  # 4s
test_fit_passes_kwargs_to_model  # 4s
test_regress_uses_glm_model  # 3s
test_regress_attributes_match_glm  # 3s
test_threshold  # 3s
test_fit_predict_glm_workflow  # 3s
test_regress_backward_compatible_dict  # 3s
test_regress_numerical_equivalence  # 3s
test_regress_emits_future_warning  # 3s
```

**From test_stats.py** (3 tests):
```python
test_find_spikes  # 34s - ULTRA SLOW
test_align_without_isc  # 24s - ULTRA SLOW
test_permutation  # 4s
```

**From test_efficient_copy.py** (4 tests):
```python
test_no_accidental_deep_copies  # 5s
test_transform_methods_efficient  # 4s
test_shallow_copy_with_data  # 2.5s (setup)
test_chained_operations_preserve_efficiency  # 2s
```

**From test_ridge.py** (1 test):
```python
test_auto_backend_selection  # 3s
```

**From test_adjacency.py** (1 test):
```python
test_similarity  # 2s
```

**From test_simulator.py** (1 test):
```python
test_simulator  # 2s
```

---

## Expected Performance Improvements

### Current State (no tiers)
```
Full suite: 390s (6:30) with timing overhead
           ~255s (4:15) without overhead
All tests run every time: SLOW feedback loop
```

### After Tier Implementation

**Tier 1 Only** (development iteration):
```
~330 tests in ~51s base + overhead = ~70s (<2 minutes)
Speedup: 3.6× faster than current full suite
Feedback: FAST - run on every save
```

**Tier 2 Only** (pre-commit):
```
~50 tests in ~339s base + overhead = ~450s (~7.5 minutes)
Still comprehensive, but targeted
```

**Combined** (before releases):
```
Full suite: ~390s (~6.5 minutes)
No change, but organized for selective running
```

**With pytest-xdist -n4** (parallel execution):
```
Tier 1: ~70s / 4 = ~18s (!!!)
Tier 2: ~450s / 4 = ~113s (~2 minutes)
Full suite: ~390s / 4 = ~98s (~1.5 minutes)
```

### Projected Workflow

**During development**:
```bash
# Fast feedback loop: ~18s with parallelization
uv run pytest -m tier1 -n auto
```

**Before commits**:
```bash
# Quick tier1 check: ~18s
uv run pytest -m tier1 -n auto

# Run tier2 for affected areas only: ~30-60s
uv run pytest nltools/tests/shell/test_brain_data.py -m tier2 -x
```

**Before releases**:
```bash
# Full comprehensive suite: ~98s with parallelization
uv run pytest -m "tier1 or tier2" -n auto
```

---

## Implementation Priority

### Phase 1: Mark Obvious Slow Tests (30 minutes)
**Goal**: Quick wins - mark the 19 tests >5s

**Impact**: Prevents developers from running 294s of slow tests during rapid iteration
**Savings**: ~5 minutes per development cycle

**Tests to mark** (19 tests, 294s total):
1. test_hyperalignment (53s)
2. test_find_spikes (34s)
3. test_temporal_resample (32s)
4. test_threshold_cluster_with_upper_only (29s)
5. test_align_without_isc (24s)
6. test_threshold_cluster_with_lower_only (14s)
7. test_threshold_cluster_with_binarize (14s)
8. test_threshold_cluster_basic (14s)
9. test_compute_contrasts_numeric_vector (10s)
10. test_compute_contrasts_string_parsing (9s)
11. test_regress_glm_parameters (9s)
12. test_compute_contrasts_invalid_length (9s)
13. test_compute_contrasts_multiple_dict (9s)
14. test_extract_roi (9s)
15. test_regress (7s)
16. test_decompose (6s)
17. test_glm_fit_matches_current_regress (6s)
18. test_predict_multi (5s)
19. test_no_accidental_deep_copies (5s)

### Phase 2: Add pytest Configuration (5 minutes)
Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "tier1: Fast core tests (<3s, run on every iteration)",
    "tier2: Comprehensive tests (>3s, run before commits)",
]
# Default: run tier1 only for fast feedback
addopts = "-m 'not tier2'"
```

### Phase 3: Mark Remaining Tier 2 Tests (1 hour)
Mark remaining ~31 tests that are 3-5s each.

### Phase 4: Update Documentation (15 minutes)
Update CLAUDE.md with new test commands and workflow.

### Phase 5: Add Parallel Execution (optional, 10 minutes)
```toml
# pyproject.toml - add to dev dependencies
[dependency-groups]
dev = [
    # ... existing deps ...
    "pytest-xdist>=3.0.0",  # Parallel test execution
]
```

---

## Risk Analysis

### "Will we miss bugs by not running tier2?"

**Mitigation**:
- Tier1 still tests correctness (just with smaller data)
- Tier2 runs automatically before PRs merge
- Nightly CI runs full suite
- Release checklist: full suite must pass

### "What if tier1 becomes slow over time?"

**Mitigation**:
- Monitor with `pytest --durations=20`
- Set time budget: tier1 must stay <2 min
- Move slow tests to tier2
- Regular audits (quarterly)

### "Classification errors?"

**Mitigation**:
- Clear 3s threshold (objective)
- Document rationale in test docstrings
- Easy to reclassify: just change marker
- Review in PRs

---

## Maintenance Strategy

**Quarterly audit**:
```bash
# Check if any tier1 tests have become slow
uv run pytest -m tier1 --durations=20

# Check total tier1 time
uv run pytest -m tier1 --collect-only | wc -l
uv run pytest -m tier1 --quiet  # Should be <2 min
```

**When adding new tests**:
- If test uses full brain data (238K voxels) → tier2
- If test uses expensive nilearn operations → tier2
- If test uses network/external resources → tier2
- If test is performance benchmark → tier2
- Otherwise → tier1 (default)

---

## Next Steps

1. **Get approval for approach** ✅
2. **Phase 1**: Mark 19 slowest tests as tier2 (30 min)
3. **Phase 2**: Add pytest configuration (5 min)
4. **Phase 3**: Mark remaining tier2 tests (1 hour)
5. **Phase 4**: Update CLAUDE.md (15 min)
6. **Phase 5** (optional): Add pytest-xdist (10 min)
7. **Validate**: Run tier1, measure time, ensure <2 min
8. **Commit**: Document changes in refactor-progress.md

---

## Appendix: Full Top 50 Timing Data

```
53.16s  test_hyperalignment
33.62s  test_find_spikes
32.11s  test_temporal_resample[2mm]
28.55s  test_threshold_cluster_with_upper_only[2mm]
24.40s  test_align_without_isc
14.49s  test_threshold_cluster_with_lower_only[2mm]
14.28s  test_threshold_cluster_with_binarize[2mm]
13.81s  test_threshold_cluster_basic[2mm]
9.83s   test_compute_contrasts_numeric_vector
8.96s   test_compute_contrasts_string_parsing
8.84s   test_regress_glm_parameters[2mm]
8.81s   test_compute_contrasts_invalid_length
8.72s   test_compute_contrasts_multiple_dict
8.63s   test_extract_roi[2mm]
6.74s   test_regress[2mm]
6.31s   test_decompose[2mm]
5.53s   test_glm_fit_matches_current_regress[2mm]
5.15s   test_predict_multi
4.57s   test_no_accidental_deep_copies
4.49s   test_permutation
4.43s   test_transform_methods_efficient
4.34s   test_load
4.27s   test_threshold_cluster_realistic_neuroimaging[2mm]
3.74s   test_fit_passes_kwargs_to_model[2mm]
3.23s   test_regress_uses_glm_model[2mm]
3.17s   test_regress_attributes_match_glm[2mm]
3.11s   test_threshold
3.09s   test_auto_backend_selection
3.06s   test_fit_predict_glm_workflow[2mm]
2.84s   test_regress_backward_compatible_dict[2mm]
2.83s   test_regress_numerical_equivalence[2mm]
2.79s   test_regress_emits_future_warning[2mm]
2.47s   test_shallow_copy_with_data[2mm] (setup)
2.45s   test_chained_operations_preserve_efficiency
2.37s   test_simulator
2.30s   test_comparison_with_deepcopy
2.25s   test_similarity
1.92s   test_create_sphere
1.90s   test_regress_returns_backward_compatible_dict[2mm]
1.85s   test_check_brain_data[2mm] (setup)
1.82s   test_regress_supports_self_X_pattern[2mm]
1.81s   test_regress_calls_fit_internally[2mm]
1.80s   test_regress_ignores_mode_robust_silently[2mm]
1.76s   test_shape[2mm] (setup)
1.56s   test_append_correctness
1.47s   test_roi_to_brain
1.45s   test_apply_mask[2mm]
1.19s   test_apply_mask_resampling[2mm]
0.89s   test_check_brain_data[2mm]
```

---

**Generated**: 2025-10-29
**Tool**: pytest --durations=50
**Total Runtime**: 390.60s (6:30)
