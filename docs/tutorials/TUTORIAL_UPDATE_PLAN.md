# Tutorial Update Plan for nltools v0.6.0

## Overview
This document outlines the plan to update all tutorials to match the current codebase API and use the Haxby dataset consistently.

## Key API Changes Identified

### Class Name Changes
- `Brain_Data` → `BrainData`
- `Design_Matrix` → `DesignMatrix`
- `BIDSLayout` → Still used, but tutorials should use `fetch_haxby()` instead

### Method Changes
- `data.regress()` → `data.fit(model="glm", X=design_matrix)`
- `stats['beta']` → `data.glm_betas[i]` or `data.compute_contrasts()`
- `stats['t']` → `data.glm_t[i]`
- `stats['p']` → `data.glm_p[i]`
- `stats['residual']` → `data.glm_residuals`

### Design Matrix Changes
- `DesignMatrix` now uses Polars internally but maintains pandas compatibility
- `sampling_freq` parameter instead of direct TR handling
- `onsets_to_dm()` now returns `DesignMatrix` directly (already convolved if hrf_model specified)
- `.convolve()` method still exists
- `.add_dct_basis()` → `.add_dct_basis(duration=128)` (same API)
- `.add_poly()` → `.add_poly(order=2, include_lower=True)` (same API)
- `.append()` → Still works but may need axis parameter

### Dataset Changes
- Replace localizer dataset with Haxby dataset
- Use `fetch_haxby()` from `nltools.datasets`
- Returns `(brain_data, design_matrix)` tuples per run

## Tutorial-by-Tutorial Plan

### 01_basics.py - Introduction to Neuroimaging Data
**Status**: Needs major updates

**Changes Needed**:
1. Replace BIDS localizer dataset with Haxby dataset
2. Update `Brain_Data` → `BrainData`
3. Remove BIDSLayout examples (or keep minimal for BIDS intro)
4. Use `fetch_haxby()` to load data
5. Update all method calls to current API
6. Keep BIDS introduction but simplify to focus on nltools usage

**Key Updates**:
- Import: `from nltools.datasets import fetch_haxby`
- Load: `brain_data, design_matrix = fetch_haxby(n_subjects=1)`
- Update all `Brain_Data()` → `BrainData()`

### 02-05_glm-*.py - GLM Tutorials (CONSOLIDATE)
**Status**: Consolidate into single comprehensive tutorial

**New Structure**: `02_glm.py` - Comprehensive GLM Tutorial

**Sections to Include**:
1. **Introduction to GLM** (from 03_glm-II.py)
   - Simulation examples
   - Basic GLM concepts
   - OLS estimation
   - Contrasts

2. **Single Subject GLM** (from 02_glm-I.py)
   - Building design matrices
   - HRF convolution
   - Nuisance variables (motion, spikes, DCT, polynomials)
   - Fitting GLM models
   - Computing contrasts

3. **Group Analysis** (from 04_glm-III.py)
   - Mixed effects models
   - Second-level analysis
   - One-sample t-tests
   - Group-level contrasts

4. **Thresholding** (from 05_glm-IV.py)
   - Multiple comparisons correction
   - FWER vs FDR
   - Thresholding strategies

**Changes Needed**:
1. Use Haxby dataset throughout
2. Update all API calls:
   - `data.regress()` → `data.fit(model="glm", X=dm)`
   - `stats['beta']` → `data.glm_betas[0]`
   - `Design_Matrix` → `DesignMatrix`
3. Update design matrix creation:
   - Use `onsets_to_dm()` with proper TR and run_length
   - Chain methods: `.convolve().add_dct_basis().add_poly()`
4. Update contrast computation:
   - Use `data.compute_contrasts()` method
5. Remove references to localizer-specific conditions

### 06_rsa.py - Representational Similarity Analysis
**Status**: Needs updates

**Changes Needed**:
1. Replace localizer dataset with Haxby
2. Update `Brain_Data` → `BrainData`
3. Update beta loading to use GLM results from Haxby
4. Update adjacency matrix operations if API changed
5. Update group analysis to use current inference methods

**Key Updates**:
- Use Haxby categories (faces, houses, etc.) instead of localizer conditions
- Load betas from GLM fits on Haxby data
- Update any deprecated methods

### 07_decoding.py - Multivariate Prediction
**Status**: Needs updates

**Changes Needed**:
1. Replace localizer dataset with Haxby
2. Update `Brain_Data` → `BrainData`
3. Update prediction API if changed
4. Use Haxby categories for classification examples
5. Update cross-validation examples

**Key Updates**:
- Use Haxby categories (e.g., faces vs houses) for classification
- Update `data.predict()` calls if API changed
- Update feature selection examples

### 09_encoding.py - Encoding Models
**Status**: Incomplete, needs completion

**Changes Needed**:
1. Complete the tutorial
2. Use Haxby dataset
3. Demonstrate ridge and banded ridge encoding models
4. Show current API for encoding model fitting

**Key Updates**:
- Use `data.fit(model="ridge", X=features)` for encoding
- Show banded ridge if available
- Use Haxby stimulus features

## Source Code Fixes to Flag

### Potential Issues Identified:
1. **DesignMatrix.append()** - May need to check axis parameter handling
2. **BrainData.fit()** - Ensure proper handling of DesignMatrix objects
3. **compute_contrasts()** - Verify contrast vector format
4. **fetch_haxby()** - Ensure returns are consistent and easy to use

### Testing Requirements:
1. Each tutorial should run end-to-end without errors
2. All imports should work
3. All method calls should match current API
4. Outputs should be reasonable (no obvious bugs)

## Implementation Order

1. ✅ Review and plan (current step)
2. Update 01_basics.py
3. Consolidate and update GLM tutorials (02-05 → 02_glm.py)
4. Update 06_rsa.py
5. Update 07_decoding.py
6. Complete 09_encoding.py
7. Test all tutorials
8. Flag any source code issues

## Notes

- Keep educational content and explanations
- Update code examples to be accurate and runnable
- Maintain consistency across tutorials
- Use Haxby dataset consistently for reproducibility
- Ensure tutorials demonstrate current best practices

