# Release Preparation Plan

**Goal**: Prepare and execute v0.6.0 release (version bump, CHANGELOG, Git tags, PyPI)

**Estimated Effort**: 1-2 hours

**Priority**: HIGH (final step before release)

**Status**: 📋 READY FOR SUB-AGENT

---

## Context

After all verification is complete, we need to:
1. Update version to 0.6.0
2. Create/update CHANGELOG.md
3. Create Git tag
4. Build and test package
5. Release to PyPI

**IMPORTANT**: Only proceed if all previous phases (Phase 4, Phase 5, Pre-Release Verification) are complete.

---

## Pre-Release Checklist

Before starting, verify:
- [ ] Phase 4 (Code Quality) complete
- [ ] Phase 5 (Documentation) complete
- [ ] Pre-Release Verification complete
- [ ] All tests passing
- [ ] User approval obtained

---

## Task 1: Update Version

### Location
- `pyproject.toml`

### Steps

1. **Read current version**:
   ```bash
   grep "version" pyproject.toml
   ```

2. **Update version to 0.6.0**:
   ```toml
   [project]
   version = "0.6.0"
   ```

3. **Verify version update**:
   ```bash
   grep "version" pyproject.toml
   uv run python -c "import nltools; print(nltools.__version__)"
   ```
   **Expected**: Version shows 0.6.0

### Success Criteria
- [ ] Version updated to 0.6.0 in pyproject.toml
- [ ] Version importable and correct

---

## Task 2: Create/Update CHANGELOG.md

### Location
- `CHANGELOG.md` (root directory)

### Steps

1. **Check if CHANGELOG.md exists**:
   ```bash
   ls -la CHANGELOG.md
   ```

2. **Create or update CHANGELOG.md**:

   If file doesn't exist, create:
   ```markdown
   # Changelog
   
   All notable changes to nltools will be documented in this file.
   
   The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
   and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
   
   ## [0.6.0] - 2025-01-XX
   
   ### Added
   - GPU-accelerated inference module (10-100× speedup for permutation tests)
   - CPU parallelization for permutation tests (4-8× speedup with n_jobs=-1)
   - Fit dataclass for immutable results and serialization
   - Polars DesignMatrix backend (2-5× speedup for resampling operations)
   - OnlineBootstrapStats for memory-efficient bootstrap calculations
   - New inference module functions: `isc_permutation_test()`, `isc_group_permutation_test()`, etc.
   - `BrainData.fit(inplace=False)` parameter to return Fit objects
   
   ### Changed
   - `isc()`, `isc_group()`, `isfc()` now use inference module internally (wrappers maintained)
   - `DesignMatrix` uses Polars instead of pandas (automatic migration, backward compatible)
   - `BrainData.fit()` supports `inplace=False` parameter (default=True for backward compatibility)
   - Return keys: `null_dist` → `null_distribution` (wrapper handles mapping)
   - Improved error handling and cross-backend determinism
   
   ### Deprecated
   - `stats.pearson()` - Use `scipy.stats.pearsonr` or inference module correlation tests
   - Direct use of `stats.isc()` etc. - Use `inference` module functions directly for better performance
   
   ### Removed
   - `stats.regress()` - Use `nltools.models.Glm` or `BrainData.fit(model='glm')`
   - `stats.regress_permutation()` - Use inference module permutation tests
   - `stats.correlation()` - Use `correlation_permutation_test()` from inference module
   - `_permute_sign()`, `_permute_func()` helper functions (internal, replaced)
   - ARMA functionality and statsmodels dependency
   
   ### Fixed
   - Return key mapping for `isc()` and `isc_group()` wrappers (`null_dist` → `null_distribution`)
   - Int64→int32 conversions for FSL/SPM compatibility
   - Nilearn 0.12 compatibility warnings
   - PyTables performance warnings
   - Cross-backend determinism (0.000% variance between NumPy and PyTorch)
   
   ### Performance
   - Inference module: 4-8× CPU speedup, 10-100× GPU speedup
   - DesignMatrix: 2-5× speedup for resampling operations
   - Memory-efficient bootstrap calculations
   - Optimized test suite (tier1 runtime ~36s)
   
   ### Documentation
   - Updated migration guide with inference module examples
   - Added Fit dataclass usage examples
   - Added GPU acceleration guide
   - Created breaking changes summary
   - Updated API documentation
   ```

   If file exists, add new section at top.

3. **Verify CHANGELOG.md format**:
   ```bash
   head -50 CHANGELOG.md
   ```

### Success Criteria
- [ ] CHANGELOG.md created/updated
- [ ] All major changes documented
- [ ] Format follows Keep a Changelog standard
- [ ] Breaking changes clearly marked

---

## Task 3: Git Operations

### Steps

1. **Check Git status**:
   ```bash
   git status
   ```

2. **Review changes**:
   ```bash
   git diff --stat
   ```

3. **Stage all changes**:
   ```bash
   git add .
   ```

4. **Create commit** (if not already committed):
   ```bash
   git commit -m "Release v0.6.0

   - GPU-accelerated inference module (10-100× speedup)
   - CPU parallelization for permutation tests (4-8× speedup)
   - Fit dataclass for immutable results
   - Polars DesignMatrix backend (2-5× speedup)
   - Stats migration complete (isc, isc_group, isfc)
   - Breaking changes documented in migration guide
   
   See CHANGELOG.md for full details."
   ```

5. **Create Git tag**:
   ```bash
   git tag -a v0.6.0 -m "Release v0.6.0

   Major release with GPU acceleration, inference module, and performance improvements.
   
   See CHANGELOG.md for full details."
   ```

6. **Verify tag**:
   ```bash
   git tag -l "v0.6.0"
   git show v0.6.0
   ```

### Success Criteria
- [ ] All changes committed
- [ ] Git tag v0.6.0 created
- [ ] Tag message includes release notes

### Notes
- **DO NOT PUSH** until user explicitly approves
- Tag should be annotated (`-a` flag)
- Commit message should be descriptive

---

## Task 4: Build and Test Package

### Steps

1. **Build package**:
   ```bash
   uv build
   ```
   **Expected**: Creates `dist/nltools-0.6.0-py3-none-any.whl` and `dist/nltools-0.6.0.tar.gz`

2. **Verify build artifacts**:
   ```bash
   ls -lh dist/nltools-0.6.0*
   ```

3. **Test install in clean environment**:
   ```bash
   # Create temporary virtual environment
   python -m venv /tmp/test_nltools_install
   source /tmp/test_nltools_install/bin/activate
   
   # Install from wheel
   pip install dist/nltools-0.6.0-py3-none-any.whl
   
   # Test import
   python -c "import nltools; print(f'Installed version: {nltools.__version__}')"
   
   # Run basic test
   python -c "from nltools.algorithms.inference import one_sample_permutation_test; import numpy as np; result = one_sample_permutation_test(np.random.randn(30, 100), n_permute=10); print('Test passed:', 'mean' in result)"
   
   # Cleanup
   deactivate
   rm -rf /tmp/test_nltools_install
   ```
   **Expected**: Package installs and basic functionality works

4. **Verify package contents**:
   ```bash
   # Check wheel contents
   python -m zipfile -l dist/nltools-0.6.0-py3-none-any.whl | head -20
   
   # Check source distribution
   tar -tzf dist/nltools-0.6.0.tar.gz | head -20
   ```

### Success Criteria
- [ ] Package builds successfully
- [ ] Build artifacts created (wheel and source)
- [ ] Package installs in clean environment
- [ ] Basic functionality works after install

---

## Task 5: TestPyPI Upload (Optional but Recommended)

### Steps

1. **Check TestPyPI credentials**:
   ```bash
   # TestPyPI URL
   echo "TestPyPI URL: https://test.pypi.org/project/nltools/"
   ```

2. **Upload to TestPyPI**:
   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ dist/nltools-0.6.0*
   ```
   **Note**: Requires TestPyPI credentials (user must provide)

3. **Test install from TestPyPI**:
   ```bash
   python -m venv /tmp/test_testpypi_install
   source /tmp/test_testpypi_install/bin/activate
   
   pip install --index-url https://test.pypi.org/simple/ nltools==0.6.0
   python -c "import nltools; print(f'TestPyPI version: {nltools.__version__}')"
   
   deactivate
   rm -rf /tmp/test_testpypi_install
   ```

### Success Criteria
- [ ] Package uploaded to TestPyPI (if credentials available)
- [ ] Package installs from TestPyPI
- [ ] No issues found

### Notes
- **Optional**: TestPyPI upload is recommended but not required
- User must provide TestPyPI credentials
- TestPyPI is separate from PyPI (separate accounts)

---

## Task 6: PyPI Release (Final Step)

### Steps

**IMPORTANT**: Only proceed after user explicitly approves PyPI release.

1. **Final verification**:
   - [ ] All tests passing
   - [ ] Documentation builds
   - [ ] Version correct
   - [ ] CHANGELOG complete
   - [ ] Git tag created
   - [ ] Package builds successfully
   - [ ] TestPyPI upload successful (if done)

2. **Upload to PyPI**:
   ```bash
   uv publish dist/nltools-0.6.0*
   ```
   **Note**: Requires PyPI credentials (user must provide)

3. **Verify PyPI upload**:
   ```bash
   # Check PyPI page
   echo "PyPI URL: https://pypi.org/project/nltools/0.6.0/"
   ```

4. **Test install from PyPI**:
   ```bash
   python -m venv /tmp/test_pypi_install
   source /tmp/test_pypi_install/bin/activate
   
   pip install nltools==0.6.0
   python -c "import nltools; print(f'PyPI version: {nltools.__version__}')"
   
   deactivate
   rm -rf /tmp/test_pypi_install
   ```

### Success Criteria
- [ ] Package uploaded to PyPI
- [ ] Package installs from PyPI
- [ ] Version correct on PyPI
- [ ] No issues found

### Notes
- **DO NOT PROCEED** without explicit user approval
- PyPI upload is permanent (cannot be deleted easily)
- User must provide PyPI credentials

---

## Task 7: GitHub Release

### Steps

1. **Push commits and tag** (after user approval):
   ```bash
   git push origin main
   git push origin v0.6.0
   ```

2. **Create GitHub Release**:
   - Go to GitHub repository
   - Click "Releases" → "Draft a new release"
   - Select tag: `v0.6.0`
   - Title: `v0.6.0 - GPU Acceleration & Inference Module`
   - Description: Copy from CHANGELOG.md [0.6.0] section
   - Check "Set as the latest release"
   - Click "Publish release"

### Success Criteria
- [ ] Commits pushed to GitHub
- [ ] Tag pushed to GitHub
- [ ] GitHub release created
- [ ] Release notes included

---

## Release Checklist Summary

Before releasing, verify:
- [ ] Version updated to 0.6.0
- [ ] CHANGELOG.md created/updated
- [ ] All tests passing
- [ ] Documentation builds
- [ ] Git tag created
- [ ] Package builds successfully
- [ ] TestPyPI upload successful (optional)
- [ ] User approval obtained
- [ ] PyPI upload successful
- [ ] GitHub release created

---

## Success Criteria for Release Preparation

- [ ] Version updated correctly
- [ ] CHANGELOG.md complete
- [ ] Git tag created
- [ ] Package builds and installs correctly
- [ ] TestPyPI upload successful (if done)
- [ ] PyPI upload successful (after approval)
- [ ] GitHub release created
- [ ] All release artifacts verified

---

## Notes

- **User Approval Required**: Do not push to Git or upload to PyPI without explicit approval
- **Credentials**: User must provide PyPI/TestPyPI credentials
- **Reversibility**: PyPI uploads are permanent - verify everything before uploading
- **Backup**: Consider creating backup branch before release

---

## Reference Files

- `pyproject.toml` - Version configuration
- `CHANGELOG.md` - Release notes
- `v0.6.0-VERIFICATION.md` - Verification checklist
- `v0.6.0-ACTION-PLAN.md` - Overall action plan

---

**Last Updated**: 2025-01-03  
**Status**: Ready for sub-agent execution  
**IMPORTANT**: Requires user approval before PyPI upload and Git push

