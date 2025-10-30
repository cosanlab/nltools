# Polars Migration Finalization Plan

**Date**: 2025-10-30
**Status**: Ready for implementation
**Estimated Time**: ~4-6 hours
**Branch**: uv-cleanup

---

## Executive Summary

**Goal**: Finalize the completed Polars migration by:
1. Optimizing code with polars selectors and idiomatic patterns
2. Improving error messages for better user experience
3. Removing all vestiges of old pandas implementation
4. Consolidating naming (design_matrix_new.py → design_matrix.py)
5. Documenting future optimization paths (pyarrow, GPU)
6. Archiving completed research/planning documents

**Current State**: ✅ Polars migration 100% complete (344/385 tests passing)
- All integration work finished (GLM, file_reader, Adjacency)
- Code is functional and well-tested
- Ready for optimization and cleanup

---

## Phase 1: Code Optimization (2-3 hours)

### 1.1 Use Polars Selectors for Column Filtering

**Why**: More declarative, optimizable by Polars engine, aligns with teaching philosophy

**Locations to update**:

#### `vif()` method (lines ~913-924)
**Before**:
```python
if exclude_polys and self.polys:
    cols_to_use = [c for c in self.columns if c not in self.polys]
else:
    cols_to_use = [c for c in self.columns if "poly_0" not in c]

subset_df = self._df.select(cols_to_use)
```

**After**:
```python
from polars import selectors as cs

if exclude_polys and self.polys:
    # More declarative: "select all columns except these patterns"
    subset_df = self._df.select(cs.exclude(self.polys))
else:
    # Select columns that don't contain "poly_0"
    subset_df = self._df.select([c for c in self.columns if "poly_0" not in c])
```

**Benefits**:
- More declarative code
- Polars can optimize selector-based operations
- Aligns with Eshin's tutorial patterns

#### `clean()` method (similar pattern)
Update to use `cs.exclude()` for consistency.

---

### 1.2 Simplify `convolve()` Multi-Kernel Case

**Current** (lines 456-476, complex logic with manual concatenation):
```python
# Multiple kernels: shape is (samples, n_kernels)
n_kernels = conv_func.shape[1]
all_convolved_data = {}

for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    for k_idx in range(n_kernels):
        kernel = conv_func[:, k_idx]
        convolved = np.convolve(col_data, kernel)[:n_rows]
        all_convolved_data[f"{col}_c{k_idx}"] = convolved

# Create new DataFrame with all convolved columns + non-convolved
convolved_df = pl.DataFrame(all_convolved_data)
non_convolved_df = self._df.select(non_convolved_cols) if non_convolved_cols else pl.DataFrame()

# Concatenate horizontally
if non_convolved_cols:
    new_df = pl.concat([convolved_df, non_convolved_df], how="horizontal")
else:
    new_df = convolved_df
```

**After** (use `.with_columns()` pattern from single-kernel case):
```python
# Multiple kernels
n_kernels = conv_func.shape[1]
convolved_series = []

for col in columns_to_convolve:
    col_data = self._df[col].to_numpy()
    for k_idx in range(n_kernels):
        kernel = conv_func[:, k_idx]
        convolved = np.convolve(col_data, kernel)[:n_rows]
        convolved_series.append(pl.Series(f"{col}_c{k_idx}", convolved))

# Drop original columns, add convolved variants
new_df = self._df.drop(columns_to_convolve).with_columns(convolved_series)
```

**Benefits**:
- 50% less code (no manual DataFrame creation)
- Consistent with single-kernel case (uses `.with_columns()`)
- Clearer intent (drop old, add new)
- No intermediate DataFrame allocations

**Rationale**: Matches the optimization already done for single-kernel case. Uses idiomatic Polars `.with_columns()` pattern.

---

### 1.3 Enhance Error Messages

**Philosophy**: Error messages should suggest fixes, not just state problems.

**Locations to update**:

#### `downsample()` - Line 286
**Before**:
```python
raise ValueError(
    f"Target ({target} Hz) must be less than current sampling_freq ({self.sampling_freq} Hz)"
)
```

**After**:
```python
raise ValueError(
    f"Downsampling target ({target} Hz) must be less than current sampling_freq "
    f"({self.sampling_freq} Hz). For upsampling, use .upsample() instead."
)
```

#### `upsample()` - Similar pattern
```python
raise ValueError(
    f"Upsampling target ({target} Hz) must be greater than current sampling_freq "
    f"({self.sampling_freq} Hz). For downsampling, use .downsample() instead."
)
```

#### `convolve()` - Add context about expected shapes
```python
raise ValueError(
    f"HRF function must be 1D (shape: (samples,)) or 2D (shape: (samples, n_kernels)). "
    f"Got shape: {conv_func.shape}. Tip: Use nltools.utils.glover_hrf() to generate HRFs."
)
```

#### `vif()` - Explain VIF thresholds
```python
raise ValueError(
    f"VIF threshold must be positive. Got: {threshold}. "
    f"Common thresholds: 5 (moderate collinearity), 10 (high collinearity)."
)
```

#### `clean()` - Explain correlation thresholds
```python
raise ValueError(
    f"Correlation threshold must be between 0 and 1. Got: {threshold}. "
    f"Typical values: 0.9 (strict), 0.95 (moderate), 0.99 (lenient)."
)
```

#### `add_poly()` - Clarify polynomial orders
```python
raise ValueError(
    f"Polynomial order must be >= 0. Got: {order}. "
    f"Common orders: 0 (intercept only), 1 (linear trend), 2 (quadratic)."
)
```

**Pattern**: All error messages now include:
1. What's wrong (clear statement)
2. What was provided (actual values)
3. What's expected (valid ranges/types)
4. Suggestion (how to fix or common values)

**Estimated effort**: ~30 minutes (10-15 error messages to enhance)

---

## Phase 2: Remove Old Design Matrix Vestiges (30 minutes)

### 2.1 Files to Delete

#### `design_matrix_old.py` (37KB)
- **Why**: Reference implementation no longer needed
- **Verification**: Only referenced in `design_matrix.py` import shim (which we'll replace)
- **Action**: Delete file

#### `test_design_matrix.py` (175 lines)
- **Why**: Old legacy tests superseded by test_design_matrix_new.py (1499 lines)
- **Verification**:
  - Old tests: 10 tests (basic functionality)
  - New tests: 68 tests (comprehensive Polars implementation)
- **Action**: Delete file

### 2.2 Verification Before Deletion

```bash
# Confirm no imports of design_matrix_old outside of design_matrix.py
uv run grep -r "design_matrix_old" nltools/ --exclude-dir=tests

# Confirm test_design_matrix_new has all functionality
uv run pytest nltools/tests/shell/test_design_matrix_new.py -v

# Expected: 68 tests passing
```

---

## Phase 3: Rename and Consolidate (45 minutes)

### 3.1 File Renaming Strategy

**Goal**: Make the Polars implementation THE canonical implementation.

**Step 1**: Back up current state (safety)
```bash
git add -A
git commit -m "checkpoint: before design_matrix consolidation"
```

**Step 2**: Remove old files
```bash
rm nltools/data/design_matrix_old.py
rm nltools/tests/shell/test_design_matrix.py
```

**Step 3**: Rewrite design_matrix.py to be the main implementation
```bash
# Replace the 798B shim with the actual implementation
mv nltools/data/design_matrix_new.py nltools/data/design_matrix.py.tmp
rm nltools/data/design_matrix.py  # Remove old shim
mv nltools/data/design_matrix.py.tmp nltools/data/design_matrix.py
```

**Step 4**: Update design_matrix.py header
Remove the "new" references and make it the canonical version:
```python
"""
DesignMatrix - Polars-based design matrix for neuroimaging analysis

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.
"""
```

**Step 5**: Rename test file
```bash
mv nltools/tests/shell/test_design_matrix_new.py nltools/tests/shell/test_design_matrix.py
```

**Step 6**: Update test file header and imports
```python
"""Tests for DesignMatrix (Polars implementation)"""
from nltools.data import DesignMatrix, Design_Matrix
```

### 3.2 Import Updates

**Files to check** (from grep results):
- `nltools/models/glm.py` - Uses `from nltools.data import DesignMatrix` ✅ (already correct)
- `nltools/data/adjacency.py` - Check if references old class
- `nltools/data/brain_data.py` - Check if references old class
- `nltools/data/__init__.py` - Update to import from design_matrix (not design_matrix_new)
- `nltools/file_reader.py` - Check import statement

**Expected changes in `__init__.py`**:
```python
# OLD
from .design_matrix import DesignMatrix, Design_Matrix, Design_Matrix_Series

# NEW (since design_matrix.py is now the implementation)
from .design_matrix import DesignMatrix, Design_Matrix, Design_Matrix_Series
# (No change needed! The shim file just gets replaced with real implementation)
```

---

## Phase 4: Update Documentation (30 minutes)

### 4.1 Update `refactor-todos.md`

Add new section for v0.7.0+ optimization tasks:

```markdown
## Priority 4: Future (v0.7.0+) - Polars Optimization

### Polars PyArrow Integration

| Task | Status | Effort | Benefits |
|------|--------|--------|----------|
| Add pyarrow as optional dependency | 📋 | 30 min | 10-100x faster Polars↔pandas conversion |
| Update `_to_pandas()` to use pyarrow path | 📋 | 15 min | Zero-copy conversions via Arrow |
| Add pyarrow fallback logic | 📋 | 15 min | Graceful degradation if not installed |
| Benchmark conversion performance | 📋 | 30 min | Quantify actual speedup |

**Action Items**:
1. Add optional dependency in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   performance = ["pyarrow>=15.0.0"]
   ```

2. Update `_to_pandas()` in design_matrix.py:
   ```python
   def _to_pandas(self) -> pd.DataFrame:
       """Convert to pandas DataFrame (zero-copy via pyarrow if available)."""
       try:
           import pyarrow  # noqa
           # Zero-copy conversion via Arrow (10-100x faster on large data)
           return self._df.to_pandas(use_pyarrow_extension_array=True)
       except ImportError:
           # Fallback to dict-based conversion (works without pyarrow)
           return pd.DataFrame(self._df.to_dict(as_series=False))
   ```

3. Document in migration guide:
   ```markdown
   ### Performance Optimization (Optional)

   For faster Polars↔pandas conversions, install pyarrow:
   ```bash
   pip install nltools[performance]
   # or
   pip install pyarrow
   ```

   This provides 10-100x speedup for `.downsample()`, `.upsample()`, and `.heatmap()`
   operations on large design matrices (1000+ rows).
   ```

**Estimated Impact**:
- Large design matrices (10,000 rows): 100x speedup (1s → 10ms)
- Medium design matrices (1,000 rows): 10x speedup (100ms → 10ms)
- Small design matrices (<100 rows): Minimal impact (already fast)

**Dependencies**: ~50MB (pyarrow)

**Risk**: Low (optional, has fallback)

---

### Polars GPU Support Integration

| Task | Status | Effort | Benefits |
|------|--------|--------|----------|
| Research current GPU integration architecture | 📋 | 1 hour | Understand existing patterns |
| Design Polars GPU path for DesignMatrix | 📋 | 2 hours | Plan GPU-accelerated operations |
| Add GPU backend selection in DesignMatrix | 📋 | 3 hours | Enable GPU convolution/stats |
| Test GPU correctness (results match CPU) | 📋 | 2 hours | Ensure numerical equivalence |
| Benchmark GPU performance vs CPU | 📋 | 1 hour | Quantify speedup |

**Action Items**:

1. **Review existing GPU integration** (nltools currently has some GPU support):
   ```bash
   # Search for existing GPU/CUDA patterns
   uv run grep -r "cuda\|gpu\|cupy" nltools/ --include="*.py"
   ```

2. **Design GPU selection pattern**:
   ```python
   # In design_matrix.py
   def _get_array_backend(self, use_gpu: bool = False):
       """Select numpy or cupy based on GPU preference and availability."""
       if use_gpu:
           try:
               import cupy as cp
               # Verify GPU is actually available
               if cp.cuda.runtime.getDeviceCount() > 0:
                   return cp
           except (ImportError, cp.cuda.runtime.CUDARuntimeError):
               import warnings
               warnings.warn("GPU requested but not available, falling back to CPU")
       return np
   ```

3. **Update convolution to support GPU**:
   ```python
   def convolve(self, ..., use_gpu: bool = False):
       """Convolve columns with HRF (CPU or GPU accelerated)."""
       xp = self._get_array_backend(use_gpu=use_gpu)

       for col in columns_to_convolve:
           # Convert to GPU array if using cupy
           col_data = xp.asarray(self._df[col].to_numpy())
           convolved = xp.convolve(col_data, conv_func)[:n_rows]
           # Convert back to CPU for Polars
           convolved_cpu = cp.asnumpy(convolved) if use_gpu else convolved
           convolved_series.append(pl.Series(col, convolved_cpu))
   ```

4. **Integration with existing GPU patterns**:
   - Check how Brain_Data currently handles GPU operations
   - Ensure consistent API: `use_gpu` parameter pattern
   - Share backend selection utilities

5. **Testing strategy**:
   ```python
   # In test_design_matrix.py
   @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
   def test_convolve_gpu_matches_cpu():
       """GPU convolution should produce identical results to CPU."""
       dm = DesignMatrix(...)
       dm_cpu = dm.convolve(hrf, use_gpu=False)
       dm_gpu = dm.convolve(hrf, use_gpu=True)
       assert np.allclose(dm_cpu.to_numpy(), dm_gpu.to_numpy())
   ```

**Expected Speedup**:
- Convolution (1000+ timepoints): 5-10x
- Statistics (correlations, VIF): 2-5x
- Resampling operations: Minimal (I/O bound)

**Dependencies**:
- cupy (matches CUDA version)
- polars with GPU support (experimental in Polars 1.x)

**Risk**: Medium (GPU support in Polars is evolving)

**Timeline**: v0.7.0 (after CPU implementation is stable)

---

### Polars-Native Resampling

| Task | Status | Effort | Benefits |
|------|--------|--------|----------|
| Replace stats.downsample with Polars `.group_by_dynamic()` | 📋 | 2 hours | 2-5x speedup, native Polars |
| Replace stats.upsample with Polars interpolation | 📋 | 2 hours | Cleaner code, better performance |
| Update DesignMatrix.downsample() to use Polars-native | 📋 | 1 hour | Remove stats.py dependency |
| Update DesignMatrix.upsample() to use Polars-native | 📋 | 1 hour | Remove stats.py dependency |
| Test equivalence with old implementation | 📋 | 1 hour | Ensure backward compatibility |

**Action Items**:

1. **Implement Polars-native downsampling**:
   ```python
   def downsample(self, target: float, sampling_freq: Optional[float] = None) -> "DesignMatrix":
       """Downsample using Polars group_by_dynamic (native, fast)."""
       if sampling_freq is None:
           sampling_freq = self.sampling_freq

       # Calculate time intervals
       interval = f"{1/target}s"

       # Use Polars group_by_dynamic for efficient downsampling
       downsampled_df = (
           self._df
           .with_row_count("time_index")
           .with_columns((pl.col("time_index") / sampling_freq).alias("time"))
           .group_by_dynamic("time", every=interval)
           .agg([pl.col(col).mean() for col in self._get_data_columns()])
           .drop("time")
       )

       return self._copy_with(data=downsampled_df, sampling_freq=target)
   ```

2. **Implement Polars-native upsampling**:
   ```python
   def upsample(self, target: float, method: str = "linear") -> "DesignMatrix":
       """Upsample using Polars interpolation (native, fast)."""
       # Polars interpolation with specified method
       upsampled_df = (
           self._df
           .with_row_count("index")
           .upsample(time_column="index", every=f"{1/target}s")
           .interpolate(method=method)  # 'linear', 'nearest', etc.
       )

       return self._copy_with(data=upsampled_df, sampling_freq=target)
   ```

**Benefits**:
- Remove dependency on stats.py for resampling
- 2-5x faster (native Polars operations)
- Cleaner code (no pandas conversions)
- More options (Polars has richer interpolation methods)

**Risk**: Low (Polars group_by_dynamic is stable)

**Timeline**: v0.7.0

```

### 4.2 Update `refactor-progress.md`

Add final section documenting completion:

```markdown
## Polars Migration Finalization - COMPLETE ✅ (2025-10-30)

### What Was Completed

**1. Code Optimization**
- ✅ Added polars selectors to `vif()` and `clean()` methods
- ✅ Simplified `convolve()` multi-kernel case (50% code reduction)
- ✅ Enhanced 12 error messages with actionable suggestions
- ✅ Applied idiomatic Polars patterns throughout

**2. File Consolidation**
- ✅ Removed `design_matrix_old.py` (37KB pandas reference)
- ✅ Removed `test_design_matrix.py` (175 lines old tests)
- ✅ Renamed `design_matrix_new.py` → `design_matrix.py` (canonical implementation)
- ✅ Renamed `test_design_matrix_new.py` → `test_design_matrix.py` (canonical tests)
- ✅ Updated all imports and references

**3. Documentation**
- ✅ Updated `refactor-todos.md` with v0.7.0+ optimization paths
- ✅ Added detailed pyarrow integration action items
- ✅ Added detailed GPU support integration plan
- ✅ Added Polars-native resampling plan
- ✅ Archived completed polars markdown files to `claude-research/`

**4. Test Verification**
- ✅ All 68 DesignMatrix tests passing
- ✅ All 344 tests passing (5 skipped, 36 deselected)
- ✅ No regressions introduced

**Files Archived** (to `claude-research/`):
- `polars-integration-status.md`
- `polars-refactoring-summary.md`
- `polars-code-review.md`
- `glm-integration-summary.md`
- `claude-guidelines/polars-migration.md`
- `claude-guidelines/polars-migration-v2.md`
- `claude-guidelines/polars-migration-v3-tdd.md`

**Time**: ~4 hours (optimization, cleanup, consolidation, documentation)

**Impact**:
- Cleaner, more maintainable codebase
- Better error messages for users
- Clear path forward for v0.7.0 optimizations
- All Polars work complete and ready for release

---

## Future Work (v0.7.0+)

See `refactor-todos.md` for detailed action items:
- PyArrow integration (10-100x faster conversions)
- GPU support integration (5-10x faster operations)
- Polars-native resampling (2-5x faster, cleaner code)
```

### 4.3 Update `CLAUDE.md` (if needed)

Check if any references to "design_matrix_new" or old file structure exist. Update to reflect new consolidated structure.

---

## Phase 5: Archive Completed Research (15 minutes)

### 5.1 Files to Archive

Move these files to `claude-research/` directory:

**Root markdown files**:
- `polars-integration-status.md` → `claude-research/polars-integration-status.md`
- `polars-refactoring-summary.md` → `claude-research/polars-refactoring-summary.md`
- `polars-code-review.md` → `claude-research/polars-code-review.md`
- `glm-integration-summary.md` → `claude-research/glm-integration-summary.md`

**Claude-guidelines files**:
- `claude-guidelines/polars-migration.md` → `claude-research/polars-migration-v1.md`
- `claude-guidelines/polars-migration-v2.md` → `claude-research/polars-migration-v2.md`
- `claude-guidelines/polars-migration-v3-tdd.md` → `claude-research/polars-migration-v3-tdd.md`

### 5.2 Archival Script

```bash
# Create archive directory
mkdir -p claude-research/

# Move root polars files
mv polars-integration-status.md claude-research/
mv polars-refactoring-summary.md claude-research/
mv polars-code-review.md claude-research/
mv glm-integration-summary.md claude-research/

# Move guidelines files with rename
mv claude-guidelines/polars-migration.md claude-research/polars-migration-v1.md
mv claude-guidelines/polars-migration-v2.md claude-research/polars-migration-v2.md
mv claude-guidelines/polars-migration-v3-tdd.md claude-research/polars-migration-v3-tdd.md

# Add a README to the archive
cat > claude-research/README.md << 'EOF'
# Claude Research Archive

This directory contains completed research, planning documents, and implementation
notes that are no longer actively referenced but are preserved for historical context.

## Polars Migration (v0.6.0)

Completed research and planning documents for the pandas → Polars migration:
- `polars-migration-v1.md` - Initial migration plan
- `polars-migration-v2.md` - Revised plan with composition pattern
- `polars-migration-v3-tdd.md` - Final TDD implementation plan
- `polars-integration-status.md` - Integration tracking document
- `polars-refactoring-summary.md` - Phase 1 refactoring summary
- `polars-code-review.md` - Code review and optimization opportunities
- `glm-integration-summary.md` - GLM integration fix details

**Status**: ✅ Complete (2025-10-30)
**Result**: 68 tests, 100% passing, all integration work finished

See `refactor-progress.md` for implementation details.
EOF

# Verify archive
ls -lh claude-research/
```

### 5.3 Update .gitignore (if needed)

Ensure `claude-research/` is tracked (not ignored) so archived research is preserved in git history:

```bash
# Check if claude-research is ignored
git check-ignore claude-research/

# If ignored, update .gitignore to track it
# (Most likely it's not ignored, but verify)
```

---

## Phase 6: Final Verification (30 minutes)

### 6.1 Test Suite Verification

```bash
# Run all DesignMatrix tests (should be 68 passing)
uv run pytest nltools/tests/shell/test_design_matrix.py -v

# Run tier1 tests in parallel (fast verification)
uv run pytest -m tier1 -n auto

# Check for any import errors
uv run python -c "from nltools.data import DesignMatrix, Design_Matrix; print('Imports OK')"

# Check that old files are truly deleted
test ! -f nltools/data/design_matrix_old.py && echo "✅ Old file deleted"
test ! -f nltools/data/design_matrix_new.py && echo "✅ New file renamed"
test ! -f nltools/tests/shell/test_design_matrix_new.py && echo "✅ Test file renamed"
```

### 6.2 Import Verification

Test that all imports work correctly:

```python
# Test script to verify imports
from nltools.data import DesignMatrix, Design_Matrix, Design_Matrix_Series
from nltools.models import Glm
import numpy as np

# Test basic functionality
dm = DesignMatrix({'a': [1, 2, 3]}, sampling_freq=1.0)
assert dm.shape == (3, 1)
assert 'a' in dm.columns
print("✅ DesignMatrix imports and basic functionality OK")

# Test backward compatibility alias
dm2 = Design_Matrix({'b': [4, 5, 6]}, sampling_freq=1.0)
assert dm2.shape == (3, 1)
print("✅ Design_Matrix alias OK")

# Test GLM integration
print("✅ All imports verified")
```

### 6.3 Documentation Review

- [ ] Check that migration-guide.md references are accurate
- [ ] Verify refactor-todos.md has correct future tasks
- [ ] Confirm refactor-progress.md is up to date
- [ ] Ensure CLAUDE.md reflects new structure (if it references old files)

---

## Success Criteria

### Code Quality
- ✅ All polars selectors used where appropriate
- ✅ All error messages enhanced with suggestions
- ✅ Multi-kernel convolution simplified
- ✅ Code follows idiomatic Polars patterns

### File Structure
- ✅ `design_matrix.py` is the canonical implementation (not a shim)
- ✅ `test_design_matrix.py` is the canonical test file
- ✅ No `design_matrix_old.py` or `design_matrix_new.py` files exist
- ✅ All imports reference correct file locations

### Testing
- ✅ All 68 DesignMatrix tests passing
- ✅ All 344 tier1 tests passing
- ✅ No import errors
- ✅ No test failures

### Documentation
- ✅ `refactor-todos.md` updated with v0.7.0 action items
- ✅ `refactor-progress.md` documents finalization
- ✅ Research files archived to `claude-research/`
- ✅ Migration guide accurate and complete

---

## Risk Assessment

**Low Risk**:
- File renaming (git handles this well)
- Archive file moves (non-code files)
- Error message enhancements (user-facing only)

**Medium Risk**:
- Polars selector changes (changes logic, but well-tested)
- Multi-kernel convolution refactor (complex logic, needs careful testing)

**Mitigation**:
- Create checkpoint commit before starting
- Test incrementally after each change
- Run full test suite before final commit
- Review diffs carefully

---

## Estimated Timeline

| Phase | Tasks | Time | Complexity |
|-------|-------|------|------------|
| Phase 1 | Code optimization | 2-3 hours | Medium |
| Phase 2 | Remove old files | 30 min | Low |
| Phase 3 | Rename/consolidate | 45 min | Low |
| Phase 4 | Update docs | 30 min | Low |
| Phase 5 | Archive research | 15 min | Low |
| Phase 6 | Verification | 30 min | Low |
| **Total** | | **4.5-6 hours** | |

**Best approach**: Work in phases with commits after each phase for safety.

---

## Commit Strategy

**Checkpoint (before starting)**:
```bash
git add -A
git commit -m "checkpoint: before polars migration finalization"
```

**Phase 1 (code optimization)**:
```bash
git add nltools/data/design_matrix_new.py
git commit -m "refactor(polars): Optimize DesignMatrix with selectors and enhanced errors"
```

**Phase 2-3 (consolidation)**:
```bash
git add nltools/data/ nltools/tests/shell/
git commit -m "refactor(polars): Consolidate design_matrix files and remove old implementation"
```

**Phase 4-5 (documentation)**:
```bash
git add refactor-todos.md refactor-progress.md claude-research/ docs/
git commit -m "docs(polars): Update refactor docs and archive completed research"
```

**Phase 6 (final verification)**:
```bash
git add -A
git commit -m "test(polars): Verify polars migration finalization complete"
```

---

## Next Steps After Completion

1. **Review with Eshin**: Present changes and get approval
2. **Consider merge to master**: If v0.6.0 release is ready
3. **Begin next priority**: Check refactor-todos.md for next task
4. **Update changelog**: Document Polars migration completion

---

**Status**: Ready for implementation
**Estimated completion**: 4-6 hours of focused work
**Risk level**: Low (well-tested, incremental approach)
**Test coverage**: Excellent (68 DesignMatrix tests, 344 total tests)

