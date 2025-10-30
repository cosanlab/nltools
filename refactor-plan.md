# nltools v0.6.0 Refactoring Plan

**Purpose**: Strategic vision for v0.6.0 refactoring - the "what" and "why" without status tracking.

For task progress, see `refactor-todos.md`. For session context, see `refactor-progress.md`.

---

## Vision

**What We're Building**: Python neuroimaging library that wraps nilearn with intuitive APIs - think "requests library for neuroimaging". We don't reinvent, we simplify.

**Philosophy**: Pragmatic, user-friendly neuroimaging tools that leverage the excellent work done in nilearn while providing a cleaner, more intuitive interface.

---

## Architecture

**"Functional-core, imperative shell" pattern:**

- **Imperative shell** (`nltools/data/`): Brain_Data, Adjacency, Design_Matrix
  - User-facing classes with stateful APIs
  - Manage nifti images, masks, metadata
  - Provide convenient method chaining

- **Functional core** (`nltools/algorithms/`, `nltools/models/`, `stats.py`, `utils.py`):
  - Pure functions and sklearn-compatible classes
  - Hyperalignment, SRM, Ridge, GLM implementations
  - Statistical utilities and helpers

**Baseline**: v0.5.1 functionality must work or deprecate gracefully

---

## Core Strategies

### 1. Wrap nilearn, don't reimplement
- Leverage nilearn's battle-tested implementations
- Add value through convenience, not reimplementation
- Examples: `.apply_mask()` → nilearn.masking, `.extract_roi()` → NiftiLabelsMasker

### 2. Sklearn-style APIs
- `.fit()` and `.predict()` methods for models
- Store fitted models as `model_` attribute
- Results as attributes (e.g., `ridge_weights`, `glm_betas`)
- Cross-validation support via `cv` parameter

### 3. Efficient copying pattern
- `_shallow_copy_with_data()` helper for method chaining
- ~80% performance improvement in pipelines
- Share immutable objects (mask, nifti_masker)
- Copy only data arrays

### 4. Deprecation with migration path
- Strong `FutureWarning` (not silent `DeprecationWarning`)
- Clear error messages with migration instructions
- Backward compatibility during transition
- Document all changes in migration guide

### 5. Property decorators for cleaner API
- Convert methods → properties where appropriate
- Examples: `.shape()` → `.shape`, `.isempty()` → `.isempty`
- Improves discoverability and usage

---

## Priority Tiers

### Priority 1: Core Refactoring (MUST for v0.6.0)
- Delete deprecated Priority 3 files (brain_collection, model specs)
- Add deprecation stubs with clear errors
- Integrate nilearn for core functionality
- Backward compatibility via deprecation wrappers

### Priority 2: Polish & Enhancement (SHOULD for v0.6.0)
- Test organization (shell/core/support structure)
- Documentation migration (Sphinx → Jupyter Book)
- Sklearn-style fit/predict API
- Cross-validation support
- HyperAlignment class extraction
- Nilearn integration enhancements
- Codebase audit and cleanup

### Priority 3: Medium Priority (v0.6.0 or v0.6.1)
- Polars migration for Design_Matrix
- fit() inplace parameter + Fit dataclass
- Adjacency refactoring
- Plotting integration minimization

### Priority 4: Future (v0.7.0+)
- BrainCollection class design
- Advanced ML workflows
- Model class reimplementation

---

## Design Decisions

### Why fit/predict API?
- **Familiarity**: Sklearn users know this pattern
- **Composability**: Enables sklearn pipelines
- **Clarity**: Explicit model storage and reuse
- **Extensibility**: Easy to add new model types

### Why deprecate .regress()?
- Name doesn't indicate what model is used (GLM)
- Returns dict instead of storing as attributes
- Doesn't follow sklearn conventions
- Refactored as thin wrapper calling `fit(model='glm')`

### Why efficient copying?
- Method chaining is common in neuroimaging pipelines
- Deep copying masks/maskers is wasteful (immutable)
- Shallow copy with new data array is safe and fast
- Measured ~80% improvement in chained operations

### Why HyperAlignment class?
- Procrustes algorithm buried in `align()` function
- Sklearn users expect classes, not functions
- Enables model reuse (align new subjects to existing space)
- Better discoverability and testing

### Why test organization (shell/core/support)?
- Matches architectural pattern
- Enables targeted test running by category
- Clear separation: usage patterns vs computational correctness
- Faster development cycles (run only relevant tests)

---

## Success Criteria for v0.6.0

**Must Have:**
- All v0.5.1 public APIs work (even if deprecated)
- All tests pass with documented skips
- Clear deprecation warnings with migration path
- No performance regressions
- Documentation updated

**Nice to Have:**
- Reduced code size (eliminate redundancy)
- Improved test coverage
- Cleaner separation of concerns
- Enhanced functionality (CV support, fit/predict)

---

## Key Implementation Patterns

### Targeted Test-Driven Development
- Write/identify specific test first
- Run ONLY that test (or small subset)
- Implement minimal code to pass
- Verify with same targeted test
- Check regressions with related tests
- NEVER run full suite during development

### Efficient Copying
```python
def _shallow_copy_with_data(self, data):
    """Create shallow copy with new data array."""
    new = copy(self)  # Shallow copy
    new.data = data   # New data array
    return new        # Shares mask, nifti_masker, etc.
```

### Deprecation Wrapper
```python
def deprecated_method(self, *args, **kwargs):
    warnings.warn(
        "deprecated_method() is deprecated and will be removed in v0.7.0. "
        "Use new_method() instead.",
        FutureWarning,
        stacklevel=2
    )
    return self.new_method(*args, **kwargs)
```

### Sklearn-style fit/predict
```python
def fit(self, model='ridge', X=None, **kwargs):
    """Fit model to data."""
    # Create model instance
    # Fit to data
    # Store model as self.model_
    # Store training data as self.X_
    # Store results as attributes (ridge_*, glm_*)
    return self

def predict(self, X=None):
    """Predict using fitted model."""
    if X is None:
        X = self.X_  # Use training data
    return self.model_.predict(X)
```

---

## Git Tags for Reference

- `v0.6.0-test-refactor`: Original test implementations for deprecated methods
- `v0.6.0-docs-removal`: Sphinx docs removal reference point

---

*This document describes the strategic vision and should remain relatively stable. For task tracking, see `refactor-todos.md`. For session context and decisions, see `refactor-progress.md`.*

*Last updated: 2025-10-29*
