# nltools Design Philosophy & Decisions

*Reference document for understanding the design decisions behind nltools v0.6.0 refactoring*

---

## Core Design Philosophy

**nltools as "The requests library for neuroimaging"**

We don't reinvent the wheel, we make the wheel easier to use. Our mission is to wrap lower-level tools (primarily nilearn) with intuitive APIs that make fMRI analysis more accessible.

**Architecture Pattern: "Functional-core, imperative shell"**
- **Imperative shell** (`nltools/data/`): Stateful classes (Brain_Data, Adjacency, DesignMatrix) that hold data and coordinate operations
- **Functional core**: Pure functions for computations (stats.py, utils.py, etc.)
- **v0.5.1 = Baseline**: This is our compatibility target - everything from v0.5.1 must work or deprecate gracefully
- **Post-v0.5.1 features**: Deferred to Priority 3 (Model class, Brain_Collection)

---

## The nilearn Integration Philosophy

**Context**: "Can we do this more easily in nilearn?"

### Golden Rule: ALWAYS check nilearn first

Our integration rules:
1. **If nilearn has it** → wrap it for better UX
2. **If nilearn doesn't have it** → consider if we really need it
3. **If we must implement** → follow nilearn patterns for consistency

### Current nilearn Dependencies We Leverage

**Core functionality:**
- `NiftiMasker`: Core data loading and masking
- `FirstLevelModel`: GLM implementation
- `NiftiLabelsMasker`: ROI extraction (new in our refactor)
- Image functions: `smooth_img`, `resample_to_img`
- Plotting: Most visualization functions

### When to Push Back

If someone (including Claude) suggests reimplementing something nilearn provides, **challenge them**. The correct response is to find the nilearn function and wrap it.

**Anti-pattern to avoid:**
```python
# BAD: Reimplementing what nilearn provides
def custom_smooth(data, kernel):
    # Custom smoothing implementation
    ...
```

**Correct pattern:**
```python
# GOOD: Leverage nilearn
from nilearn.image import smooth_img
smoothed = Brain_Data(smooth_img(brain.to_nifti(), fwhm=6))
```

---

## Why We Implemented `.regress()` This Way

**Context**: "Why did you implement regress in this way?"

### Decision: Wrap `nilearn.glm.first_level.FirstLevelModel`

**Rationale:**
1. **Don't reinvent the wheel**: nilearn already has robust, tested GLM implementation
2. **Store as attributes**: Changed from returning dict to storing `.glm_betas`, `.glm_t`, etc. as attributes for easier access and consistency with other methods
3. **Override defaults**: We disable smoothing/scaling/drift because users should control these preprocessing steps explicitly
4. **DesignMatrix required**: Forces explicit experimental design specification rather than implicit assumptions

### API Evolution

```python
# Old pattern (v0.5.1)
brain.X = design_matrix
results_dict = brain.regress()
betas = results_dict['beta']
t_stats = results_dict['t']

# New pattern (v0.6.0)
brain.regress(design_matrix)  # Stores results as attributes
betas = brain.glm_betas      # Direct attribute access
t_stats = brain.glm_t        # Consistent with other Brain_Data attributes
```

### Trade-offs

**Pros:**
- ✅ Cleaner, more consistent API
- ✅ Leverages well-tested nilearn implementation
- ✅ Reduces maintenance burden
- ✅ Follows "wrap don't reinvent" philosophy

**Cons:**
- ❌ Breaking change from v0.5.1
- ❌ Requires users to update their code

**Decision**: Worth it for long-term maintainability and consistency.

---

## The Efficient Copying Implementation

**Context**: "How do we support method chaining without deep copy overhead?"

### The Problem

Many Brain_Data methods return modified copies to support method chaining:
```python
result = brain.scale(100).standardize().threshold(3.0)
```

Without optimization, each method would trigger a full deep copy, including:
- Large data arrays (potentially GBs)
- NiftiMasker objects (expensive to copy)
- Mask arrays
- Metadata DataFrames

This created ~3x overhead for simple chains.

### The Solution: Hybrid Shallow Copy (2025-10-28)

**Implementation:**
1. **`_shallow_copy_with_data()` method**: Creates new Brain_Data instance that shares immutable objects
2. **Smart attribute handling**:
   - **Share**: mask, nifti_masker (immutable, expensive to copy)
   - **Copy**: X, Y DataFrames (small, mutable, user might modify)
   - **Defer**: data array (let each method decide when to copy)

### Example Pattern

```python
def scale(self, scale_val=100.0):
    """Scale data to a target mean value."""
    out = self._shallow_copy_with_data()  # Fast object creation (shares immutables)
    out.data = self.data.copy()           # Only copy data when actually modifying
    out.data = out.data / out.data.mean() * scale_val
    return out
```

### Performance Impact

**Before optimization:**
- 3-method chain: ~3x full deep copy overhead
- Large datasets (1GB): seconds of copying overhead

**After optimization:**
- 3-method chain: ~80% reduction in copy overhead
- Large datasets: milliseconds of object creation overhead
- Data array only copied when modified (optimal)

### Key Insight

**Share immutable, expensive objects; copy mutable, cheap objects; defer data copies until needed.**

This pattern enables efficient method chaining without sacrificing safety or predictability.

---

## The Deprecation Strategy

**Context**: Methods moved to future Model class

### Methods Deprecated in v0.6.0

These methods are removed from Brain_Data and will be implemented in a future Model class:
- `.predict()` → Model class (ML workflows)
- `.ttest()` → Model class (statistical testing)
- `.randomise()` → Model class (permutation testing)
- `.predict_multi()` → Model class (searchlight/multi-ROI)

### Implementation: Explicit Deprecation

Rather than silently removing these methods, we added deprecation stubs:

```python
def predict(self, *args, **kwargs):
    raise NotImplementedError(
        "The .predict() method has been moved to the Model class. "
        "This method will be available in a future release when the Model class is implemented."
    )
```

### Rationale

**Why explicit deprecation stubs?**
1. **Prevents silent failures**: Users get clear error messages, not AttributeError
2. **Communicates migration path**: Error message explains where functionality went
3. **Documents future roadmap**: Serves as TODO list for Model class implementation
4. **Maintains discoverability**: Methods still appear in dir(brain_data)

**Why not warnings first?**
- v0.6.0 is explicitly a breaking release
- These features were added after v0.5.1 (not in baseline)
- Better to be explicit than allow code to run with warnings users might miss

### Testing Deprecated Methods

We use pytest.raises to verify deprecation stubs work correctly:

```python
def test_predict_deprecated(brain_data):
    """Verify .predict() raises NotImplementedError with helpful message."""
    with pytest.raises(NotImplementedError, match="Model class"):
        brain_data.predict()
```

See git tag `v0.6.0-test-refactor` for the test implementations we created.

---

## Code Patterns to Follow

### Input Validation

**DO**: Use centralized validation
```python
from nltools.data._validation import validate_data_type
data = validate_data_type(data)
```

**DON'T**: Inline validation everywhere
```python
if not isinstance(data, Brain_Data):
    if isinstance(data, str):
        # ... lots of conversion logic
```

### Leveraging nilearn

**DO**: Import and use nilearn functions
```python
from nilearn.image import smooth_img, resample_to_img
smoothed = smooth_img(brain.to_nifti(), fwhm=6)
```

**DON'T**: Reimplement image operations
```python
def my_smooth_function(data, kernel_size):
    # Custom convolution implementation
    ...
```

### Memory Management

**DO**: Process large data in chunks
```python
for chunk in np.array_split(indices, n_chunks):
    results.append(process(brain_data[chunk]))
```

**DON'T**: Load everything into memory
```python
all_results = [process(brain_data[i]) for i in range(len(brain_data))]  # OOM risk
```

---

## When to Reference This File

**From CLAUDE.md**: Link here when discussing:
- "Why did we implement X this way?"
- "Can we simplify this?"
- "Should we use nilearn for this?"

**During development**: Review before:
- Adding new Brain_Data methods
- Reimplementing functionality that might exist in nilearn
- Making breaking API changes
- Optimizing performance of copy-heavy operations

**During code review**: Check that PRs follow:
- nilearn-first philosophy
- Efficient copying patterns
- Explicit deprecation strategy

---

*Last updated: 2025-10-28*
*For current status and active work, see CLAUDE.md*
