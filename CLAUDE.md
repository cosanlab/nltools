# CLAUDE.md - nltools Development Guide

## Project Overview

### Mission
We are developing `nltools` - a Python library that provides an intuitive way to work with and analyze fMRI data. The library serves as a user-friendly wrapper around lower-level neuroimaging tools, reducing boilerplate code and making neuroimaging analysis more accessible to researchers.

### Core Philosophy
- **"Functional-core, imperative shell"** design pattern:
  - **Imperative shell** (`nltools/data/`): Stateful classes that manage data and coordinate operations
  - **Functional core** (all other modules): Pure functions for computations, statistics, and utilities
- **User-centered design**: Prioritize ease of use and clear APIs over complex optimizations
- **Leverage existing tools**: Build upon nilearn, scikit-learn, and other established libraries rather than reimplementing functionality

## Architecture & Codebase Structure

### Imperative Classes (nltools/data/)

**Brain_Data** (`brain_data.py`)
- **Purpose**: Primary class for 3D/4D neuroimaging data stored as vectorized 2D arrays (observations × voxels)
- **Key attributes**:
  - `.data`: numpy array (n_observations × n_voxels) - the core data matrix
  - `.Y`: pandas DataFrame - labels/targets for prediction
  - `.X`: pandas DataFrame - design matrix for GLM (being refactored to `.design_matrix`)
  - `.mask`: NiftiMasker object - handles spatial transformations
  - `.file_name`: list of source files
  - `.nifti_masker`: the masker object used for transformations
- **Common operations**:
  - Loading: from files, nibabel objects, URLs, or other Brain_Data
  - Preprocessing: smoothing, filtering, standardization
  - Statistics: t-tests, regression, prediction
  - Visualization: plotting brain maps and montages
- **Memory considerations**: Large 4D data can consume significant RAM; consider chunking or iterative processing

**Design_Matrix** (`design_matrix.py`)
- **Purpose**: Enhanced pandas DataFrame for experimental designs with temporal metadata
- **Key features**:
  - Tracks sampling frequency (TR)
  - HRF convolution for event-related designs
  - Polynomial detrending and temporal filtering
  - Multicollinearity detection (VIF calculation)
  - Automatic upsampling/downsampling
- **Integration**: Used as input for Brain_Data.regress() method

**Adjacency** (`adjacency.py`)
- **Purpose**: Efficient representation of similarity/connectivity matrices
- **Storage**: Upper triangle vectorization to save memory
- **Types**: similarity, distance, directed graphs
- **Operations**:
  - Statistical testing (t-test, threshold)
  - Network metrics (clustering, shortest paths)
  - Visualization (matrix plots, network graphs)

### Functional Modules

**stats.py**: Statistical functions
- Bootstrap operations (`matrix_permutation`, `summarize_bootstrap`)
- Correlation transforms (`fisher_r_to_z`, `fisher_z_to_r`)
- Matrix operations (`procrustes`, `transform_pairwise`)
- Outlier detection (`find_spikes`)

**utils.py**: Utility functions
- HDF5 I/O (`to_h5`, `load_brain_data_h5`)
- Data validation (`check_brain_data`, `check_square_numpy_matrix`)
- Algorithm selection (`set_algorithm`, `set_decomposition_algorithm`)
- Bootstrapping support (`_bootstrap_apply_func`)
- Data concatenation (`concatenate`)

**plotting.py**: Visualization
- Brain plotting (`plot_brain`, `plot_glass_brain`)
- Statistical maps (`plot_stats`)
- Design matrices (`plot_design_matrix`)
- Uses nilearn plotting functions with nltools defaults

**mask.py**: Masking operations
- Mask creation and manipulation
- ROI extraction
- Mask dilation/erosion (`expand_mask`, `collapse_mask`)

**analysis.py**: Analysis pipelines
- Functional connectivity
- Decomposition methods (PCA, ICA, NMF)
- Pattern analysis

**cross_validation.py**: Machine learning
- Cross-validation schemes for neuroimaging
- Prediction pipelines
- Feature selection

**datasets.py**: Data loading
- Example datasets for testing and tutorials
- Fetchers for various neuroimaging atlases
- Integration with nilearn datasets

## Development Workflow

### Before Starting Work

1. **Check existing research**:
```bash
ls claude-research/
# Review relevant research docs for your task
```

2. **Verify nilearn capabilities**:
```python
# Always check if nilearn already has the functionality
from nilearn import plotting, maskers, image
# Check their latest docs: https://nilearn.github.io/stable/
```

3. **Set up environment**:
```bash
# Use uv for dependency management
uv sync
uv run python  # Ensures correct environment
```

### Code Patterns & Best Practices

#### Input Validation Pattern
```python
# GOOD: Use validation module
from nltools.data._validation import validate_data_type, validate_frame

def method(self, data):
    data = validate_data_type(data)  # Handles various input types

# AVOID: Inline type checking
def method(self, data):
    if isinstance(data, str):
        # Handle string
    elif isinstance(data, list):
        # Handle list
    # etc...
```

#### Memory-Efficient Operations
```python
# GOOD: Process large datasets in chunks
def process_large_data(brain_data, chunk_size=100):
    n_samples = brain_data.shape()[0]
    results = []
    for i in range(0, n_samples, chunk_size):
        chunk = brain_data[i:i+chunk_size]
        results.append(process_chunk(chunk))
    return Brain_Data(data=results)

# AVOID: Loading everything at once
all_subjects = Brain_Data([f"sub_{i}.nii" for i in range(1000)])  # Memory explosion
```

#### Leveraging nilearn
```python
# GOOD: Use nilearn's optimized functions
from nilearn.image import smooth_img, resample_to_img
smoothed = Brain_Data(smooth_img(brain_data.to_nifti(), fwhm=6))

# AVOID: Reimplementing existing functionality
def custom_smooth(data, kernel_size):
    # Don't reinvent the wheel
    pass
```

#### Error Handling
```python
# GOOD: Informative error messages
if data.shape()[0] != labels.shape[0]:
    raise ValueError(
        f"Data has {data.shape()[0]} samples but labels has {labels.shape[0]}. "
        f"Ensure each observation has a corresponding label."
    )

# AVOID: Generic errors
if data.shape()[0] != labels.shape[0]:
    raise ValueError("Shape mismatch")
```

#### Docstring Format (Google style)
```python
def compute_similarity(self, other, metric="correlation"):
    """Compute similarity between brain images.

    Args:
        other (Brain_Data): Brain data to compare with
        metric (str): Similarity metric ('correlation', 'euclidean', 'cosine')
            Default: 'correlation'

    Returns:
        Adjacency: Similarity matrix between all pairs of images

    Raises:
        ValueError: If images have different dimensions

    Examples:
        >>> sim = brain_data1.compute_similarity(brain_data2)
        >>> sim.plot()
    """
```

### Testing Strategy

#### Test-Driven Development Workflow
```python
# 1. Write test first (nltools/tests/test_feature.py)
def test_new_smoothing_method(sim_brain_data):
    # Arrange - use fixtures from conftest.py
    data = sim_brain_data
    expected_fwhm = 6.0

    # Act - call method to test
    smoothed = data.smooth_new(fwhm=expected_fwhm)

    # Assert - specific checks
    assert smoothed.shape() == data.shape()
    assert smoothed.data.std() < data.data.std()  # Smoothing reduces variance

# 2. Run test to see it fail
uv run pytest -k test_new_smoothing_method -v

# 3. Implement minimal code to pass
# 4. Refactor while keeping tests green
```

#### Using Test Fixtures
```python
# Available fixtures in conftest.py:
# - sim_brain_data: Small simulated Brain_Data for testing
# - sim_design_matrix: Sample Design_Matrix with 4 conditions
# - sim_adjacency_single: Single adjacency matrix
# - sim_adjacency_multiple: Multiple adjacency matrices

def test_with_fixture(sim_brain_data, sim_design_matrix):
    # Fixtures are automatically provided
    result = sim_brain_data.regress(sim_design_matrix)
    assert hasattr(result, 'beta')
```

#### Testing Large Data
```python
# For operations that need real neuroimaging data:
def test_large_data_operation():
    # Use the test data file (92MB)
    test_file = "nltools/tests/data.nii.gz"
    brain = Brain_Data(test_file)

    # But subsample for speed
    brain_subset = brain[::10]  # Every 10th volume
    result = expensive_operation(brain_subset)
```

## Common Operations Guide

### Loading Data
```python
from nltools.data import Brain_Data

# From file
brain = Brain_Data("path/to/data.nii.gz")

# From list of files
brain = Brain_Data(["sub1.nii", "sub2.nii", "sub3.nii"])

# From nibabel object
import nibabel as nib
img = nib.load("data.nii.gz")
brain = Brain_Data(img)

# From URL
brain = Brain_Data("https://neurovault.org/media/images/image.nii.gz")

# With custom mask
brain = Brain_Data("data.nii.gz", mask="custom_mask.nii.gz")

# Empty initialization for building incrementally
brain = Brain_Data()
```

### Preprocessing
```python
# Smoothing
brain_smooth = brain.smooth(fwhm=6)

# Standardization
brain_std = brain.standardize(axis=0)  # Standardize each voxel

# Temporal filtering
brain_filt = brain.filter(low_pass=0.1, high_pass=0.01, tr=2.0)

# Apply mask
masked = brain.apply_mask(mask_img)
```

### Statistical Operations
```python
# T-test
t_stats = brain.ttest()  # One-sample t-test
t_stats = brain.ttest(brain2)  # Two-sample t-test

# Regression with Design Matrix
dm = Design_Matrix(X, sampling_freq=0.5)
results = brain.regress(dm)

# Prediction
brain.predict(algorithm='ridge', cv_dict={'type': 'kfolds', 'n_folds': 5})

# Bootstrap
bootstrap_results = brain.bootstrap(function=np.mean, n_samples=1000)
```

### Working with ROIs
```python
# Extract ROI means
roi_data = brain.extract_roi(mask="atlas.nii.gz")

# Create similarity matrix between ROIs
similarity = brain.compute_similarity()

# Threshold
thresholded = brain.threshold(lower=2.3, upper=None)
```

## Troubleshooting Guide

### Memory Issues
```python
# Problem: MemoryError when loading many subjects
# Solution 1: Process iteratively
results = []
for subject_file in subject_files:
    brain = Brain_Data(subject_file)
    result = brain.some_operation()
    results.append(result.mean())  # Store summary, not full data
    del brain  # Explicitly free memory

# Solution 2: Use HDF5 for large datasets
brain.write("large_data.h5", compression=9)
brain = Brain_Data("large_data.h5")  # Memory-mapped loading
```

### Performance Bottlenecks
```python
# Problem: Slow operations on large data
# Solution: Use parallel processing
from joblib import Parallel, delayed

def process_subject(file):
    return Brain_Data(file).ttest()

results = Parallel(n_jobs=4)(
    delayed(process_subject)(f) for f in files
)
```

### NiftiMasker Issues
```python
# Problem: "Image has incompatible shape" errors
# Solution: Ensure consistent image dimensions
from nilearn.image import resample_to_img

# Resample to match template
template = Brain_Data().mask
resampled = resample_to_img(source_img, template)
brain = Brain_Data(resampled)

# Or use same mask for all subjects
common_mask = create_common_mask(subject_files)
brain1 = Brain_Data(file1, mask=common_mask)
brain2 = Brain_Data(file2, mask=common_mask)
```

### Design Matrix Integration Issues
```python
# Problem: Design matrix doesn't align with data
# Solution: Check sampling rates and length
dm = Design_Matrix(design_df, sampling_freq=1/TR)

# Ensure lengths match
if len(dm) != brain.shape()[0]:
    # Option 1: Resample design matrix
    dm = dm.resample(target_length=brain.shape()[0])

    # Option 2: Trim data
    brain = brain[:len(dm)]
```

## Quality Assurance Checklist

Before committing changes:

- [ ] **Linting**: Run `uv run ruff check` and fix all issues
- [ ] **Auto-fix**: Run `uv run ruff check --fix` for automatic fixes
- [ ] **Tests pass**: Run `uv run pytest -k your_test` for specific tests
- [ ] **Full test suite**: Run `uv run pytest` before pushing
- [ ] **Docstrings**: Updated/added in Google format
- [ ] **Type hints**: Added where they improve clarity (optional but helpful)
- [ ] **Memory check**: Tested with realistic data sizes
- [ ] **Documentation**: Update notebooks if API changed
- [ ] **Build docs**: Verify with `uv run jupyter-book build docs/`

## Research Documents Reference

Key research documents in `claude-research/` to consult:

- **apply_mask_analysis.md**: Optimization strategies for masking operations
- **nilearn_features_analysis.md**: Comprehensive review of nilearn capabilities we can leverage
- **research-nilearn-maskers.md**: Alternative masker types (Labels, Maps, Spheres) for ROI analysis
- **braindata-refactor.md**: Ongoing refactoring priorities and patterns
- **warning_analysis.md**: Common warnings and their solutions
- **comprehensive_improvements.md**: Overall improvement roadmap

Before implementing new functionality, check these documents to see if the problem has already been researched.

## Performance Considerations

### Neuroimaging-Specific Optimizations

1. **Vectorization over voxels**: Operations on `.data` attribute are faster than converting back to 3D
2. **Sparse operations**: Many brain images are mostly zeros - consider sparse matrices for connectivity
3. **Chunk processing**: For 4D data, process time-points in chunks to avoid memory issues
4. **Parallel voxel operations**: Use joblib for embarrassingly parallel operations across voxels
5. **HDF5 for persistence**: Use `.write()` method with HDF5 for efficient storage and loading

### Profiling Operations
```python
import time
import tracemalloc

# Time profiling
start = time.time()
result = brain.expensive_operation()
print(f"Operation took {time.time() - start:.2f} seconds")

# Memory profiling
tracemalloc.start()
result = brain.expensive_operation()
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 10**6:.1f} MB")
tracemalloc.stop()
```

## Contributing Guidelines

### Git Workflow
```bash
# Create feature branch from master
git checkout -b feature/your-feature

# Make changes following TDD
# Write test -> Run test -> Implement -> Refactor

# Commit with clear messages
git add -p  # Stage selectively
git commit -m "Add ROI extraction using NiftiLabelsMasker

- Replace custom implementation with nilearn masker
- Add tests for multiple atlas types
- Update docstrings with examples"

# Push and create PR
git push origin feature/your-feature
```

### PR Guidelines
- Reference any related issues
- Describe what changed and why
- Include test results
- Note any breaking changes
- Update documentation if needed

## Current Refactoring Priorities

Based on `refactor.md`, ongoing work focuses on:

1. **Reducing code surface**: Leveraging nilearn instead of custom implementations
2. **Brain_Data simplification**:
   - Removing redundant prediction methods
   - Standardizing attribute names (.design_matrix instead of .X)
   - Storing GLM results as attributes instead of returning dictionaries
3. **Improved masker support**: Enabling alternative maskers (ROI, searchlight, etc.)
4. **Memory efficiency**: Better handling of large datasets

## Quick Command Reference

```bash
# Environment setup
uv sync                           # Install/update dependencies
uv run python                     # Run Python in environment

# Testing
uv run pytest                     # Run all tests
uv run pytest -k test_name        # Run specific test
uv run pytest -v                  # Verbose output
uv run pytest --pdb              # Debug on failure

# Code quality
uv run ruff check                 # Check for issues
uv run ruff check --fix          # Auto-fix issues
uv run ruff format               # Format code (if configured)

# Documentation
uv run jupyter-book build docs/  # Build documentation
uv run jupyter notebook          # Launch notebook for testing

# Research
grep -r "pattern" .              # Search codebase
find . -name "*.py" | xargs grep "function"  # Search Python files
```

## Additional Resources

- **nltools documentation**: https://nltools.org
- **nilearn documentation**: https://nilearn.github.io/stable/
- **Issues tracker**: https://github.com/cosanlab/nltools/issues
- **Example notebooks**: `docs/tutorials/`

---

*Last updated: 2024-10-28*
*This is a living document - update it as patterns evolve*