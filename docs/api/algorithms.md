# `nltools.algorithms`

**Optimized Algorithms for Neuroimaging**

High-performance implementations of core algorithms with optional GPU acceleration.

---

(ridge-regression)=
## Ridge Regression

Efficient ridge regression implementation using SVD decomposition with support for CPU and GPU backends.

```{eval-rst}
.. automodule:: nltools.algorithms.ridge
    :members:
    :undoc-members:
    :show-inheritance:
```

### Quick Start

```python
from nltools.algorithms.ridge import ridge_svd, ridge_cv
import numpy as np

# Generate sample data
X = np.random.randn(300, 50000)
y = np.random.randn(300)

# Single ridge regression with specified alpha
coef = ridge_svd(X, y, alpha=1.0, backend='auto')

# Cross-validation to select best alpha
result = ridge_cv(X, y, cv=5, backend='auto')
print(f"Best alpha: {result['alpha']}")
print(f"Coefficients shape: {result['coef'].shape}")
print(f"Backend used: {result['backend']}")
```

### Key Features

- **SVD-based solution**: Numerically stable, inspired by himalaya library
- **GPU acceleration**: Optional PyTorch backend for large problems
- **Auto-selection**: Intelligent backend choice based on problem size
- **Multi-target**: Supports both single and multi-target regression
- **Cross-validation**: Built-in k-fold CV for hyperparameter selection

### Performance

For detailed benchmarks and backend selection guidance, see the [Performance Guide](../performance.md).

**Rule of thumb:**
- **Small problems** (< 10M elements): NumPy is sufficient
- **Large problems** (> 30M elements): GPU provides significant speedup
- **Cross-validation**: GPU recommended even for medium problems

---

(hrf)=
## HRF Functions

Hemodynamic Response Functions for fMRI analysis, implemented by NiPy.

```{eval-rst}
.. automodule:: nltools.algorithms.hrf
    :members:
    :undoc-members:
    :show-inheritance:
```

### Quick Start

```python
from nltools.algorithms.hrf import spm_hrf, glover_hrf
import numpy as np

# Generate SPM canonical HRF
tr = 2.0  # Repetition time in seconds
hrf_spm = spm_hrf(tr, oversampling=16)

# Generate Glover HRF
hrf_glover = glover_hrf(tr, oversampling=16)

# Use HRF for convolution with stimulus
import matplotlib.pyplot as plt
plt.plot(hrf_spm, label='SPM HRF')
plt.plot(hrf_glover, label='Glover HRF')
plt.xlabel('Time (TR)')
plt.ylabel('Response')
plt.legend()
plt.show()
```

### Available Functions

- **`spm_hrf`**: SPM canonical HRF
- **`glover_hrf`**: Glover HRF model
- **`spm_time_derivative`**: Temporal derivative of SPM HRF
- **`glover_time_derivative`**: Temporal derivative of Glover HRF
- **`spm_dispersion_derivative`**: Dispersion derivative of SPM HRF

### Use Cases

HRF functions are essential for:
- **Design matrix creation**: Convolving stimulus timing with HRF
- **GLM analysis**: Modeling BOLD response in fMRI
- **Temporal analysis**: Using derivatives for flexible timing models
- **Method comparison**: Testing different HRF assumptions

---

## Shared Response Model

Implementation of Shared Response Model (SRM) for multi-subject fMRI alignment.

```{eval-rst}
.. automodule:: nltools.algorithms.srm
    :members:
    :undoc-members:
    :show-inheritance:
```

### Quick Start

```python
from nltools.algorithms.srm import SRM
import numpy as np

# Multi-subject data: list of arrays (subjects × timepoints × voxels)
data = [np.random.randn(200, 1000) for _ in range(10)]  # 10 subjects

# Fit SRM to learn shared space
srm = SRM(n_iter=10, features=50)
srm.fit(data)

# Transform subjects to shared space
shared_data = srm.transform(data)

# Get transformation matrices
w_transforms = srm.w_  # Per-subject transformations
```

### Key Features

- **Multi-subject alignment**: Learn shared response space across subjects
- **Dimensionality reduction**: Project to lower-dimensional shared features
- **Deterministic variant**: `DetSRM` for deterministic initialization
- **Scalable**: Handles large datasets efficiently

### Applications

- **Inter-subject correlation (ISC)**: Align subjects before computing ISC
- **Decoding across subjects**: Train models on aligned data
- **Hyperalignment**: Alternative to anatomical alignment
- **Neural pattern similarity**: Compare patterns across subjects

### References

**SRM:**
Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J., & Ramadge, P. J. (2015).
A reduced-dimension fMRI shared response model. *Advances in Neural Information Processing Systems*, 460-468.

**DetSRM:**
Anderson, M. J., Capota, M., Turek, J. S., Zhu, X., Willke, T. L., Wang, Y., & Norman, K. A. (2016).
Enabling factor analysis on thousand-subject neuroimaging datasets. *IEEE Big Data*, 1151-1160.

---

## HyperAlignment

Procrustes-based hyperalignment for aligning multi-subject neuroimaging data through iterative template refinement.

```{eval-rst}
.. automodule:: nltools.algorithms.hyperalignment
    :members:
    :undoc-members:
    :show-inheritance:
```

### Quick Start

```python
from nltools.algorithms import HyperAlignment
import numpy as np

# Multi-subject data: list of arrays [features × timepoints]
data = [
    np.random.randn(1000, 200),  # Subject 1
    np.random.randn(1000, 200),  # Subject 2
    np.random.randn(1000, 200),  # Subject 3
]

# Fit hyperalignment model
hyper = HyperAlignment(n_iter=2, auto_pad=True)
hyper.fit(data)

# Transform subjects to common space
aligned_data = hyper.transform(data)

# Access common template and transformations
common_template = hyper.s_  # or hyper.common_model_
transformations = hyper.w_  # Orthogonal transformation matrices
quality_metrics = hyper.disparity_  # Alignment quality per subject

# Align a new subject to the common space
new_subject = np.random.randn(1000, 200)
transformed, R, disparity, scale = hyper.transform_subject(new_subject)
```

### Key Features

- **Three-stage iterative refinement**: Initial template → refined template → final alignment
- **Orthogonal Procrustes**: Minimizes sum of squared differences using optimal rotation/reflection
- **Template refinement**: `n_iter` parameter controls iterative improvement
- **Automatic padding**: Handles different-sized matrices with `auto_pad=True`
- **Sklearn-compatible**: Follows `BaseEstimator`/`TransformerMixin` pattern
- **Quality metrics**: Provides disparity and scale factors for each subject

### Parameters

- **`n_iter`** (int, default=2): Number of template refinement iterations. Higher values improve alignment but increase computation time.
- **`auto_pad`** (bool, default=True): Automatically zero-pad matrices to handle different feature counts across subjects. Set to `False` for validation when all matrices have identical shapes.

### Attributes

After fitting:
- **`w_`**: List of orthogonal transformation matrices (one per subject)
- **`s_`**: Common template in aligned space `[features × timepoints]`
- **`common_model_`**: Alias for `s_` (backward compatibility)
- **`disparity_`**: Alignment quality (sum of squared differences per subject)
- **`scale_`**: Scale factors applied to each subject

### Applications

- **Multi-subject fMRI analysis**: Align functional responses across subjects
- **Inter-subject correlation**: Improve ISC by first aligning representational spaces
- **Transfer learning**: Align a new subject to existing group template
- **Functional connectivity**: Compare connectivity patterns across aligned subjects
- **ROI-based analysis**: Hyperalign ROI data for cross-subject comparisons

### Algorithm Details

HyperAlignment uses a three-stage iterative process:

1. **Stage 1**: Create initial template by iteratively aligning subjects
2. **Stage 2**: Refine template through `n_iter` iterations of alignment and averaging
3. **Stage 3**: Final alignment of all subjects to refined template

Each alignment step solves the orthogonal Procrustes problem using SVD:
- Center and normalize data matrices
- Compute optimal rotation/reflection via `scipy.linalg.orthogonal_procrustes`
- Apply transformation to minimize Frobenius norm of differences

### Comparison with SRM

| Feature | HyperAlignment | SRM |
|---------|---------------|-----|
| Method | Procrustes (orthogonal) | Probabilistic/Deterministic |
| Dimensionality | Preserves features | Reduces to k features |
| Iterations | Template refinement | Feature learning |
| Use Case | Full-resolution alignment | Dimensionality reduction + alignment |

**When to use:**
- **HyperAlignment**: When preserving full feature space (e.g., ROI-level alignment)
- **SRM**: When dimensionality reduction is desired (e.g., whole-brain alignment to shared low-dimensional space)

### References

**Hyperalignment:**
Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O., Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in human ventral temporal cortex. *Neuron*, 72(2), 404-416.

**Procrustes Analysis:**
Gower, J. C., & Dijksterhuis, G. B. (2004). *Procrustes problems* (Vol. 30). Oxford University Press.

### See Also

- **`nltools.stats.align()`** - High-level interface supporting multiple alignment methods
- **`nltools.algorithms.SRM`** - Shared Response Model for dimensionality reduction + alignment
- **`scipy.linalg.orthogonal_procrustes`** - Underlying Procrustes solver

---

## See Also

- {doc}`algorithms/inference` - Statistical inference module (permutation tests, bootstrap)
- {doc}`backends` - CPU/GPU backend abstraction
- {doc}`../performance` - Performance benchmarks and optimization tips
- {doc}`../migration-guide` - Migration guide for v0.6.0 changes
