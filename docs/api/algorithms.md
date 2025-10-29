# `nltools.algorithms`

**Optimized Algorithms for Neuroimaging**

High-performance implementations of core algorithms with optional GPU acceleration.

---

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

## See Also

- [Backend API](backends.md) - CPU/GPU backend abstraction
- [Performance Guide](../performance.md) - Benchmarks and optimization tips
- [Stats API](stats.md) - Statistical functions
