# dartbrains Course Migration: v0.5 → v0.6

This document catalogs every nltools API change needed in the [dartbrains course](https://dartbrains.org) notebooks. For the full migration guide, see [](migration-guide.md).

---

## Per-Notebook Reference

| Notebook | Changes Needed | Difficulty |
|----------|----------------|------------|
| **Connectivity.ipynb** | Class renames, `onsets_to_dm` import, `.regress()` (3 calls, deprecated) | Medium |
| **Introduction_to_ICA.ipynb** | Class renames, `get_anatomical` removed, `.shape()` → `.shape` (3 calls) | Medium |
| **GLM_Single_Subject_Model.ipynb** | Class renames, `onsets_to_dm` + `regress` imports, `.regress()` (1 call, deprecated) | Medium |
| **Group_Analysis.ipynb** | Class renames, `glover_hrf` + `regress` imports, `.regress()` (1 call), `.ttest()` (2 calls, removed) | **High** |
| **RSA.ipynb** | Class renames, `one_sample_permutation` (deprecated, still works) | Low |
| **Thresholding_Group_Analyses.ipynb** | Class rename, `SimulateGrid` import, `.ttest()` (4 calls, removed) | **High** |
| **Multivariate_Prediction.ipynb** | Class rename, `.predict(algorithm=, cv_dict=)` → new API (8 calls) | **High** |
| **Introduction_to_Neuroimaging_Data.ipynb** | Class rename, `get_anatomical` removed, `.shape()` → `.shape` (5 calls) | Medium |
| **ICA.ipynb** | Class rename only | Low |
| **Parcellations.ipynb** | Class rename only | Low |
| **GLM.ipynb** | `glover_hrf` import change | Low |
| **Signal_Processing.ipynb** | `glover_hrf` import change | Low |
| **Glossary.ipynb** | Text references to old class/function names | Low |

---

## Detailed Changes by Notebook

### Connectivity.ipynb

**Imports** (cell 5):
```python
# OLD
from nltools.data import Brain_Data, Design_Matrix, Adjacency
from nltools.stats import zscore, fdr, one_sample_permutation
from nltools.file_reader import onsets_to_dm
from nltools.plotting import component_viewer

# NEW
from nltools.data import BrainData, DesignMatrix, Adjacency
from nltools.stats import zscore, fdr, one_sample_permutation  # still works (deprecated wrapper)
from nltools.io import onsets_to_dm
from nltools.plotting import component_viewer  # unchanged
```

**`.regress()` calls** (3 occurrences — deprecated wrapper still works, but should migrate):
```python
# OLD (still works in v0.6.0 via deprecation wrapper, will be removed in v0.7.0)
smoothed.X = dm
stats = smoothed.regress()
betas = stats['beta']
residual = stats['residual']

# NEW (recommended)
smoothed.fit(model='glm', X=dm)
betas = smoothed.glm_betas
residual = smoothed.glm_residual
```

**Class renames**: Replace all `Brain_Data` → `BrainData`, `Design_Matrix` → `DesignMatrix` throughout.

---

### Introduction_to_ICA.ipynb

**Imports** (cell 3):
```python
# OLD
from nltools import Brain_Data, Design_Matrix
from nltools.mask import create_sphere
from nltools.utils import get_anatomical

# NEW
from nltools import BrainData, DesignMatrix
from nltools.mask import create_sphere  # unchanged
from nilearn.datasets import load_mni152_template  # replaces get_anatomical
```

**`get_anatomical()` removal** (3 occurrences):
```python
# OLD
simulated_data = Brain_Data(get_anatomical())

# NEW
from nilearn.datasets import load_mni152_template
simulated_data = BrainData(load_mni152_template(resolution=2))
```

**`.shape()` → `.shape`** (3 occurrences):
```python
# OLD
simulated_data.data = np.zeros([n_tr, simulated_data.shape()[0]])

# NEW
simulated_data.data = np.zeros([n_tr, simulated_data.shape[0]])
```

---

### GLM_Single_Subject_Model.ipynb

**Imports** (cell 2):
```python
# OLD
from nltools.file_reader import onsets_to_dm
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.stats import find_spikes

# NEW
from nltools.io import onsets_to_dm
from nltools.stats import zscore, find_spikes  # regress removed from stats
from nltools.data import BrainData, DesignMatrix
```

**`.regress()` call** (1 occurrence — deprecated wrapper):
```python
# OLD
smoothed.X = dm_conv_filt
stats = smoothed.regress()
print(stats.keys())
# dict_keys(['beta', 't', 'p', 'df', 'residual'])

# NEW (recommended)
smoothed.fit(model='glm', X=dm_conv_filt)
# Results stored as attributes:
#   smoothed.glm_betas, smoothed.glm_t, smoothed.glm_p,
#   smoothed.glm_residual, smoothed.glm_r2
```

**Class renames**: Replace `Brain_Data` → `BrainData`, `Design_Matrix` → `DesignMatrix` throughout.

---

### Group_Analysis.ipynb

**Imports** (cell 8):
```python
# OLD
from nltools.stats import regress, zscore
from nltools.data import Brain_Data, Design_Matrix
from nltools.external import glover_hrf

# NEW
from nltools.stats import zscore  # regress removed
from nltools.data import BrainData, DesignMatrix
from nltools.algorithms.hrf import glover_hrf
```

**`.regress()` call** (1 occurrence):
```python
# OLD (inside a loop)
data.X = dm
stats = data.regress()
all_betas.append(stats['beta'][:-1])

# NEW
data.fit(model='glm', X=dm)
all_betas.append(data.glm_betas[:-1])
```

**`.ttest()` calls** (2 occurrences — **removed from BrainData, must fix**):
```python
# OLD
con1_stats = con1_dat.ttest()
con1_stats['thr_t'].plot()

con1_v_con2_stats = con1_v_con2.ttest()
con1_v_con2_stats['thr_t'].plot()

# NEW — use scipy directly
from scipy.stats import ttest_1samp
t_vals, p_vals = ttest_1samp(con1_dat.data, 0, axis=0)
# To create a thresholded BrainData for plotting:
t_brain = con1_dat[0].copy()
t_brain.data = t_vals
t_brain.data[p_vals > 0.05] = 0  # threshold at p < 0.05
t_brain.plot()
```

---

### RSA.ipynb

**Imports** (cell 2):
```python
# OLD
from nltools.data import Brain_Data, Adjacency
from nltools.mask import expand_mask, roi_to_brain
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation

# NEW
from nltools.data import BrainData, Adjacency
from nltools.mask import expand_mask, roi_to_brain  # unchanged
from nltools.stats import fdr, threshold, fisher_r_to_z, one_sample_permutation  # all still work
```

**`one_sample_permutation` usage** (1 occurrence — deprecated wrapper, still works):
```python
# OLD (still works in v0.6.0)
rsa_stats.append(one_sample_permutation(fisher_r_to_z(all_sub_motor_rsa[i])))

# NEW (recommended, for future-proofing)
from nltools.algorithms.inference import one_sample_permutation_test
rsa_stats.append(one_sample_permutation_test(fisher_r_to_z(all_sub_motor_rsa[i])))
```

**Class renames**: Replace `Brain_Data` → `BrainData` throughout.

---

### Thresholding_Group_Analyses.ipynb

**Imports** (cell 4):
```python
# OLD
from nltools.data import Brain_Data
from nltools.simulator import SimulateGrid

# NEW
from nltools.data import BrainData
from nltools import SimulateGrid  # or: from nltools.data import SimulateGrid
```

**`.ttest()` calls** (4 occurrences — **removed from BrainData, must fix**):
```python
# OLD
con1_stats = con1_dat.ttest(threshold_dict={'unc': .001})
con1_stats['thr_t'].plot()

con1_stats = con1_dat.ttest(threshold_dict={'fdr': .05})

# NEW — use scipy + manual thresholding
from scipy.stats import ttest_1samp

t_vals, p_vals = ttest_1samp(con1_dat.data, 0, axis=0)
t_brain = con1_dat[0].copy()
t_brain.data = t_vals

# Uncorrected threshold at p < .001
t_brain_unc = t_brain.copy()
t_brain_unc.data[p_vals > .001] = 0
t_brain_unc.plot()

# FDR-corrected threshold at q < .05
from nltools.stats import fdr
fdr_p = fdr(p_vals, q=0.05)
t_brain_fdr = t_brain.copy()
t_brain_fdr.data[p_vals > fdr_p] = 0
t_brain_fdr.plot()
```

**Class renames**: Replace `Brain_Data` → `BrainData` throughout.

---

### Multivariate_Prediction.ipynb

**Imports**:
```python
# OLD
from nltools.data import Brain_Data
# NEW
from nltools.data import BrainData
```

**`.predict()` API changes** (8 occurrences):
```python
# OLD — algorithm= and cv_dict= keywords
svm_stats = data.predict(algorithm='svm', **{'kernel': "linear"})

svm_stats = data.predict(
    algorithm='svm',
    cv_dict={'type': 'kfolds', 'n_folds': 5, 'subject_id': subject_id},
    **{'kernel': "linear"}
)

ridge_stats = data.predict(algorithm='ridgeClassifier', 
    cv_dict={'type': 'kfolds', 'n_folds': 5, 'subject_id': subject_id})

# NEW — method= and cv= keywords
svm_stats = data.predict(y=data.Y, method='svm', kernel='linear')

from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
svm_stats = data.predict(
    y=data.Y,
    method='svm',
    cv=gkf.split(data.data, groups=subject_id),
    kernel='linear'
)

ridge_stats = data.predict(
    y=data.Y,
    method='ridgeClassifier',
    cv=gkf.split(data.data, groups=subject_id)
)
```

---

### Introduction_to_Neuroimaging_Data.ipynb

**Imports**:
```python
# OLD
from nltools.data import Brain_Data
from nltools.utils import get_anatomical
# NEW
from nltools.data import BrainData
from nilearn.datasets import load_mni152_template
```

**`get_anatomical()` removal** (1 occurrence):
```python
# OLD
anat = Brain_Data(get_anatomical())
# NEW
anat = BrainData(load_mni152_template(resolution=2))
```

**`.shape()` → `.shape`** (5 occurrences):
```python
# OLD
print(data.shape())
print(data[5].shape())
print(data.mean().shape())

# NEW — just remove the parentheses
print(data.shape)
print(data[5].shape)
print(data.mean().shape)
```

---

### ICA.ipynb, Parcellations.ipynb

Class rename only:
```python
# OLD
from nltools.data import Brain_Data
# NEW
from nltools.data import BrainData
```

---

### GLM.ipynb, Signal_Processing.ipynb

`glover_hrf` import change only:
```python
# OLD
from nltools.external import glover_hrf
# NEW
from nltools.algorithms.hrf import glover_hrf
```

---

## Quick-Find Commands

Run these from the `dartbrains/content/` directory to locate all affected lines:

```bash
# Class renames (hard break)
rg 'Brain_Data|Design_Matrix' --glob '*.ipynb'

# Removed imports (hard break)
rg 'nltools\.(file_reader|external|simulator|utils)' --glob '*.ipynb'
rg 'from nltools.stats import.*regress' --glob '*.ipynb'

# Removed methods (hard break)
rg '\.ttest\(' --glob '*.ipynb'

# Changed methods (will silently break)
rg '\.shape\(\)' --glob '*.ipynb'

# Deprecated (still works, should update)
rg '\.regress\(\)' --glob '*.ipynb'
rg 'one_sample_permutation\b' --glob '*.ipynb'

# Predict API changes
rg 'algorithm=|cv_dict=' --glob '*.ipynb'
```

---

## Summary of Changes by Category

### Must fix (ImportError / AttributeError)
- **14 files**: `Brain_Data` → `BrainData`
- **5 files**: `Design_Matrix` → `DesignMatrix`
- **2 files**: `from nltools.file_reader` → `from nltools.io`
- **3 files**: `from nltools.external` → `from nltools.algorithms.hrf`
- **1 file**: `from nltools.simulator` → `from nltools` or `from nltools.data`
- **2 files**: `get_anatomical()` → `nilearn.datasets.load_mni152_template()`
- **2 files**: `from nltools.stats import regress` → removed
- **2 files**: `.ttest()` (6 calls) → `scipy.stats.ttest_1samp()`
- **2 files**: `.shape()` (8 calls) → `.shape`

### Should fix (deprecated, works for now)
- **3 files**: `.regress()` (5 calls) → `.fit(model='glm')`
- **2 files**: `one_sample_permutation` → `one_sample_permutation_test`
- **1 file**: `.predict(algorithm=, cv_dict=)` (8 calls) → `.predict(method=, cv=)`

---

*See [](migration-guide.md) for the complete migration guide with all v0.6.0 changes.*

*Last updated: 2026-04-06*
