## `inference`

BrainCollection inference functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Methods:**

Name | Description
---- | -----------
[`anova`](#anova) | One-way ANOVA across groups defined by metadata.
[`extract_for_isc`](#extract_for_isc) | Extract data for ISC computation.
[`extract_roi`](#extract_roi) | Extract mean signal per ROI.
[`extract_searchlight`](#extract_searchlight) | Extract mean signal per searchlight sphere.
[`extract_voxelwise`](#extract_voxelwise) | Extract raw voxel data.
[`isc`](#isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#isc_test) | Compute ISC with permutation testing for statistical inference.
[`permutation_test`](#permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#permutation_test2) | Two-sample permutation test between collections.
[`project_to_brain`](#project_to_brain) | Project ISC values back to brain space.
[`ttest`](#ttest) | One-sample t-test across images.
[`ttest2`](#ttest2) | Two-sample t-test between collections.



### Classes

### Methods

#### `anova`

```python
anova(bc: BrainCollection, groups: str | list | np.ndarray) -> tuple
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to test. | *required*
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

#### `extract_for_isc`

```python
extract_for_isc(bc: BrainCollection, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, progress_bar: bool = False) -> tuple[np.ndarray, dict]
```

Extract data for ISC computation.

Memory-efficient extraction that processes one subject at a time.
Returns data in ISC-compatible format: (n_obs, n_subjects, n_features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to extract from. | *required*
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | If provided, extract mean per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius_mm` | <code>[float](#float) \| None</code> | searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [dict](#dict)]</code> | Tuple of: - extracted_data: Array of shape (n_obs, n_subjects, n_features) - extraction_info: Dict with metadata for projection back:     - 'mode': 'roi', 'searchlight', or 'voxelwise'     - 'n_features': Number of features     - 'roi_mask': ROI mask if mode='roi'     - 'neighborhoods': SphereNeighborhoods if mode='searchlight'

#### `extract_roi`

```python
extract_roi(bc: BrainCollection, roi_mask: nib.Nifti1Image | Path | str, progress_bar: bool = False) -> tuple[np.ndarray, dict]
```

Extract mean signal per ROI.

#### `extract_searchlight`

```python
extract_searchlight(bc: BrainCollection, radius_mm: float, progress_bar: bool = False) -> tuple[np.ndarray, dict]
```

Extract mean signal per searchlight sphere.

#### `extract_voxelwise`

```python
extract_voxelwise(bc: BrainCollection, progress_bar: bool = False) -> tuple[np.ndarray, dict]
```

Extract raw voxel data.

#### `isc`

```python
isc(bc: BrainCollection, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to compute ISC on. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius_mm` | <code>[float](#float) \| None</code> | searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`device` | <code>[str](#str)</code> | Compute device: 'cpu' (default) or 'gpu' (via PyTorch). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU jobs (-1 = all cores, 1 = single-threaded). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius_mm=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius_mm=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

#### `isc_test`

```python
isc_test(bc: BrainCollection, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to test. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius_mm` | <code>[float](#float) \| None</code> | searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`device` | <code>[str](#str)</code> | Compute device: 'cpu' (default) or 'gpu' (via PyTorch). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU jobs (-1 = all cores, 1 = single-threaded). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius_mm=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius_mm=None,
...     permutation_method='phase_randomize',
...     device='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

#### `permutation_test`

```python
permutation_test(bc: BrainCollection, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', max_gpu_memory_gb: float = 4.0, return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to test. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`device` | <code>[str](#str)</code> | Compute device: 'cpu' (default) or 'gpu' (via PyTorch). | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU jobs (-1 = all cores, 1 = single-threaded). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'device': compute device used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(device='gpu')
```

#### `permutation_test2`

```python
permutation_test2(bc: BrainCollection, other: BrainCollection, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', max_gpu_memory_gb: float = 4.0, return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | First BrainCollection. | *required*
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`device` | <code>[str](#str)</code> | Compute device: 'cpu' (default) or 'gpu' (via PyTorch). | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU jobs (-1 = all cores, 1 = single-threaded). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'device': compute device used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

#### `project_to_brain`

```python
project_to_brain(bc: BrainCollection, values: np.ndarray, extraction_info: dict)
```

Project ISC values back to brain space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection (used for mask). | *required*
`values` | <code>[ndarray](#numpy.ndarray)</code> | ISC values, shape depends on extraction mode: - ROI mode: (n_rois,) - Searchlight/voxelwise: (n_voxels,) | *required*
`extraction_info` | <code>[dict](#dict)</code> | Dict from extract_for_isc with mode info. | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData with ISC values in brain space.

#### `ttest`

```python
ttest(bc: BrainCollection, popmean: float = 0.0, axis: int | str = 0) -> tuple
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to test. | *required*
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>[tuple](#tuple)</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

#### `ttest2`

```python
ttest2(bc: BrainCollection, other: BrainCollection, equal_var: bool = True) -> tuple
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | First BrainCollection. | *required*
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

