# nilearn.glm — Generalized Linear Models

Analyzing fMRI data using GLMs. Provides first-level (subject) and second-level (group) modelling, contrast estimation, multiple-comparison correction, HRF utilities, and BIDS-aware helpers.

**Source:** https://nilearn.github.io/dev/modules/glm.html

## Submodules
- `nilearn.glm.first_level` — first-level (subject-level) GLM
- `nilearn.glm.second_level` — second-level (group) GLM

## Top-level inventory

### Classes
| Class | Purpose |
|---|---|
| `Contrast(effect, variance, dim, dof, ...)` | Handles the estimation of statistical contrasts (t or F) on a fitted model |
| `FContrastResults(effect, covariance, F, df_num)` | Results from an F contrast of coefficients in a parametric model |
| `TContrastResults(t, sd, effect, df_den)` | Results from a t contrast of coefficients in a parametric model |
| `ARModel(design, rho)` | Regression model with an AR(p) covariance structure |
| `OLSModel(design)` | Simple ordinary least squares model |
| `LikelihoodModelResults(theta, Y, model, ...)` | Container for results from likelihood models |
| `RegressionResults(theta, Y, model, ...)` | Summarises the fit of a linear regression model |
| `SimpleRegressionResults(results)` | Minimal information from a model fit, sufficient for contrast computation |

### Functions
| Function | Purpose |
|---|---|
| `compute_contrast(labels, regression_result, ...)` | Compute the specified contrast given an estimated GLM |
| `compute_fixed_effects(contrast_imgs, ...)` | Combine effect/variance images across runs (fixed-effects meta) |
| `expression_to_contrast_vector(expression, columns)` | Convert a string contrast expression to a vector against design-matrix columns |
| `fdr_threshold(z_vals, alpha)` | Benjamini–Hochberg FDR threshold for input z-values |
| `cluster_level_inference(stat_img, ...)` | Proportion of active voxels for clusters defined by a threshold |
| `threshold_stats_img(stat_img, mask_img, ...)` | Compute the required height threshold and return the thresholded map |
| `save_glm_to_bids(model, contrasts, ...)` | Save GLM results to BIDS-like derivatives |

## nilearn.glm.first_level

### Classes
| Class | Purpose |
|---|---|
| `FirstLevelModel(t_r, slice_time_ref, ...)` | General Linear Model for single-run / single-subject fMRI data |

### Functions
| Function | Purpose |
|---|---|
| `check_design_matrix(design_matrix)` | Validate a design-matrix DataFrame; returns `(frame_times, matrix, names)` triplet |
| `compute_regressor(exp_condition, hrf_model, frame_times, ...)` | Convolve regressors with an HRF model |
| `first_level_from_bids(dataset_path, task_label, ...)` | Build `FirstLevelModel` objects + fit args from a BIDS dataset |
| `glover_hrf(t_r, oversampling, time_length, onset)` | Glover canonical HRF |
| `glover_time_derivative(t_r, ...)` | Glover time-derivative HRF (dHRF) |
| `glover_dispersion_derivative(t_r, ...)` | Glover dispersion-derivative HRF |
| `spm_hrf(t_r, oversampling, time_length, onset)` | SPM canonical HRF |
| `spm_time_derivative(t_r, ...)` | SPM time-derivative HRF |
| `spm_dispersion_derivative(t_r, ...)` | SPM dispersion-derivative HRF |
| `make_first_level_design_matrix(frame_times, ...)` | Generate a design matrix from frame times, events, drifts, regressors |
| `mean_scaling(Y, axis)` | Rescale data to percent-of-baseline change along an axis |
| `run_glm(Y, X, noise_model, bins, n_jobs, ...)` | Low-level GLM fit for an fMRI data matrix |

### FirstLevelModel
```python
from nilearn.glm.first_level import FirstLevelModel

flm = FirstLevelModel(
    t_r=None,                   # repetition time (seconds)
    slice_time_ref=0.0,         # 0-1 fraction of t_r
    hrf_model='glover',         # see HRF table below
    drift_model='cosine',       # 'cosine'|'polynomial'|None
    high_pass=0.01,             # Hz
    noise_model='ar1',          # 'ar1'|'ar2'|...|'arN'|'ols'
    smoothing_fwhm=None,        # mm
    mask_img=None,              # Niimg-like|masker|False|None
    signal_scaling=0,           # 0=mean scaling, False=none
    minimize_memory=True,
    n_jobs=1,
)

flm.fit(run_imgs, events=events_df, confounds=confounds, sample_masks=sample_mask)
z_map = flm.compute_contrast('face - house', output_type='z_score')
```

Post-fit attributes:
- `flm.design_matrices_` — list of DataFrames, one per run
- `flm.predicted` — list of `Nifti1Image` (only if `minimize_memory=False`)
- `flm.residuals` — list of `Nifti1Image`
- `flm.r_square` — list of `Nifti1Image`

### HRF model strings
| String | Regressors | Notes |
|---|---|---|
| `'glover'` | 1 | Canonical Glover |
| `'glover + derivative'` | 2 | + time derivative |
| `'glover + derivative + dispersion'` | 3 | + dispersion derivative |
| `'spm'` | 1 | SPM canonical |
| `'spm + derivative'` | 2 | + time derivative |
| `'spm + derivative + dispersion'` | 3 | + dispersion derivative |
| `'fir'` | `len(fir_delays)` | Finite impulse response |
| callable / list of callables | 1 per fn | Custom HRF basis |
| `None` | 1 | No convolution (raw regressor) |

### Events DataFrame requirements
Required columns: `onset` (sec), `duration` (sec), `trial_type` (str). Optional: `modulation` (float, parametric modulation).

### Contrast specification
```python
import numpy as np
from nilearn.glm import expression_to_contrast_vector

# String expression (resolves against design-matrix column names)
z = flm.compute_contrast('face - house')
z = flm.compute_contrast('2*face - house - scrambled')

# Numeric vector (1D = t-test)
z = flm.compute_contrast(np.array([1, -1, 0, 0]))

# F-test (2D matrix)
z = flm.compute_contrast(np.array([[1, -1, 0], [0, 1, -1]]), stat_type='F')

# Parse expression to vector against a specific design
vec = expression_to_contrast_vector('face - house', flm.design_matrices_[0].columns)
```

`output_type` of `compute_contrast`: `'z_score'|'stat'|'p_value'|'effect_size'|'effect_variance'|'all'` (`'all'` returns a dict of all five).

### Building design matrices manually
```python
from nilearn.glm.first_level import make_first_level_design_matrix

dm = make_first_level_design_matrix(
    frame_times,                # array of scan times in seconds
    events=events_df,
    hrf_model='glover',
    drift_model='cosine',
    high_pass=0.01,
    add_regs=confounds,         # ndarray (n_scans, n_regs)
    add_reg_names=confound_names,
)
```

### From BIDS
```python
from nilearn.glm.first_level import first_level_from_bids

models, imgs, events, confounds = first_level_from_bids(
    dataset_path,
    task_label='task',
    space_label='MNI152NLin2009cAsym',
)
```

### Low-level fit
```python
from nilearn.glm.first_level import run_glm
labels, results = run_glm(Y, X, noise_model='ar1', bins=100, n_jobs=1)
```

## nilearn.glm.second_level

### Classes
| Class | Purpose |
|---|---|
| `SecondLevelModel(mask_img, target_affine, ...)` | General Linear Model for multi-subject (group) fMRI data |

### Functions
| Function | Purpose |
|---|---|
| `make_second_level_design_matrix(subjects_label, ...)` | Build a second-level design matrix (intercept + covariates) |
| `non_parametric_inference(second_level_input, ...)` | Permutation-based group inference (optionally with TFCE) |

### SecondLevelModel
```python
from nilearn.glm.second_level import SecondLevelModel

slm = SecondLevelModel(mask_img=None, smoothing_fwhm=None, n_jobs=1)

# Input can be: list of FirstLevelModel, list of images, or 4D image
slm.fit(z_maps_list, design_matrix=group_dm)

group_z = slm.compute_contrast('group_mean', output_type='z_score')
```

### Non-parametric inference
```python
from nilearn.glm.second_level import non_parametric_inference

out = non_parametric_inference(
    z_maps,
    design_matrix=dm,
    n_perm=10000,
    two_sided_test=False,
    threshold=3.0,              # cluster-forming threshold
    tfce=False,                 # threshold-free cluster enhancement
)
```

## Thresholding & inference

```python
from nilearn.glm import threshold_stats_img, fdr_threshold, cluster_level_inference

# Multiple-comparison correction (returns a TUPLE)
thresholded_img, threshold_val = threshold_stats_img(
    stat_img,
    alpha=0.05,
    height_control='fdr',       # 'fpr'|'fdr'|'bonferroni'|None
    cluster_threshold=10,
)

# FDR threshold from raw z-values
thr = fdr_threshold(z_vals, alpha=0.05)

# Cluster-level inference (proportion of true positives per cluster)
cluster_img = cluster_level_inference(stat_img, threshold=3.0, alpha=0.05)
```

## Combining across runs (fixed effects)
```python
from nilearn.glm import compute_fixed_effects

fx, var, t = compute_fixed_effects(
    contrast_imgs,              # list of effect images (per run)
    variance_imgs,              # matching variance images
    mask_img=mask,
)
```

## Reporting

Use the model's own `generate_report` (replaces deprecated `make_glm_report`):

```python
report = flm.generate_report(contrasts=['face - house'])
report.save_as_html('report.html')
```

For cluster tables, see `references/reporting.md` (`get_clusters_table`).

## Saving to BIDS
```python
from nilearn.glm import save_glm_to_bids

save_glm_to_bids(
    flm,
    contrasts=['face - house'],
    out_dir='derivatives/nilearn-glm',
)
```

## Common patterns

### fMRIPrep -> first-level GLM
```python
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img

confounds, sample_mask = load_confounds(
    fmri_img,
    strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
    motion='full', scrub=5,
)

flm = FirstLevelModel(
    t_r=2.0, hrf_model='glover', noise_model='ar1',
    smoothing_fwhm=6, high_pass=0.01, mask_img=mask,
)
flm.fit(fmri_img, events=events_df,
        confounds=confounds, sample_masks=sample_mask)

z_map = flm.compute_contrast('active - rest', output_type='z_score')
thresholded, thresh = threshold_stats_img(
    z_map, alpha=0.05, height_control='fdr', cluster_threshold=10,
)
```

## Gotchas
- `events` column names are strict: `onset`, `duration`, `trial_type` (case-sensitive).
- `threshold_stats_img` returns a tuple `(img, threshold_value)` — always unpack.
- `output_type` of `compute_contrast`: `'z_score'|'stat'|'p_value'|'effect_size'|'effect_variance'|'all'`.
- `make_glm_report` is deprecated — use `model.generate_report()`.
- `load_confounds` returns `(confounds_df, sample_mask)` — pass `sample_mask` to `flm.fit(sample_masks=...)` (note plural).
- `signal_scaling=0` (the default) means scaling along axis 0 (time); use `False` to disable scaling entirely.
- `minimize_memory=True` (default) suppresses `predicted`, `residuals`, `r_square` — set `False` to access them.
- `compute_contrast` accepts a 1D array (t-test) or 2D matrix (F-test); pass `stat_type='F'` for F-tests.
- `first_level_from_bids` returns four lists; entries are aligned by subject — iterate together.
- `SecondLevelModel.fit` accepts a list of `FirstLevelModel`, a list of images, or a 4D image; ensure `design_matrix` rows match.

## See also
- `references/maskers.md` — `NiftiMasker`, `NiftiLabelsMasker`
- `references/reporting.md` — `get_clusters_table`, HTML reports
- `references/interfaces.md` — `load_confounds`, BIDS helpers
- `references/image.md` — `threshold_img`, `math_img`
- https://nilearn.github.io/dev/modules/glm.html
