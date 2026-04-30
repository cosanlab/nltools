# nilearn.signal — Time Series Preprocessing

Preprocessing functions for time series. All functions take `X` matrices shaped `(n_samples, n_features)` (timepoints x voxels/regions). `signal.clean` is the function used internally by every masker for confound removal, filtering, and standardization.

**Source:** https://nilearn.github.io/dev/modules/signal.html

## Inventory

### Functions
| Function | Purpose |
|---|---|
| `butterworth(signals, sampling_rate, ...)` | Apply low-pass, high-pass or band-pass Butterworth filter. |
| `clean(signals, ...)` | Improve SNR on masked fMRI signals (detrend, filter, regress confounds, standardize). |
| `high_variance_confounds(series, ...)` | Return confounds time series extracted from series with highest variance. |

## clean

```python
from nilearn.signal import clean

cleaned = clean(
    signals,                        # (n_timepoints, n_features)
    runs=None,                      # per-timepoint run labels for per-run cleaning
    detrend=True,
    standardize='zscore_sample',    # 'zscore_sample'|'zscore'|'psc'|True|False
    confounds=None,                 # array, DataFrame, file path, or list thereof
    standardize_confounds=True,
    filter='butterworth',           # 'butterworth'|'cosine'|False
    low_pass=None,                  # Hz
    high_pass=None,                 # Hz
    t_r=None,                       # required for filtering
    ensure_finite=False,
    sample_mask=None,               # boolean/index array for scrubbing
    extrapolate=True,               # interpolate censored volumes before filtering
)
```

Processing order: **detrend -> filter -> confound removal -> standardize**. Filters are also applied to `confounds` for consistency.

Standardize options:
- `'zscore_sample'` — z-score using sample std (`ddof=1`); recommended.
- `'zscore'` — z-score using population std (`ddof=0`); same as `True`.
- `'psc'` — percent signal change relative to temporal mean.
- `True` / `False` — `'zscore'` / no standardization.

Filter options:
- `'butterworth'` — IIR Butterworth, requires `t_r`, plus `low_pass` and/or `high_pass`.
- `'cosine'` — discrete cosine transform basis, high-pass only via `high_pass`.
- `False` — disable filtering.

## butterworth

```python
filtered = butterworth(signals, sampling_rate, low_pass=None, high_pass=None,
                        order=5, padtype='odd', padlen=None, copy=True)
```

`sampling_rate` is in Hz (i.e., `1 / t_r`). At least one of `low_pass`/`high_pass` must be set.

## high_variance_confounds

```python
confounds = high_variance_confounds(series, n_confounds=5, percentile=2.0,
                                     detrend=True)
```

CompCor-style: returns the top principal components from the highest-variance voxels. Useful as physiological-noise regressors when fMRIPrep aCompCor outputs aren't available.

## Common patterns

Clean time series with motion confounds and bandpass filter:

```python
from nilearn.signal import clean

cleaned = clean(
    signals, t_r=2.0,
    detrend=True, standardize='zscore_sample',
    confounds=confounds_df,
    filter='butterworth', low_pass=0.1, high_pass=0.01,
)
```

Scrubbing via `sample_mask`:

```python
# sample_mask is the indices of volumes to KEEP (from load_confounds)
cleaned = clean(signals, t_r=2.0, sample_mask=sample_mask,
                confounds=confounds_df, standardize='zscore_sample')
```

Per-run cleaning (concat across runs, then clean each run independently):

```python
runs = np.concatenate([np.full(n1, 0), np.full(n2, 1)])
cleaned = clean(np.vstack([ts1, ts2]), runs=runs, t_r=2.0, detrend=True)
```

CompCor-like confounds:

```python
conf = high_variance_confounds(fmri_series, n_confounds=5, percentile=2.0)
```

## Gotchas

- The processing order matters: confounds are regressed **after** filtering, so high-frequency variance in the confounds is removed before regression. Pass already-filtered confounds at your own risk.
- `t_r` is required for any filter; `signal.clean` will raise without it.
- `standardize=True` is `'zscore'` (ddof=0). For most analyses prefer `'zscore_sample'` (ddof=1) — matches scikit-learn convention.
- `sample_mask` removes scrubbed volumes **after** filtering by default (with `extrapolate=True` it interpolates them first to keep the filter well-defined). Setting `extrapolate=False` censors them up front, which can produce filter artifacts.
- `signal.clean` operates on 2D arrays. For 4D NIfTI use `nilearn.image.clean_img`, or pass cleaning args to a masker.

## See also

- `nilearn.image.clean_img` — same logic for 4D NIfTI inputs.
- `nilearn.maskers.NiftiMasker` — applies `signal.clean` after masking.
- `nilearn.interfaces.fmriprep.load_confounds` — produces `(confounds, sample_mask)` in the right format.
- https://nilearn.github.io/dev/modules/signal.html
