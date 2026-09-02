---
title: Design matrices, HRF & GLM
label: page-tasks-design-and-glm
---

Build a first-level model. `events_to_dm` turns an events table into a [DesignMatrix](../data/design_matrix.md); the HRF functions sample the SPM and Glover responses and their derivatives for convolution. `regress` is the standalone numpy GLM. For 4D data use `BrainData.fit(model='glm')`, which raises the warning classes listed here when a design is rank-deficient or nearly collinear.

**Classes:**

Name | Description
---- | -----------
[`RankDeficientDesignWarning`](#tasks-design-and-glm-rankdeficientdesignwarning) | The design matrix supplied to ``fit()`` is rank deficient.
[`NearCollinearDesignWarning`](#tasks-design-and-glm-nearcollineardesignwarning) | The design matrix supplied to ``fit()`` is full rank but nearly collinear.
[`DesignMatrixWarning`](#tasks-design-and-glm-designmatrixwarning) | A ``DesignMatrix`` operation was a no-op or partially skipped.
[`ResamplingWarning`](#tasks-design-and-glm-resamplingwarning) | Data is (or will be) resampled to a different space than it arrived in.

**Functions:**

Name | Description
---- | -----------
[`events_to_dm`](#tasks-design-and-glm-events-to-dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
[`spm_hrf`](#tasks-design-and-glm-spm-hrf) | Sample the SPM canonical hemodynamic response function.
[`spm_time_derivative`](#tasks-design-and-glm-spm-time-derivative) | Sample the time derivative of the SPM canonical hemodynamic response function.
[`spm_dispersion_derivative`](#tasks-design-and-glm-spm-dispersion-derivative) | Sample the dispersion derivative of the SPM canonical hemodynamic response function.
[`glover_hrf`](#tasks-design-and-glm-glover-hrf) | Sample the Glover hemodynamic response function.
[`glover_time_derivative`](#tasks-design-and-glm-glover-time-derivative) | Sample the time derivative of the Glover hemodynamic response function.
[`glover_dispersion_derivative`](#tasks-design-and-glm-glover-dispersion-derivative) | Sample the dispersion derivative of the Glover hemodynamic response function.
[`regress`](#tasks-design-and-glm-regress) | Fit an OLS regression of `Y` on `X`.

## Classes

(tasks-design-and-glm-rankdeficientdesignwarning)=
### `RankDeficientDesignWarning`

Bases: `UserWarning`

The design matrix supplied to ``fit()`` is rank deficient.

Subclasses ``UserWarning`` so it participates in default filtering, while
remaining individually silenceable:
``warnings.filterwarnings("ignore", category=RankDeficientDesignWarning)``.

(tasks-design-and-glm-nearcollineardesignwarning)=
### `NearCollinearDesignWarning`

Bases: `UserWarning`

The design matrix supplied to ``fit()`` is full rank but nearly collinear.

Subclasses ``UserWarning`` so it participates in default filtering, while
remaining individually silenceable:
``warnings.filterwarnings("ignore", category=NearCollinearDesignWarning)``.

(tasks-design-and-glm-designmatrixwarning)=
### `DesignMatrixWarning`

Bases: `UserWarning`

A ``DesignMatrix`` operation was a no-op or partially skipped.

Raised by regressor builders (``add_poly``, ``add_dct_basis``,
``convolve``) when the requested columns already exist and are skipped.
Subclasses ``UserWarning`` so it participates in default filtering while
staying individually silenceable:
``warnings.filterwarnings("ignore", category=DesignMatrixWarning)``.

(tasks-design-and-glm-resamplingwarning)=
### `ResamplingWarning`

Bases: `UserWarning`

Data is (or will be) resampled to a different space than it arrived in.

Raised when a data image does not match the mask/template it is loaded
against — a detected template at another resolution, or a mask in a
different space — and nltools resamples to reconcile them. Subclasses
``UserWarning`` so it participates in default filtering while staying
individually silenceable:
``warnings.filterwarnings("ignore", category=ResamplingWarning)``.

## Functions

(tasks-design-and-glm-events-to-dm)=
### `events_to_dm`

```python
events_to_dm(events: pl.DataFrame | pd.DataFrame, *, run_length: int, sampling_freq: float) -> pl.DataFrame
```

Convert a BIDS events table to boxcar regressors aligned to TRs.

Uses `nilearn.glm.first_level.make_first_level_design_matrix` with
`hrf_model=None` to sample events onto the TR grid without HRF
convolution — the caller is expected to call `DesignMatrix.convolve()`
explicitly when convolution is desired. Drops nilearn's auto-added
`constant` column; users add the intercept via `add_poly(0)`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>DataFrame \| DataFrame</code> | Events table with BIDS columns `onset`, `duration`, `trial_type` (required); `modulation` is passed through if present. | *required*
`run_length` | <code>int</code> | Number of TRs the run contains. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz (= 1/TR). | *required*

**Returns:**

Type | Description
---- | -----------
<code>DataFrame</code> | One column per unique `trial_type`, values in     {0, modulation} indicating where each condition is active.

(tasks-design-and-glm-spm-hrf)=
### `spm_hrf`

```python
spm_hrf(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(tasks-design-and-glm-spm-time-derivative)=
### `spm_time_derivative`

```python
spm_time_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The time derivative sampled every `t_r / oversampling` seconds.

(tasks-design-and-glm-spm-dispersion-derivative)=
### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(tasks-design-and-glm-glover-hrf)=
### `glover_hrf`

```python
glover_hrf(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(tasks-design-and-glm-glover-time-derivative)=
### `glover_time_derivative`

```python
glover_time_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The time derivative sampled every `t_r / oversampling` seconds.

(tasks-design-and-glm-glover-dispersion-derivative)=
### `glover_dispersion_derivative`

```python
glover_dispersion_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(tasks-design-and-glm-regress)=
### `regress`

```python
regress(X, Y, *, method: str = 'ols', stats: str = 'full', tail: int | str = 2)
```

Fit an OLS regression of `Y` on `X`.

Does not add an intercept; include one in `X` explicitly. If `Y` is 2D, a
separate regression is fit to each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Design matrix, shape (n_samples, n_regressors). | *required*
`Y` | <code>ndarray</code> | Response, shape (n_samples,) or (n_samples, n_targets). | *required*
`method` | <code>str</code> | Only 'ols' is implemented; for robust or ARMA fits use statsmodels. Defaults to 'ols'. | <code>'ols'</code>
`stats` | <code>str</code> | 'full' returns the 6-tuple below, 'betas' returns just `b`, 'tstats' returns `(b, t)`. Defaults to 'full'. | <code>'full'</code>
`tail` | <code>int \| str</code> | 2 or 'two' for two-tailed p-values (default); 1 or 'one' for a one-tailed test of beta > 0 (negate a regressor for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple</code> | `(b, se, t, p, df, res)` when `stats='full'`: coefficients,     standard errors, t-statistics, p-values (per `tail`), residual     degrees of freedom, and residuals. `stats='betas'` returns just `b`;     `stats='tstats'` returns `(b, t)`.
