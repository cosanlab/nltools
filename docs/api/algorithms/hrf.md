---
title: algorithms.hrf
label: algorithms-hrf
---

Hemodynamic response functions — thin wrappers over nilearn.

nilearn ships canonical SPM and Glover HRFs (and their derivatives) under
``nilearn.glm.first_level``. This module wraps them (same
parameters, keyword-only after ``t_r`` per the nltools convention) so
``nltools.algorithms.hrf`` imports keep working and the API reference renders
Markdown docstrings.

Every function returns a 1D array sampled every ``t_r / oversampling``
seconds for ``time_length`` seconds, and is scaled so the canonical HRF
peaks at 1.

**Functions:**

Name | Description
---- | -----------
[`glover_dispersion_derivative`](#algorithms-hrf-glover-dispersion-derivative) | Sample the dispersion derivative of the Glover hemodynamic response function.
[`glover_hrf`](#algorithms-hrf-glover-hrf) | Sample the Glover hemodynamic response function.
[`glover_time_derivative`](#algorithms-hrf-glover-time-derivative) | Sample the time derivative of the Glover hemodynamic response function.
[`spm_dispersion_derivative`](#algorithms-hrf-spm-dispersion-derivative) | Sample the dispersion derivative of the SPM canonical hemodynamic response function.
[`spm_hrf`](#algorithms-hrf-spm-hrf) | Sample the SPM canonical hemodynamic response function.
[`spm_time_derivative`](#algorithms-hrf-spm-time-derivative) | Sample the time derivative of the SPM canonical hemodynamic response function.



## Functions

(algorithms-hrf-glover-dispersion-derivative)=
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

(algorithms-hrf-glover-hrf)=
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

(algorithms-hrf-glover-time-derivative)=
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

(algorithms-hrf-spm-dispersion-derivative)=
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

(algorithms-hrf-spm-hrf)=
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

(algorithms-hrf-spm-time-derivative)=
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
