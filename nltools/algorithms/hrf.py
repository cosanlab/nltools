"""Hemodynamic response functions — thin wrappers over nilearn.

The canonical SPM and Glover HRFs and their time and dispersion derivatives,
as shipped in `nilearn.glm.first_level`, with the same parameters (keyword-only
after `t_r`).

Every function returns a 1D array sampled every `t_r / oversampling` seconds
for `time_length` seconds; the canonical HRFs are scaled to peak at 1.
"""

from nilearn.glm import first_level as _nilearn

__all__ = [
    "glover_dispersion_derivative",
    "glover_hrf",
    "glover_time_derivative",
    "spm_dispersion_derivative",
    "spm_hrf",
    "spm_time_derivative",
]


def glover_hrf(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the Glover hemodynamic response function.

    Thin wrapper over nilearn's `glover_hrf`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.
    """
    return _nilearn.glover_hrf(t_r, oversampling, time_length, onset)


def glover_time_derivative(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the time derivative of the Glover hemodynamic response function.

    Thin wrapper over nilearn's `glover_time_derivative`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The time derivative sampled every `t_r / oversampling` seconds.
    """
    return _nilearn.glover_time_derivative(t_r, oversampling, time_length, onset)


def glover_dispersion_derivative(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the dispersion derivative of the Glover hemodynamic response function.

    Thin wrapper over nilearn's `glover_dispersion_derivative`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The dispersion derivative sampled every `t_r / oversampling` seconds.
    """
    return _nilearn.glover_dispersion_derivative(t_r, oversampling, time_length, onset)


def spm_hrf(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the SPM canonical hemodynamic response function.

    Thin wrapper over nilearn's `spm_hrf`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.
    """
    return _nilearn.spm_hrf(t_r, oversampling, time_length, onset)


def spm_time_derivative(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the time derivative of the SPM canonical hemodynamic response function.

    Thin wrapper over nilearn's `spm_time_derivative`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The time derivative sampled every `t_r / oversampling` seconds.
    """
    return _nilearn.spm_time_derivative(t_r, oversampling, time_length, onset)


def spm_dispersion_derivative(t_r, *, oversampling=50, time_length=32.0, onset=0.0):
    """Sample the dispersion derivative of the SPM canonical hemodynamic response function.

    Thin wrapper over nilearn's `spm_dispersion_derivative`.

    Args:
        t_r (float): Repetition time in seconds.
        oversampling (int): Temporal oversampling factor (default: 50).
        time_length (float): HRF kernel length in seconds (default: 32.0).
        onset (float): Onset of the response in seconds (default: 0.0).

    Returns:
        ndarray: The dispersion derivative sampled every `t_r / oversampling` seconds.
    """
    return _nilearn.spm_dispersion_derivative(t_r, oversampling, time_length, onset)
