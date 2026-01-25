"""
Hemodynamic Response Functions (HRFs) for fMRI analysis, implemented by NiPy.

This module provides standard HRF implementations for fMRI analysis:
- SPM HRF: The canonical HRF used in SPM12
- Glover HRF: The HRF model from Glover (1999)
- Time derivatives: First-order temporal derivatives
- Dispersion derivatives: HRF shape variations

Performance:
    - Computational cost: Negligible (simple mathematical operations)
    - Memory usage: Small (typically <10KB per HRF)
    - No parallelization needed: Functions are fast enough for typical use
    - Typical use: Convolve HRF with design matrix for GLM analysis

Usage examples:
    Basic HRF generation:

    >>> from nltools.algorithms.hrf import spm_hrf, glover_hrf
    >>> import numpy as np
    >>>
    >>> # Generate SPM HRF (default: TR=2.0 seconds)
    >>> hrf_spm = spm_hrf(tr=2.0)
    >>> len(hrf_spm)  # Length depends on time_length (default: 32 seconds)
    256
    >>>
    >>> # Generate Glover HRF
    >>> hrf_glover = glover_hrf(tr=2.0)
    >>>
    >>> # Generate HRF with custom parameters
    >>> hrf_custom = spm_hrf(tr=1.0, time_length=40.0, oversampling=32)
    >>>
    >>> # Generate time derivative for SPM
    >>> dhrf = spm_time_derivative(tr=2.0)
    >>>
    >>> # Generate dispersion derivative
    >>> dhrf_disp = spm_dispersion_derivative(tr=2.0)

    Integration with design matrices:

    >>> from nltools.algorithms.hrf import spm_hrf
    >>> from nltools.data import DesignMatrix
    >>> import numpy as np
    >>>
    >>> # Create event timing (e.g., stimulus onsets)
    >>> events = np.array([10, 30, 50, 70])  # Time points in seconds
    >>> tr = 2.0  # TR in seconds
    >>> n_trs = 100  # Number of TRs
    >>>
    >>> # Generate HRF
    >>> hrf = spm_hrf(tr=tr)
    >>>
    >>> # Create design matrix with HRF convolution
    >>> # (This is typically done via DesignMatrix class)
    >>> # dm = DesignMatrix(...)  # See DesignMatrix documentation

Copyright (c) 2006-2017, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NIPY Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = [
    "spm_hrf",
    "glover_hrf",
    "spm_time_derivative",
    "glover_time_derivative",
    "spm_dispersion_derivative",
]

from scipy.stats import gamma
import numpy as np


def _gamma_difference_hrf(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
    delay: float = 6.0,
    undershoot: float = 16.0,
    dispersion: float = 1.0,
    u_dispersion: float = 1.0,
    ratio: float = 0.167,
) -> np.ndarray:
    """Compute an hrf as the difference of two gamma functions

    Args:
        tr (float): scan repeat time, in seconds
        oversampling (int, optional): temporal oversampling factor
        time_length (int): hrf kernel length, in seconds
        onset (float): onset of the hrf
        delay (float): delay parameter for gamma function
        undershoot (float): undershoot parameter for gamma function
        dispersion (float): dispersion parameter for gamma function
        u_dispersion (float): undershoot dispersion parameter
        ratio (float): ratio between the two gamma functions

    Returns:
        numpy.ndarray: hrf sampling on the oversampled time grid, shape (length / tr * oversampling,)

    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, int(time_length / dt))
    time_stamps -= onset / dt
    hrf = gamma.pdf(
        time_stamps, delay / dispersion, dt / dispersion
    ) - ratio * gamma.pdf(time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= hrf.sum()
    return hrf


def spm_hrf(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Implementation of the SPM hrf model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        hrf: array of shape(length / tr * oversampling, float),
            hrf sampling on the oversampled time grid

    """

    return _gamma_difference_hrf(tr, oversampling, time_length, onset)


def glover_hrf(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Implementation of the Glover hrf model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: int, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        hrf: array of shape(length / tr * oversampling, float),
            hrf sampling on the oversampled time grid

    """

    return _gamma_difference_hrf(
        tr,
        oversampling,
        time_length,
        onset,
        delay=6,
        undershoot=12.0,
        dispersion=0.9,
        u_dispersion=0.9,
        ratio=0.35,
    )


def spm_time_derivative(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Implementation of the SPM time derivative hrf (dhrf) model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        dhrf: array of shape(length / tr, float),
              dhrf sampling on the provided grid

    """

    do = 0.1
    dhrf = (
        1.0
        / do
        * (
            spm_hrf(tr, oversampling, time_length, onset + do)
            - spm_hrf(tr, oversampling, time_length, onset)
        )
    )
    return dhrf


def glover_time_derivative(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Implementation of the flover time derivative hrf (dhrf) model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        dhrf: array of shape(length / tr, float),
              dhrf sampling on the provided grid

    """

    do = 0.1
    dhrf = (
        1.0
        / do
        * (
            glover_hrf(tr, oversampling, time_length, onset + do)
            - glover_hrf(tr, oversampling, time_length, onset)
        )
    )
    return dhrf


def spm_dispersion_derivative(
    tr: float,
    oversampling: int = 16,
    time_length: float = 32.0,
    onset: float = 0.0,
) -> np.ndarray:
    """Implementation of the SPM dispersion derivative hrf model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        dhrf: array of shape(length / tr * oversampling, float),
              dhrf sampling on the oversampled time grid

    """

    dd = 0.01
    dhrf = (
        1.0
        / dd
        * (
            _gamma_difference_hrf(
                tr, oversampling, time_length, onset, dispersion=1.0 + dd
            )
            - spm_hrf(tr, oversampling, time_length, onset)
        )
    )
    return dhrf
