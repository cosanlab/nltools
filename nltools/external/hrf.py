'''
HRF Functions
=============

Various Hemodynamic Response Functions (HRFs) implemented by NiPy

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
'''

__all__ = ['spm_hrf',
            'glover_hrf',
            'spm_time_derivative',
            'glover_time_derivative',
            'spm_dispersion_derivative']

from os.path import dirname, join, sep as pathsep
import nibabel as nib
import importlib
import os
from sklearn.pipeline import Pipeline
from scipy.stats import gamma
import numpy as np
import collections
from types import GeneratorType

def _gamma_difference_hrf(tr, oversampling=16, time_length=32., onset=0.,
                        delay=6, undershoot=16., dispersion=1.,
                        u_dispersion=1., ratio=0.167):
    """ Compute an hrf as the difference of two gamma functions
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the hrf
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, float(time_length) / dt)
    time_stamps -= onset / dt
    hrf = gamma.pdf(time_stamps, delay / dispersion, dt / dispersion) - \
        ratio * gamma.pdf(
        time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the SPM hrf model.

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


def glover_hrf(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the Glover hrf model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        hrf: array of shape(length / tr * oversampling, float),
            hrf sampling on the oversampled time grid

    """

    return _gamma_difference_hrf(tr, oversampling, time_length, onset,
                                delay=6, undershoot=12., dispersion=.9,
                                u_dispersion=.9, ratio=.35)


def spm_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
    """ Implementation of the SPM time derivative hrf (dhrf) model.

    Args:
        tr: float, scan repeat time, in seconds
        oversampling: int, temporal oversampling factor, optional
        time_length: float, hrf kernel length, in seconds
        onset: float, onset of the response

    Returns:
        dhrf: array of shape(length / tr, float),
              dhrf sampling on the provided grid

    """

    do = .1
    dhrf = 1. / do * (spm_hrf(tr, oversampling, time_length, onset + do) -
                      spm_hrf(tr, oversampling, time_length, onset))
    return dhrf

def glover_time_derivative(tr, oversampling=16, time_length=32., onset=0.):
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

    do = .1
    dhrf = 1. / do * (glover_hrf(tr, oversampling, time_length, onset + do) -
                      glover_hrf(tr, oversampling, time_length, onset))
    return dhrf

def spm_dispersion_derivative(tr, oversampling=16, time_length=32., onset=0.):
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

    dd = .01
    dhrf = 1. / dd * (_gamma_difference_hrf(tr, oversampling, time_length,
                                           onset, dispersion=1. + dd) -
                      spm_hrf(tr, oversampling, time_length, onset))
    return dhrf
