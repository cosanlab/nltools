'''
NeuroLearn Utilities
====================

handy utilities.

'''
__all__ = ['get_resource_path',
            'get_anatomical',
            'set_algorithm',
            'spm_hrf',
            'glover_hrf',
            'spm_time_derivative',
            'glover_time_derivative',
            'spm_dispersion_derivative',
            'make_cosine_basis']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, sep as pathsep
import nibabel as nib
import importlib
import os
from sklearn.pipeline import Pipeline
from scipy.stats import gamma
import numpy as np

def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep

def get_anatomical():
    """ Get nltools default anatomical image. """
    return nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'))

def set_algorithm(algorithm, **kwargs):
    """ Setup the algorithm to use in subsequent prediction analyses.

    Args:
        algorithm: The prediction algorithm to use. Either a string or an
                    (uninitialized) scikit-learn prediction object. If string,
                    must be one of 'svm','svr', linear','logistic','lasso',
                    'lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
                    'randomforest', or 'randomforestClassifier'
        kwargs: Additional keyword arguments to pass onto the scikit-learn
                clustering object.

    Returns:
        predictor_settings: dictionary of settings for prediction

    """

    # NOTE: function currently located here instead of analysis.py to avoid circular imports

    predictor_settings = {}
    predictor_settings['algorithm'] = algorithm

    def load_class(import_string):
        class_data = import_string.split(".")
        module_path = '.'.join(class_data[:-1])
        class_str = class_data[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_str)

    algs_classify = {
        'svm': 'sklearn.svm.SVC',
        'logistic': 'sklearn.linear_model.LogisticRegression',
        'ridgeClassifier': 'sklearn.linear_model.RidgeClassifier',
        'ridgeClassifierCV': 'sklearn.linear_model.RidgeClassifierCV',
        'randomforestClassifier': 'sklearn.ensemble.RandomForestClassifier'
        }
    algs_predict = {
        'svr': 'sklearn.svm.SVR',
        'linear': 'sklearn.linear_model.LinearRegression',
        'lasso': 'sklearn.linear_model.Lasso',
        'lassoCV': 'sklearn.linear_model.LassoCV',
        'ridge': 'sklearn.linear_model.Ridge',
        'ridgeCV': 'sklearn.linear_model.RidgeCV',
        'randomforest': 'sklearn.ensemble.RandomForest'
        }

    if algorithm in algs_classify.keys():
        predictor_settings['prediction_type'] = 'classification'
        alg = load_class(algs_classify[algorithm])
        predictor_settings['predictor'] = alg(**kwargs)
    elif algorithm in algs_predict:
        predictor_settings['prediction_type'] = 'prediction'
        alg = load_class(algs_predict[algorithm])
        predictor_settings['predictor'] = alg(**kwargs)
    elif algorithm == 'lassopcr':
        predictor_settings['prediction_type'] = 'prediction'
        from sklearn.linear_model import Lasso
        from sklearn.decomposition import PCA
        predictor_settings['_lasso'] = Lasso()
        predictor_settings['_pca'] = PCA()
        predictor_settings['predictor'] = Pipeline(
                            steps=[('pca', predictor_settings['_pca']),
                            ('lasso', predictor_settings['_lasso'])])
    elif algorithm == 'pcr':
        predictor_settings['prediction_type'] = 'prediction'
        from sklearn.linear_model import LinearRegression
        from sklearn.decomposition import PCA
        predictor_settings['_regress'] = LinearRegression()
        predictor_settings['_pca'] = PCA()
        predictor_settings['predictor'] = Pipeline(
                            steps=[('pca', predictor_settings['_pca']),
                            ('regress', predictor_settings['_regress'])])
    else:
        raise ValueError("""Invalid prediction/classification algorithm name.
            Valid options are 'svm','svr', 'linear', 'logistic', 'lasso',
            'lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
            'randomforest', or 'randomforestClassifier'.""")

    return predictor_settings


# The following are nipy source code implementations of the hemodynamic response function HRF
# See the included nipy license file for use permission.

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

def make_cosine_basis(nsamples,sampling_freq,filter_length):
    """ Create a series of cosines basic functions for discrete cosine transform. Based off of implementation in spm_dctmtx because scipy dct can only apply transforms but not return the basis functions.

    Args:
        nsamples (int): number of observations (e.g. TRs)
        sampling_freq (float): sampling frequency in seconds (e.g. TR length)
        filter_length (int): length of filter in seconds

    Returns:
        out (ndarray): nsamples x number of basis sets numpy array

    """

    #Figure out number of basis functions to create
    order = int(np.fix(2 * (nsamples * sampling_freq)/filter_length + 1))

    n = np.arange(nsamples)

    #Initialize basis function matrix
    C = np.zeros((len(n),order))

    #Add constant
    C[:,0] = np.ones((1,len(n)))/np.sqrt(nsamples)

    #Insert higher order cosine basis functions
    for i in xrange(1,order):
        C[:,i] = np.sqrt(2./nsamples) * np.cos(np.pi*(2*n+1) * i/(2*nsamples))

    #Drop constant
    C = C[:,1:]
    if C.size == 0:
        raise ValueError('Basis function creation failed! nsamples is too small for requested filter_length.')
    else:
        return C
