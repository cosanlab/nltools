'''
NeuroLearn Utilities
====================

handy utilities.

'''
__all__ = ['get_resource_path',
           'get_anatomical',
           'set_algorithm',
           'attempt_to_import',
           'all_same',
           'concatenate',
           '_bootstrap_apply_func',
           'set_decomposition_algorithm'
           ]
__author__ = ["Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, sep as pathsep
import nibabel as nib
import importlib
import os
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
import collections
from types import GeneratorType

   
def _df_meta_to_arr(df):
    """Check what kind of data exists in pandas columns or index. If string return as numpy array 'S' type, otherwise regular numpy array. Used when saving Brain_Data objects to hdf5.
    """
    
    if len(df.columns):
        if isinstance(df.columns[0], str):
            columns = df.columns.values.astype("S")
        else:
            columns = df.columns.values
    else:
        columns = []
    
    if len(df.index):
        if isinstance(df.index[0], str):
            index = df.index.values.astype("S")
        else:
            index = df.index.values
    else:
        index = []

    return columns, index


def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep


def get_anatomical():
    """ Get nltools default anatomical image.
        DEPRECATED. See MNI_Template and resolve_mni_path from nltools.prefs
    """
    return nib.load(os.path.join(get_resource_path(), 'MNI152_T1_2mm.nii.gz'))


def get_mni_from_img_resolution(brain, img_type='plot'):
    """
    Get the path to the resolution MNI anatomical image that matches the resolution of a Brain_Data instance. Used by Brain_Data.plot() and .iplot() to set backgrounds appropriately.
    
    Args:
        brain: Brain_Data instance
    
    Returns:
        file_path: path to MNI image
    """
    
    if img_type not in ['plot', 'brain']:
        raise ValueError("img_type must be 'plot' or 'brain' ")
    
    res_array = np.abs(np.diag(brain.nifti_masker.affine_)[:3])
    voxel_dims = np.unique(abs(res_array))
    if len(voxel_dims) != 1:
        raise ValueError("Voxels are not isometric and cannot be visualized in standard space")
    else:
        dim = str(int(voxel_dims[0])) + 'mm'
        if img_type == 'brain':
            mni = f'MNI152_T1_{dim}_brain.nii.gz'
        else:
            mni = f'MNI152_T1_{dim}.nii.gz'
        return os.path.join(get_resource_path(), mni)


def set_algorithm(algorithm, *args, **kwargs):
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
        predictor_settings['predictor'] = alg(*args, **kwargs)
    elif algorithm in algs_predict:
        predictor_settings['prediction_type'] = 'prediction'
        alg = load_class(algs_predict[algorithm])
        predictor_settings['predictor'] = alg(*args, **kwargs)
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


def set_decomposition_algorithm(algorithm, n_components=None, *args, **kwargs):
    """ Setup the algorithm to use in subsequent decomposition analyses.

    Args:
        algorithm: The decomposition algorithm to use. Either a string or an
                    (uninitialized) scikit-learn decomposition object.
                    If string must be one of 'pca','nnmf', ica','fa',
                    'dictionary', 'kernelpca'.
        kwargs: Additional keyword arguments to pass onto the scikit-learn
                clustering object.

    Returns:
        predictor_settings: dictionary of settings for prediction

    """

    # NOTE: function currently located here instead of analysis.py to avoid circular imports

    def load_class(import_string):
        class_data = import_string.split(".")
        module_path = '.'.join(class_data[:-1])
        class_str = class_data[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_str)

    algs = {
        'pca': 'sklearn.decomposition.PCA',
        'ica': 'sklearn.decomposition.FastICA',
        'nnmf': 'sklearn.decomposition.NMF',
        'fa': 'sklearn.decomposition.FactorAnalysis',
        'dictionary': 'sklearn.decomposition.DictionaryLearning',
        'kernelpca': 'sklearn.decomposition.KernelPCA'}

    if algorithm in algs.keys():
        alg = load_class(algs[algorithm])
        alg = alg(n_components, *args, **kwargs)
    else:
        raise ValueError("""Invalid prediction/classification algorithm name.
            Valid options are 'pca','ica', 'nnmf', 'fa'""")
    return alg


def isiterable(obj):
    ''' Returns True if the object is one of allowable iterable types. '''
    return isinstance(obj, (list, tuple, GeneratorType))


module_names = {}
Dependency = collections.namedtuple('Dependency', 'package value')


def attempt_to_import(dependency, name=None, fromlist=None):
    if name is None:
        name = dependency
    try:
        mod = __import__(dependency, fromlist=fromlist)
    except ImportError:
        mod = None
    module_names[name] = Dependency(dependency, mod)
    return mod


def all_same(items):
    return np.all(x == items[0] for x in items)


def concatenate(data):
    '''Concatenate a list of Brain_Data() or Adjacency() objects'''

    if not isinstance(data, list):
        raise ValueError('Make sure you are passing a list of objects.')

    if all([isinstance(x, data[0].__class__) for x in data]):
        # Temporarily Removing this for circular imports (LC)
        # if not isinstance(data[0], (Brain_Data, Adjacency)):
        #     raise ValueError('Make sure you are passing a list of Brain_Data'
        #                     ' or Adjacency objects.')

        out = data[0].__class__()
        for i in data:
            out = out.append(i)
    else:
        raise ValueError('Make sure all objects in the list are the same type.')
    return out


def _bootstrap_apply_func(data, function, random_state=None, *args, **kwargs):
    '''Bootstrap helper function. Sample with replacement and apply function'''
    random_state = check_random_state(random_state)
    data_row_id = range(data.shape()[0])
    new_dat = data[random_state.choice(data_row_id,
                                       size=len(data_row_id),
                                       replace=True)]
    return getattr(new_dat, function)(*args, **kwargs)


def check_square_numpy_matrix(data):
    '''Helper function to make sure matrix is square and numpy array'''

    from nltools.data import Adjacency

    if isinstance(data, Adjacency):
        data = data.squareform()
    elif isinstance(data, pd.DataFrame):
        data = data.values
    else:
        data = np.array(data)

    if len(data.shape) != 2:
        try:
            data = squareform(data)
        except ValueError:
            raise ValueError("Array does not contain the correct number of elements to be square")
    return data


def check_brain_data(data, mask=None):
    '''Check if data is a Brain_Data Instance.'''
    from nltools.data import Brain_Data

    if not isinstance(data, Brain_Data):
        if isinstance(data, nib.Nifti1Image):
            data = Brain_Data(data, mask=mask)
        else:
            raise ValueError("Make sure data is a Brain_Data instance.")
    else:
        if mask is not None:
            data = data.apply_mask(mask)
    return data

def check_brain_data_is_single(data):
    '''Logical test if Brain_Data instance is a single image
    
    Args:
        data: brain data
    
    Returns:
        (bool)
    
    '''
    data = check_brain_data(data)
    if len(data.shape()) > 1:
        return False
    else:
        return True

def _roi_func(brain, roi, algorithm, cv_dict, **kwargs):
    '''Brain_Data.predict_multi() helper function'''
    return brain.apply_mask(roi).predict(algorithm=algorithm, cv_dict=cv_dict, plot=False, **kwargs)


class AmbiguityError(Exception):
    pass

def generate_jitter(n_trials, mean_time=5, min_time=2, max_time=12, atol=.2):
    '''Generate jitter from exponential distribution with constraints

    Draws from exponential distribution until the distribution satisfies the constraints:
    np.abs(np.mean(min_time > data < max_time) - mean_time) <= atol

    Args:
        n_trials: (int) number of trials to generate jitter
        mean_time: (float) desired mean of distribution
        min_time: (float) desired min of distribution
        max_time: (float) desired max of distribution
        atol: (float) precision of deviation from mean

    Returns:
        data: (np.array) jitter for each trial
 
    '''

    def generate_data(n_trials, scale=5, min_time=2, max_time=12):
        data = []
        i=0
        while i < n_trials:
            datam = np.random.exponential(scale=5)
            if (datam > min_time) & (datam < max_time):
                data.append(datam)
                i+=1
        return data

    mean_diff = False
    while ~mean_diff:
        data = generate_data(n_trials, min_time=min_time, max_time=max_time)
        mean_diff = np.isclose(np.mean(data), mean_time, rtol=0, atol=atol)
    return data
