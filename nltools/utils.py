'''
NeuroLearn Utilities
====================

handy utilities.

'''
__all__ = ['get_resource_path',
            'get_anatomical',
            'set_algorithm']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from os.path import dirname, join, pardir, sep as pathsep
import pandas as pd
import nibabel as nib
import importlib
import os
from sklearn.pipeline import Pipeline

def get_resource_path():
    """ Get path to nltools resource directory. """
    return join(dirname(__file__), 'resources') + pathsep

def get_anatomical():
    """ Get nltools default anatomical image. """
    return nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'))

def set_algorithm(algorithm, **kwargs):
    """ Setup the algorithm to use in subsequent prediction analyses.

    Args:
        algorithm: The prediction algorithm to use. Either a string or an (uninitialized)
        scikit-learn prediction object. If string, must be one of 'svm','svr', linear',
        'logistic','lasso','lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
        'randomforest', or 'randomforestClassifier'
        kwargs: Additional keyword arguments to pass onto the scikit-learn clustering
        object.

    Returns:
        predictor_settings: dictionary of settings for prediction

    """

    # NOTE: function currently located here instead of analysis.py to avoid circular imports

    predictor_settings={}
    predictor_settings['algorithm'] = algorithm

    def load_class(import_string):
        class_data = import_string.split(".")
        module_path = '.'.join(class_data[:-1])
        class_str = class_data[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_str)

    algs_classify = {
        'svm':'sklearn.svm.SVC',
        'logistic':'sklearn.linear_model.LogisticRegression',
        'ridgeClassifier':'sklearn.linear_model.RidgeClassifier',
        'ridgeClassifierCV':'sklearn.linear_model.RidgeClassifierCV',
        'randomforestClassifier':'sklearn.ensemble.RandomForestClassifier'
        }
    algs_predict = {
        'svr':'sklearn.svm.SVR',
        'linear':'sklearn.linear_model.LinearRegression',
        'lasso':'sklearn.linear_model.Lasso',
        'lassoCV':'sklearn.linear_model.LassoCV',
        'ridge':'sklearn.linear_model.Ridge',
        'ridgeCV':'sklearn.linear_model.RidgeCV',
        'randomforest':'sklearn.ensemble.RandomForest'
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
        predictor_settings['predictor'] = Pipeline(steps=[('pca', predictor_settings['_pca']), ('lasso', predictor_settings['_lasso'])])
    elif algorithm == 'pcr':
        predictor_settings['prediction_type'] = 'prediction'
        from sklearn.linear_model import LinearRegression
        from sklearn.decomposition import PCA
        predictor_settings['_regress'] = LinearRegression()
        predictor_settings['_pca'] = PCA()
        predictor_settings['predictor'] = Pipeline(steps=[('pca', predictor_settings['_pca']), ('regress', predictor_settings['_regress'])])
    else:
        raise ValueError("""Invalid prediction/classification algorithm name. Valid
            options are 'svm','svr', 'linear', 'logistic', 'lasso', 'lassopcr',
            'lassoCV','ridge','ridgeCV','ridgeClassifier', 'randomforest', or
            'randomforestClassifier'.""")

    return predictor_settings


