'''
    NeuroLearn Data Classes
    =========================
    Classes to represent various types of fdata

    Author: Luke Chang
    License: MIT
'''

## Notes:
# Might consider moving anatomical field out of object and just request when needed.  Probably only when plotting
# Need to figure out how to speed up loading and resampling of data

__all__ = ['Brain_Data',
            ]

import os
import nibabel as nib
from nltools.utils import get_resource_path
from nilearn.input_data import NiftiMasker
from copy import deepcopy
import pandas as pd
import numpy as np
from nilearn.plotting.img_plotting import plot_epi, plot_roi
from scipy.stats import ttest_1samp
from scipy.stats import t

import importlib
import sklearn
from sklearn.pipeline import Pipeline
from nilearn.input_data import NiftiMasker

class Brain_Data:

    def __init__(self, data=None, Y=None, X=None, mask=None, output_file=None, anatomical=None, **kwargs):
        """ Initialize Brain_Data Instance.

        Args:
            data: nibabel data instance or list of files
            Y: vector of training labels
            X: Pandas DataFrame Design Matrix for running univariate models 
            mask: binary nifiti file to mask brain data
            output_file: Name to write out to nifti file
            anatomical: anatomical image to overlay plots
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        if mask is not None:
            if not isinstance(mask, nib.Nifti1Image):
                raise ValueError("mask is not a nibabel instance")
            self.mask = mask
        else:
            self.mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))

        if anatomical is not None:
            if not isinstance(anatomical, nib.Nifti1Image):
                raise ValueError("anatomical is not a nibabel instance")
            self.anatomical = anatomical
        else:
            self.anatomical = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'))

        if type(data) is str:
            data=nib.load(data)
        elif type(data) is list:
            data=nib.concat_images(data)
        elif not isinstance(data, nib.Nifti1Image):
            raise ValueError("data is not a nibabel instance")

        self.nifti_masker = NiftiMasker(mask_img=mask)
        self.data = self.nifti_masker.fit_transform(data)

        if Y is not None:
            if type(Y) is str:
                if os.path.isfile(Y):
                    Y=np.array(pd.read_csv(Y,header=None,index_col=None))
            elif type(Y) is list:
                Y=np.array(Y)
            if self.data.shape[0]!= len(Y):
                raise ValueError("Y does not match the correct size of data")
            self.Y = Y
        else:
            self.Y = []

        if X is not None:
            if self.data.shape[0]!= X.shape[0]:
                raise ValueError("X does not match the correct size of data")
            self.X = X
        else:
            self.X = pd.DataFrame()

        if output_file is not None:
            self.file_name = output_file
        else:
            self.file_name = []

    def __repr__(self):
        return '%s.%s(data=%s, Y=%s, X=%s, mask=%s, output_file=%s, anatomical=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape(),
            self.Y.shape,
            self.X.shape,
            os.path.basename(self.mask.get_filename()),
            self.file_name,
            os.path.basename(self.anatomical.get_filename())            
            )

    def __getitem__(self, index):
        new = deepcopy(self)
        if isinstance(index, int):
            new.data = np.array(self.data[index,:]).flatten()
        elif isinstance(index, slice):
            new.data = np.array(self.data[index,:])            
        else:
            raise TypeError("index must be int or slice")
        if self.Y.size:
            new.Y = self.Y[index]
        if self.X.size:
            new.X = self.X[:,index]
        return new

    def shape(self):
        """ Get images by voxels shape.

        Args:
            self: Brain_Data instance

        """

        return self.data.shape

    def mean(self):
        """ Get mean of each voxel across images.

        Args:
            self: Brain_Data instance

        Returns:
            out: Brain_Data instance
        
        """ 

        out = deepcopy(self)
        out.data = np.mean(out.data, axis=0)
        return out

    def std(self):
        """ Get standard deviation of each voxel across images.

        Args:
            self: Brain_Data instance

        Returns:
            out: Brain_Data instance
        
        """ 

        out = deepcopy(self)
        out.data = np.std(out.data, axis=0)
        return out

    def to_nifti(self):
        """ Convert Brain_Data Instance into Nifti Object

        Args:
            self: Brain_Data instance
        
        """
        
        nifti_dat = self.nifti_masker.inverse_transform(self.data)
        return nifti_dat

    def write(self, file_name=None):
        """ Write out Brain_Data object to Nifti File.

        Args:
            self: Brain_Data instance
            file_name: name of nifti file

        """

        self.to_nifti().to_filename(file_name)

    def plot(self, limit=5):
        """ Create a quick plot of self.data.  Will plot each image separately

        Args:
            self: Brain_Data instance
            limit: max number of images to return
            mask: Binary nifti mask to calculate mean

        """

        if self.data.ndim == 1:
            plot_roi(self.to_nifti(), self.anatomical)
        else:
            for i in xrange(self.data.shape[0]):
                if i < limit:
                    plot_roi(self.nifti_masker.inverse_transform(self.data[i,:]), self.anatomical)


    def regress(self):
        """ run vectorized OLS regression across voxels.

        Args:
            self: Brain_Data instance

        Returns:
            out: dictionary of regression statistics in Brain_Data instances {'beta','t','p','df','residual'}
        
        """ 

        if not isinstance(self.X, pd.DataFrame):
            raise ValueError('Make sure self.X is a pandas DataFrame.')

        if self.X.empty:
            raise ValueError('Make sure self.X is not empty.')

        if self.data.shape[0]!= self.X.shape[0]:
            raise ValueError("self.X does not match the correct size of self.data")

        b = np.dot(np.linalg.pinv(self.X), self.data)
        res = self.data - np.dot(self.X,b)
        sigma = np.std(res,axis=0)
        stderr = np.dot(np.matrix(np.diagonal(np.linalg.inv(np.dot(self.X.T,self.X)))**.5).T,np.matrix(sigma))
        b_out = deepcopy(self)
        b_out.data = b
        t_out = deepcopy(self)
        t_out.data = b /stderr
        df = np.array([self.X.shape[0]-self.X.shape[1]] * t_out.data.shape[1])
        p_out = deepcopy(self)
        p_out.data = 2*(1-t.cdf(np.abs(t_out.data),df))

 
        # Might want to not output this info
        df_out = deepcopy(self)
        df_out.data = df
        sigma_out = deepcopy(self)
        sigma_out.data = sigma
        res_out = deepcopy(self)
        res_out.data = res

        return {'beta':b_out, 't':t_out, 'p':p_out, 'df':df_out, 'sigma':sigma_out, 'residual':res_out}

    def ttest(self, threshold_dict=None):
        """ Calculate one sample t-test across each voxel (two-sided)

        Args:
            self: Brain_Data instance
            threshold_dict: a dictionary of threshold parameters {'unc':.001} or {'fdr':.05}

        Returns:
            out: dictionary of regression statistics in Brain_Data instances {'t','p'}
        
        """ 

        # Notes:  Need to add FDR Option

        t = deepcopy(self)
        p = deepcopy(self)
        t.data, p.data = ttest_1samp(self.data, 0, 0)

        if threshold_dict is not None:
            if type(threshold_dict) is dict:
                if 'unc' in threshold_dict:
                    #Uncorrected Thresholding
                    t.data[np.where(p.data>threshold_dict['unc'])] = np.nan
                elif 'fdr' in threshold_dict:
                    pass
            else:
                raise ValueError("threshold_dict is not a dictionary.  Make sure it is in the form of {'unc':.001} or {'fdr':.05}")

        out = {'t':t, 'p':p}

        return out

    def append(self, data):
        """ Append data to Brain_Data instance

        Args:
            data: Brain_Data instance to append
        
        """   

        if not isinstance(data, Brain_Data):
            raise ValueError('Make sure data is a Brain_Data instance')
 
        out = deepcopy(self)

        if out.isempty():
            out.data = data.data            
        else:
            if len(self.shape())==1 & len(data.shape())==1:
                if self.shape()[0]!=data.shape()[0]:
                    raise ValueError('Data is a different number of voxels then the weight_map.')
            elif len(self.shape())==1 & len(data.shape())>1:
                if self.shape()[0]!=data.shape()[1]:
                    raise ValueError('Data is a different number of voxels then the weight_map.')
            elif len(self.shape())>1 & len(data.shape())==1:
                if self.shape()[1]!=data.shape()[0]:
                    raise ValueError('Data is a different number of voxels then the weight_map.')
            elif self.shape()[1]!=data.shape()[1]:
                raise ValueError('Data is a different number of voxels then the weight_map.')

            out.data = np.vstack([self.data,data.data])

        return out

    def empty(self):
        """ Initalize Brain_Data.data as empty
        
        """
        
        tmp = deepcopy(self)
        tmp.data = []
        # tmp.data = np.array([]).reshape(0,n_voxels)
        return tmp

    def isempty(self):
        """ Check if Brain_Data.data is empty
        
        Returns:
            bool
        """ 

        if not isinstance(self, Brain_Data):
            raise ValueError('Make sure data is a Brain_Data instance')

        if isinstance(self.data, list):
            if not self.data:
                boolean = True
            else:
                boolean = False
        else: #need to make this more precise later
            # if isinstance(self.data, np.ndarray) or isinstance(self.data, np.array) or isinstance(self.data, np.matrix):
            boolean = False
        
        return boolean

    def predict(self, algorithm=None, cv_dict=None, save_images=False, save_output=False,
                save_plot=False, **kwargs):

        """ Run prediction

        Args:
            algorithm: Algorithm to use for prediction.  Must be one of 'svm', 'svr',
            'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest',
            or 'randomforestClassifier'
            cv_dict: Type of cross_validation to use. A dictionary of
                {'type': 'kfolds', 'n_folds': n},
                {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                {'type': 'loso'', 'subject_id': holdout},
                where n = number of folds, and subject = vector of subject ids that corresponds to self.Y
            save_images: Boolean indicating whether or not to save images to file.
            save_output: Boolean indicating whether or not to save prediction output to file.
            save_plot: Boolean indicating whether or not to create plots.
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        def set_algorithm(self, algorithm, **kwargs):
            """ Set the algorithm to use in subsequent prediction analyses.

            Args:
                algorithm: The prediction algorithm to use. Either a string or an (uninitialized)
                scikit-learn prediction object. If string, must be one of 'svm','svr', linear',
                'logistic','lasso','lassopcr','lassoCV','ridge','ridgeCV','ridgeClassifier',
                'randomforest', or 'randomforestClassifier'
                kwargs: Additional keyword arguments to pass onto the scikit-learn clustering
                object.

            """

            self.algorithm = algorithm

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
                self.prediction_type = 'classification'
                alg = load_class(algs_classify[algorithm])
                self.predictor = alg(**kwargs)
            elif algorithm in algs_predict:
                self.prediction_type = 'prediction'
                alg = load_class(algs_predict[algorithm])
                self.predictor = alg(**kwargs)
            elif algorithm == 'lassopcr':
                self.prediction_type = 'prediction'
                from sklearn.linear_model import Lasso
                from sklearn.decomposition import PCA
                self._lasso = Lasso()
                self._pca = PCA()
                self.predictor = Pipeline(steps=[('pca', self._pca), ('lasso', self._lasso)])
            elif algorithm == 'pcr':
                self.prediction_type = 'prediction'
                from sklearn.linear_model import LinearRegression
                from sklearn.decomposition import PCA
                self._regress = LinearRegression()
                self._pca = PCA()
                self.predictor = Pipeline(steps=[('pca', self._pca), ('regress', self._regress)])
            else:
                raise ValueError("""Invalid prediction/classification algorithm name. Valid
                    options are 'svm','svr', 'linear', 'logistic', 'lasso', 'lassopcr',
                    'lassoCV','ridge','ridgeCV','ridgeClassifier', 'randomforest', or
                    'randomforestClassifier'.""")

            def set_cv(self, cv_dict):
                """ Set the CV algorithm to use in subsequent prediction analyses.

                Args:
                    cv_dict: Type of cross_validation to use. A dictionary of
                        {'type': 'kfolds', 'n_folds': n},
                        {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                        {'type': 'loso'', 'subject_id': holdout},

                 """

                if 'subject_id' in cv_dict:
                    self.subject_id = np.array(cv_dict['subject_id'])

                if type(cv_dict) is dict:
                    if cv_dict['type'] == 'kfolds':
                        if 'subject_id' in cv_dict:
                            # Hold out subjects within each fold
                            from  nltools.cross_validation import KFoldSubject
                            self.cv = KFoldSubject(len(self.Y), cv_dict['subject_id'], n_folds=cv_dict['n_folds'])
                        else:
                            # Normal Stratified K-Folds
                            from  nltools.cross_validation import KFoldStratified
                            self.cv = KFoldStratified(self.Y, n_folds=cv_dict['n_folds'])
                    elif cv_dict['type'] == 'loso':
                        # Leave One Subject Out
                        from sklearn.cross_validation import LeaveOneLabelOut
                        self.cv = LeaveOneLabelOut(labels=cv_dict['subject_id'])
                    else:
                        raise ValueError("""Make sure you specify a dictionary of
                        {'type': 'kfolds', 'n_folds': n},
                        {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                        {'type': 'loso'', 'subject_id': holdout},
                        where n = number of folds, and subject = vector of subject ids that corresponds to self.Y""")
                else:
                    raise ValueError("Make sure 'cv_dict' is a dictionary.")

        if algorithm is not None:
            self.set_algorithm(algorithm, **kwargs)

        if self.algorithm is None:
            raise ValueError("Make sure you specify an 'algorithm' to use.")

        # Overall Fit for weight map
        predictor = self.predictor
        predictor.fit(self.data, self.Y)
        self.yfit_all = predictor.predict(self.data)
        if self.prediction_type == 'classification':
            if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                self.prob_all = predictor.predict_proba(self.data)
            else:
                dist_from_hyperplane_all = predictor.decision_function(self.data)
                if self.algorithm == 'svm' and self.predictor.probability:
                    self.prob_all = predictor.predict_proba(self.data)

        # Cross-Validation Fit
        if cv_dict is not None:
            self.set_cv(cv_dict)

        dist_from_hyperplane_xval = None

        if hasattr(self, 'cv'):
            predicter_cv = self.predictor
            self.yfit_xval = self.yfit_all.copy()
            if self.prediction_type == 'classification':
                if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                    self.prob_xval = np.zeros(len(self.Y))
                else:
                    dist_from_hyperplane_xval = np.zeros(len(self.Y))
                    if self.algorithm == 'svm' and self.predictor.probability:
                        self.prob_xval = np.zeros(len(self.Y))

            for train, test in self.cv:
                predicter_cv.fit(self.data[train], self.Y[train])
                self.yfit_xval[test] = predicter_cv.predict(self.data[test])
                if self.prediction_type == 'classification':
                    if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                        self.prob_xval[test] = predicter_cv.predict_proba(self.data[test])
                    else:
                        dist_from_hyperplane_xval[test] = predicter_cv.decision_function(self.data[test])
                        if self.algorithm == 'svm' and self.predictor.probability:
                            self.prob_xval[test] = predicter_cv.predict_proba(self.data[test])

        # Save Outputs
        if save_images:
            self._save_image(predictor)

        if save_output:
            self._save_stats_output(dist_from_hyperplane_xval)

        if save_plot:
            if hasattr(self, 'cv'):
                self._save_plot(predicter_cv)
            else:
                self._save_plot(predictor)

        # Print Results
        if self.prediction_type == 'classification':
            self.mcr_all = np.mean(self.yfit_all==self.Y)
            print 'overall accuracy: %.2f' % self.mcr_all
            if hasattr(self,'cv'):
                self.mcr_xval = np.mean(self.yfit_xval==self.Y)
                print 'overall CV accuracy: %.2f' % self.mcr_xval
        elif self.prediction_type == 'prediction':
            self.rmse_all = np.sqrt(np.mean((self.yfit_all-self.Y)**2))
            self.r_all = np.corrcoef(self.Y,self.yfit_all)[0,1]
            print 'overall Root Mean Squared Error: %.2f' % self.rmse_all
            print 'overall Correlation: %.2f' % self.r_all
            if hasattr(self,'cv'):
                self.rmse_xval = np.sqrt(np.mean((self.yfit_xval-self.Y)**2))
                self.r_xval = np.corrcoef(self.Y,self.yfit_xval)[0,1]
                print 'overall CV Root Mean Squared Error: %.2f' % self.rmse_xval
                print 'overall CV Correlation: %.2f' % self.r_xval

    def similarity(self, image=None, method='correlation', ignore_missing=True):
        """ Calculate similarity of Brain_Data() instance with single Brain_Data image

            Args:
                self: Brain_Data instance of data to be applied
                weight_map: Brain_Data instance of weight map
                **kwargs: Additional parameters to pass

            Returns:
                pexp: Outputs a vector of pattern expression values

        """

        if not isinstance(self, Brain_Data):
            raise ValueError('Make sure data is a Brain_Data instance')

        if not isinstance(image, Brain_Data):
            raise ValueError('Make sure image is a Brain_Data instance')

        if self.shape()[1]!=image.shape()[0]:
            print 'Warning: Different number of voxels detected.  Resampling image into data space.'

            # raise ValueError('Data is a different number of voxels then the image.')

        # Calculate pattern expression
        if method is 'dot_product':
            pexp = np.dot(self.data, image.data)
        elif method is 'correlation':
            pexp=[]
            for w in xrange(self.data.shape[0]):
                pexp.append(pearson(self.data[w,:], image.data))
            pexp = np.array(pexp).flatten()
        return pexp

    def resample(self, target):
        """ Resample data into target space

        Args:
            self: Brain_Data instance
            target: Brain_Data instance of target space
        
        """ 

        if not isinstance(target, Brain_Data):
            raise ValueError('Make sure target is a Brain_Data instance')
 
        pass

