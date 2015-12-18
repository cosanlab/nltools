'''
    NeuroLearn Data Classes
    =========================
    Classes to represent various types of fdata

    Author: Luke Chang
    License: MIT
'''

## Notes:
# Need to figure out how to speed up loading and resampling of data

__all__ = ['Brain_Data']

import os
import nibabel as nib
from nltools.utils import get_resource_path, set_algorithm, get_anatomical
from nltools.cross_validation import set_cv
from nltools.plotting import dist_from_hyperplane_plot, scatterplot, probability_plot, roc_plot
from nltools.stats import pearson
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img
from nilearn.masking import intersect_masks
from nilearn.plotting.img_plotting import plot_epi, plot_roi, plot_stat_map
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, t, norm

import sklearn
from sklearn.pipeline import Pipeline
from nilearn.input_data import NiftiMasker

class Brain_Data(object):

    def __init__(self, data=None, Y=None, X=None, mask=None, output_file=None, resample=True, **kwargs):
        """ Initialize Brain_Data Instance.

        Args:
            data: nibabel data instance or list of files
            Y: vector of training labels
            X: Pandas DataFrame Design Matrix for running univariate models 
            mask: binary nifiti file to mask brain data
            output_file: Name to write out to nifti file
            resample: resample to mask space (on by default)
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        if mask is not None:
            if not isinstance(mask, nib.Nifti1Image):
                if type(mask) is str:
                    if os.path.isfile(mask):
                        mask = nib.load(mask)
            else:
                raise ValueError("mask is not a nibabel instance")
            self.mask = mask
        else:
            self.mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))

        if type(data) is str:
            data=nib.load(data)
        elif type(data) is list:
            data=nib.concat_images(data)
        elif not isinstance(data, nib.Nifti1Image):
            raise ValueError("data is not a nibabel instance")

        # Check if data need to be resampled into mask space
        if not ((self.mask.get_affine()==data.get_affine()).all()) & (self.mask.shape[0:3]==data.shape[0:3]):
            data = resample_img(data,target_affine=self.mask.get_affine(),target_shape=self.mask.shape)

        self.nifti_masker = NiftiMasker(mask_img=self.mask)
        self.data = self.nifti_masker.fit_transform(data)

        if Y is not None:
            if type(Y) is str:
                if os.path.isfile(Y):
                    Y=np.array(pd.read_csv(Y,header=None,index_col=None))
            elif type(Y) is list:
                Y=np.array(Y)
            if self.data.shape[0]!= len(Y):
                raise ValueError("Y does not match the correct size of data")
            if 1 in Y.shape:
                self.Y = np.array(Y).flatten()
            else:
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
        return '%s.%s(data=%s, Y=%s, X=%s, mask=%s, output_file=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.shape(),
            self.Y.shape if self.Y else self.Y,
            self.X.shape,
            os.path.basename(self.mask.get_filename()),
            self.file_name
            )

    def __getitem__(self, index):
        new = deepcopy(self)
        if isinstance(index, int):
            new.data = np.array(self.data[index,:]).flatten()
        else:
            new.data = np.array(self.data[index,:])           
        if self.Y:
            if self.Y.size:
                new.Y = self.Y[index]
        if self.X.size:
            if isinstance(self.X,pd.DataFrame):
                new.X = self.X.iloc[index]
            else:
                new.X = self.X[index,:]
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

    def plot(self, limit=5, anatomical=None):
        """ Create a quick plot of self.data.  Will plot each image separately

        Args:
            limit: max number of images to return
            anatomical: nifti image or file name to overlay

        """

        if anatomical is not None:
            if not isinstance(anatomical, nib.Nifti1Image):
                if type(anatomical) is str:
                    anatomical = nib.load(anatomical)
                else:
                    raise ValueError("anatomical is not a nibabel instance")
        else:
            anatomical = get_anatomical()

    
        if self.data.ndim == 1:
            plot_stat_map(self.to_nifti(), anatomical, cut_coords=range(-40, 50, 10), display_mode='z', 
                black_bg=True, colorbar=True, draw_cross=False)
        else:
            for i in xrange(self.data.shape[0]):
                if i < limit:
                    # plot_roi(self.nifti_masker.inverse_transform(self.data[i,:]), self.anatomical)
                    # plot_stat_map(self.nifti_masker.inverse_transform(self.data[i,:]), 
                    plot_stat_map(self[i].to_nifti(), anatomical, cut_coords=range(-40, 50, 10), display_mode='z', 
                        black_bg=True, colorbar=True, draw_cross=False)

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

        Returns:
            out: new appended Brain_Data instance
        """

        if not isinstance(data, Brain_Data):
            raise ValueError('Make sure data is a Brain_Data instance')
 
        if self.isempty():
            out = deepcopy(data)           
        else:
            out = deepcopy(self)
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
            if out.Y.size:
                out.Y = np.vstack([self.Y, data.Y])
            if self.X.size:
                if isinstance(self.X,pd.DataFrame):
                    out.X = self.X.append(data.X)
                else:
                    out.X = np.vstack([self.X, data.X])
        return out

    def empty(self, data=True, Y=True, X=True):
        """ Initalize Brain_Data.data as empty
        
        """
        
        tmp = deepcopy(self)
        if data:
            tmp.data = np.array([])
        if Y:
            tmp.Y = np.array([])
        if X:
            tmp.X = np.array([])
        # tmp.data = np.array([]).reshape(0,n_voxels)
        return tmp

    def isempty(self):
        """ Check if Brain_Data.data is empty
        
        Returns:
            bool
        """ 

        if isinstance(self.data,np.ndarray):
            if self.data.size:
                boolean = False
            else:
                boolean = True

        if isinstance(self.data, list):
            if not self.data:
                boolean = True
            else:
                boolean = False
        
        return boolean

    def similarity(self, image=None, method='correlation'):
        """ Calculate similarity of Brain_Data() instance with single Brain_Data or Nibabel image

            Args:
                self: Brain_Data instance of data to be applied
                image: Brain_Data or Nibabel instance of weight map

            Returns:
                pexp: Outputs a vector of pattern expression values

        """

        if not isinstance(image, Brain_Data):
            if isinstance(image, nib.Nifti1Image):
                image = self.nifti_masker.fit_transform(image)
            else:
                raise ValueError("Image is not a Brain_Data or nibabel instance")
        dim = image.shape()

        # Check to make sure masks are the same for each dataset and if not create a union mask
        # This might be handy code for a new Brain_Data method
        if np.sum(self.nifti_masker.mask_img.get_data()==1)!=np.sum(image.nifti_masker.mask_img.get_data()==1):
            new_mask = intersect_masks([self.nifti_masker.mask_img, image.nifti_masker.mask_img], threshold=1, connected=False)
            new_nifti_masker = NiftiMasker(mask_img=new_mask)
            data2 = new_nifti_masker.fit_transform(self.to_nifti())
            image2 = new_nifti_masker.fit_transform(image.to_nifti())
        else:
            data2 = self.data
            image2 = image.data

        # Calculate pattern expression
        if method is 'dot_product':
            pexp = np.dot(data2, image2)
        elif method is 'correlation':
            pexp = pearson(data2, image2)
        return pexp

    def predict(self, algorithm=None, cv_dict=None, plot=True, **kwargs):

        """ Run prediction

        Args:
            algorithm: Algorithm to use for prediction.  Must be one of 'svm', 'svr',
            'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest',
            or 'randomforestClassifier'
            cv_dict: Type of cross_validation to use. A dictionary of
                {'type': 'kfolds', 'n_folds': n},
                {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                {'type': 'loso', 'subject_id': holdout},
                where n = number of folds, and subject = vector of subject ids that corresponds to self.Y
            plot: Boolean indicating whether or not to create plots.
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        Returns:
            output: a dictionary of prediction parameters

        """

        # Set algorithm
        if algorithm is not None:
            predictor_settings = set_algorithm(algorithm, **kwargs)
        else:
            # Use SVR as a default
            predictor_settings = set_algorithm('svr', **{'kernel':"linear"})

        # Initialize output dictionary
        output = {}
        output['Y'] = self.Y
        
        # Overall Fit for weight map
        predictor = predictor_settings['predictor']
        predictor.fit(self.data, self.Y)
        output['yfit_all'] = predictor.predict(self.data)
        if predictor_settings['prediction_type'] == 'classification':
            if predictor_settings['algorithm'] not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                output['prob_all'] = predictor.predict_proba(self.data)
            else:
                output['dist_from_hyperplane_all'] = predictor.decision_function(self.data)
                if predictor_settings['algorithm'] == 'svm' and predictor.probability:
                    output['prob_all'] = predictor.predict_proba(self.data)
       
        output['intercept'] = predictor.intercept_

        # Weight map
        output['weight_map'] = deepcopy(self)
        if predictor_settings['algorithm'] == 'lassopcr':
            output['weight_map'].data = np.dot(predictor_settings['_pca'].components_.T,predictor_settings['_lasso'].coef_)
        elif predictor_settings['algorithm'] == 'pcr':
            output['weight_map'].data = np.dot(predictor_settings['_pca'].components_.T,predictor_settings['_regress'].coef_)
        else:
            output['weight_map'].data = predictor.coef_.squeeze()

        # Cross-Validation Fit
        if cv_dict is not None:
            output['cv'] = set_cv(cv_dict)

            predictor_cv = predictor_settings['predictor']
            output['yfit_xval'] = output['yfit_all'].copy()
            output['intercept_xval'] = []
            output['weight_map_xval'] = deepcopy(output['weight_map'])
            wt_map_xval = [];
            if predictor_settings['prediction_type'] == 'classification':
                if predictor_settings['algorithm'] not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                    output['prob_xval'] = np.zeros(len(self.Y))
                else:
                    dist_from_hyperplane_xval = np.zeros(len(self.Y))
                    if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                        output['prob_xval'] = np.zeros(len(self.Y))

            for train, test in output['cv']:
                predictor_cv.fit(self.data[train], self.Y[train])
                output['yfit_xval'][test] = predictor_cv.predict(self.data[test])
                if predictor_settings['prediction_type'] == 'classification':
                    if predictor_settings['algorithm'] not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                        output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])
                    else:
                        output['dist_from_hyperplane_xval'][test] = predictor_cv.decision_function(self.data[test])
                        if predictor_settings['algorithm'] == 'svm' and predictor_cv.probability:
                            output['prob_xval'][test] = predictor_cv.predict_proba(self.data[test])
                output['intercept_xval'].append(predictor_cv.intercept_)

                # Weight map
                if predictor_settings['algorithm'] == 'lassopcr':
                    wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T,predictor_settings['_lasso'].coef_))
                elif predictor_settings['algorithm'] == 'pcr':
                    wt_map_xval.append(np.dot(predictor_settings['_pca'].components_.T,predictor_settings['_regress'].coef_))
                else:
                    wt_map_xval.append(predictor_cv.coef_.squeeze())
                output['weight_map_xval'].data = np.array(wt_map_xval)
        
        # Print Results
        if predictor_settings['prediction_type'] == 'classification':
            output['mcr_all'] = np.mean(output['yfit_all']==self.Y)
            print 'overall accuracy: %.2f' % output['mcr_all']
            if cv_dict is not None:
                output['mcr_xval'] = np.mean(output['yfit_xval']==self.Y)
                print 'overall CV accuracy: %.2f' % output['mcr_xval']
        elif predictor_settings['prediction_type'] == 'prediction':
            output['rmse_all'] = np.sqrt(np.mean((output['yfit_all']-self.Y)**2))
            output['r_all'] = np.corrcoef(self.Y,output['yfit_all'])[0,1]
            print 'overall Root Mean Squared Error: %.2f' % output['rmse_all']
            print 'overall Correlation: %.2f' % output['r_all']
            if cv_dict is not None:
                output['rmse_xval'] = np.sqrt(np.mean((output['yfit_xval']-self.Y)**2))
                output['r_xval'] = np.corrcoef(self.Y,output['yfit_xval'])[0,1]
                print 'overall CV Root Mean Squared Error: %.2f' % output['rmse_xval']
                print 'overall CV Correlation: %.2f' % output['r_xval']

        # Plot
        if plot:
            fig1 = output['weight_map'].plot()
            if predictor_settings['prediction_type'] == 'prediction':
                fig2 = scatterplot(pd.DataFrame({'Y': output['Y'], 'yfit_xval':output['yfit_xval']}))
            elif self.prediction_type == 'classification':
                if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                    fig2 = probability_plot(pd.DataFrame({'Y': output['Y'], 'Probability_xval':output['prob_xval']})) 
                else:
                    fig2 = dist_from_hyperplane_plot(pd.DataFrame({'Y': output['Y'], 'dist_from_hyperplane_xval':output['dist_from_hyperplane_xval']}))
                    if self.algorithm == 'svm' and self.predictor.probability:
                        fig3 = probability_plot(pd.DataFrame({'Y': output['Y'], 'Probability_xval':output['prob_xval']}))

        return output

    def bootstrap(self, analysis_type=None, n_samples=10, save_weights=False, **kwargs):
        """ Bootstrap various Brain_Data analaysis methods (e.g., mean, std, regress, predict).  Currently  

        Args:
            analysis_type: Type of analysis to bootstrap (mean,std,regress,predict)
            n_samples: Number of samples to boostrap
            **kwargs: Additional keyword arguments to pass to the analysis method

        Returns:
            output: a dictionary of prediction parameters

        """

        # Notes:
        # might want to add options for [studentized, percentile, bias corrected, bias corrected accelerated] methods
        # Regress method is pretty convoluted and slow, this should be optimized better.  

        def summarize_bootstrap(sample):
            """ Calculate summary of bootstrap samples

            Args:
                sample: Brain_Data instance of samples

            Returns:
                output: dictionary of Brain_Data summary images
                
            """

            output = {}

            # Calculate SE of bootstraps
            wstd = sample.std()
            wmean = sample.mean()
            wz = deepcopy(wmean)
            wz.data = wmean.data / wstd.data
            wp = deepcopy(wmean)
            wp.data = 2*(1-norm.cdf(np.abs(wz.data)))

            # Create outputs
            output['Z'] = wz
            output['p'] = wp
            output['mean'] = wmean
            if save_weights:
                output['samples'] = sample

            return output

        analysis_list = ['mean','std','regress','predict']
        
        if analysis_type in analysis_list:
            data_row_id = range(self.shape()[0])
            sample = self.empty()
            if analysis_type is 'regress': #initialize dictionary of empty betas
                beta={}
                for i in range(self.X.shape[1]):
                    beta['b' + str(i)] = self.empty()
            for i in range(n_samples):
                this_sample = np.random.choice(data_row_id, size=len(data_row_id), replace=True) # gives sampled row numbers
                if analysis_type is 'mean':
                    sample = sample.append(self[this_sample].mean())
                elif analysis_type is 'std':
                    sample = sample.append(self[this_sample].std())
                elif analysis_type is 'regress':
                    out = self[this_sample].regress()
                    # Aggegate bootstraps for each beta separately
                    for i, b in enumerate(beta.iterkeys()):
                        beta[b]=beta[b].append(out['beta'][i])
                elif analysis_type is 'predict':
                    if 'algorithm' in kwargs:
                        algorithm = kwargs['algorithm']
                        del kwargs['algorithm']
                    else:
                        algorithm='ridge'
                    if 'cv_dict' in kwargs:
                        cv_dict = kwargs['cv_dict']
                        del kwargs['cv_dict']
                    else:
                        cv_dict=None
                    if 'plot' in ['kwargs']:
                        plot=kwargs['plot']
                        del kwargs['plot']
                    else:
                        plot=False
                    out = self[this_sample].predict(algorithm=algorithm,cv_dict=cv_dict, plot=plot,**kwargs)
                    sample = sample.append(out['weight_map'])
        else:
            raise ValueError('The analysis_type you specified (%s) is not yet implemented.' % (analysis_type))

        # Save outputs
        if analysis_type is 'regress':
            reg_out={}
            for i, b in enumerate(beta.iterkeys()):
                reg_out[b] = summarize_bootstrap(beta[b])
            output = {}
            for b in reg_out.iteritems():
                for o in b[1].iteritems():
                    if o[0] in output:
                        output[o[0]] = output[o[0]].append(o[1])
                    else:
                        output[o[0]]=o[1]
        else:
            output = summarize_bootstrap(sample)
        return output

    def apply_mask(self, mask):
        """ Mask Brain_Data instance

        Args:
            mask: mask (Brain_Data or nifti object)
            
        """

        if not isinstance(mask, (nib.Nifti1Image,Brain_Data)):
            if type(mask) is str:
                if os.path.isfile(mask):
                    mask = nib.load(mask)
            else:
                raise ValueError("Mask is not a nibabel instance, Brain_Data instance, or a valid file name.")
        
        # Check if mask need to be resampled into Brain_Data mask space
        if not ((self.mask.get_affine()==mask.get_affine()).all()) & (self.mask.shape[0:3]==mask.shape[0:3]):
            mask = resample_img(mask,target_affine=self.mask.get_affine(),target_shape=self.mask.shape)

        # transform mask into Brain_Data
        # new_mask = self.nifti_masker.fit_transform(mask)

        masked = deepcopy(self)
        nifti_masker = NiftiMasker(mask_img=mask)
        masked.data = nifti_masker.fit_transform(self.to_nifti())
        if len(self.data.shape) > 2:
            masked.data = masked.data.squeeze()
        masked.nifti_masker = nifti_masker
        return masked

    def resample(self, target):
        """ Resample data into target space

        Args:
            self: Brain_Data instance
            target: Brain_Data instance of target space
        
        """ 

        raise NotImplementedError()

def threshold(stat, p, threshold_dict={'unc':.001}):
    """ Calculate one sample t-test across each voxel (two-sided)

    Args:
        stat: Brain_Data instance of arbitrary statistic metric (e.g., beta, t, etc)
        p: Brain_data instance of p-values
        threshold_dict: a dictionary of threshold parameters {'unc':.001} or {'fdr':.05}
 
    Returns:
        out: Thresholded Brain_Data instance
    
    """
 
    if not isinstance(stat, Brain_Data):
        raise ValueError('Make sure stat is a Brain_Data instance')
        
    if not isinstance(p, Brain_Data):
        raise ValueError('Make sure p is a Brain_Data instance')

    out = deepcopy(stat)
    if 'unc' in threshold_dict:
        out.data[p.data > threshold_dict['unc']] = np.nan
    elif 'fdr' in threshold_dict:
        out.data[p.data > threshold_dict['fdr']] = np.nan
    return out

