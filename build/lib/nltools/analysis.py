'''
    NeuroLearn Analysis Tools
    =========================
    These tools provide the ability to quickly run 
    machine-learning analyses on imaging data
    Authors: Luke Chang
    License: MIT
'''

# ToDo
# 1) add roc functionality for classification
# 2) add thresholding functionality
# 3) add bootstrapping functionality
# 4) add tests

import os
import nibabel as nib
import sklearn
from nilearn.input_data import NiftiMasker
import pandas as pd
import numpy as np
from nilearn.plotting import *
import seaborn as sns    

# Paths
resource_dir = os.path.join(os.path.dirname(__file__),os.path.pardir,'resources')
  

class Predict:

    def __init__(self, data, Y, subject_id = None, algorithm=None, cv=None, mask=None, 
                output_dir='.', **kwargs):
        """ Initialize Predict.
        Args:
            data: nibabel data instance
            Y: vector of training labels
            subject_id: vector of labels corresponding to each subject
            algorithm: Algorithm to use for prediction.  Must be one of 'svm', 'svr', 
                'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest', 
                or 'randomforestClassifier'
            cv: Type of cross_validation to use. Either a string or an (uninitialized)
                scikit-learn cv object. If string, must be one of 'kfold' or 'loso'.
            mask: binary nibabel mask
            output_dir: Directory to use for writing all outputs
            **kwargs: Additional keyword arguments to pass to the prediction algorithm
        
        """
        self.output_dir = output_dir
        
        if mask is not None:
            if type(mask) is not nib.nifti1.Nifti1Image:
                raise ValueError("mask is not a nibabel instance")
            self.mask = mask
        else:
            self.mask = nib.load(os.path.join(resource_dir,'MNI152_T1_2mm_brain_mask_dil.nii.gz'))
        
        if type(data) is not nib.nifti1.Nifti1Image:
            raise ValueError("data is not a nibabel instance")
        nifti_masker = NiftiMasker(mask_img=mask)
        self.data = nifti_masker.fit_transform(data)
        
        # Could check if running classification or prediction for Y
        if self.data.shape[0]!= len(Y):
            raise ValueError("Y does not match the correct size of data")
        self.Y = Y

        self.set_algorithm(algorithm, **kwargs)

        if cv is not None:
            self.set_cv(cv, **kwargs)

        if subject_id is not None:
            self.subject_id = subject_id    


    def predict(self, algorithm=None, save_images=True, save_output=True, 
                save_plot = True, **kwargs):
        """ Run prediction
        Args:
            algorithm: Algorithm to use for prediction.  Must be one of 'svm', 'svr', 
                'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest', 
                or 'randomforestClassifier'
            save_images: Boolean indicating whether or not to save images to file.
            save_output: Boolean indicating whether or not to save prediction output to file.
            save_plot: Boolean indicating whether or not to create plots.
            **kwargs: Additional keyword arguments to pass to the prediction algorithm
        """
            
        if algorithm is not None:
            self.set_algorithm(algorithm, **kwargs)
        
        # Overall Fit for weight map
        predicter = self.predicter
        predicter.fit(self.data, self.Y)
        
        if save_images:
            self._save_image(predicter)

        if cv is not None:
            predicter_cv = self.predicter
            self.xval_dist_from_hyperplane = np.array(len(self.Y))
            for train, test in cv:
                predicter_cv.fit(self.data[train], self.Y[train])
                self.yfit[test] = self.predict(self.data[test])
                if algorithm is 'svm':
                    self.xval_dist_from_hyperplane[test] = predicter_cv.decision_function(self.data[test])

            if save_output:
                stats = pd.DataFrame({
                            'SubID' : self.subject_id, 
                            'Y' : self.Y, 
                            'yfit' : self.yfit,
                            'xval_dist_from_hyperplane' : self.xval_dist_from_hyperplane})
                self._save_stats_output(stats)

        if self.prediction_type is 'classification':
            self.mcr = np.mean(self.yfit==self.Y)
            print 'overall CV accuracy: %.2f' % self.mcr
        elif self.prediction_type is 'prediction':
            self.rmse = np.sqrt(np.mean((self.yfit-self.Y)**2))
            self.r = np.corrcoef(Y,yfit)[0,1]
            print 'overall Root Mean Squared Error: %.2f' % self.rmse
            print 'overall Correlation: %.2f' % self.r

        if save_plot:
            self._save_plot


    def set_algorithm(self, algorithm, **kwargs):
        """ Set the algorithm to use in subsequent prediction analyses.
        Args:
            algorithm: The prediction algorithm to use. Either a string or an (uninitialized)
                scikit-learn prediction object. If string, must be one of 'svm', 'svr', 
                'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest', 
                or 'randomforestClassifier'
            kwargs: Additional keyword arguments to pass onto the scikit-learn clustering
                object.
        """

        self.algorithm = algorithm

        if isinstance(algorithm, basestring):

            algs_classify = {
                'svm': sklearn.svm.SVC,
                'logistic': sklearn.linear_model.LogisticRegression,
                'ridgeClassifier': sklearn.linear_model.RidgeClassifier,
                'randomforestClassifier': sklearn.ensemble.RandomForestClassifier
            }
            algs_predict = {
                'svr': sklearn.svm.SVR,
                'linear': sklearn.linear_model.LinearRegression,
                'lasso': sklearn.linear_model.Lasso,
                'ridge': sklearn.linear_model.Ridge,
                'randomforest': sklearn.ensemble.RandomForestClassifier
            }
            if algorithm in algs_classify.keys():
                self.prediction_type = 'classification'
            elif algorithm in algs_predict.keys():
                self.prediction_type = 'prediction'
            else:
                raise ValueError("Invalid prediction algorithm name. Valid options are " + 
                    "'svm','svr', 'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier'" +
                    "'randomforest', or 'randomforestClassifier'.")

            algorithm = algs[algorithm]

        self.predicter = algorithm(**kwargs)


    def set_cv(self, cv, **kwargs):
        """ Set the CV algorithm to use in subsequent prediction analyses.
        Args:
            cv: Type of cross_validation to use. Either a string or an (uninitialized)
                scikit-learn cv object. If string, must be one of 'kfold' or 'loso'.
            **kwargs: Additional keyword arguments to pass onto the scikit-learn cv object.
        """

        self.cv_type = cv

        if isinstance(cv, basestring):

            cvs = {
                'kfold': sklearn.cross_validation.StratifiedKFold,
                'loso': sklearn.cross_validation.LeaveOneLabelOut,
            }

            if cv not in cvs.keys():
                raise ValueError("Invalid cv name. Valid options are 'kfold' or 'loso'.")
            elif cv is 'kfold':
                if n_fold not in kwargs:
                    raise ValueError("Make sure you specify n_fold when using 'kfold' cv.")

            cv = cvs[cv]

        self.cv = cv(**kwargs)


    def _save_image(self, predicter):
        """ Write out weight map to Nifti image. 
        Args:
            predicter: predicter instance
        Outputs:
            predicter_weightmap.nii.gz: Will output a nifti image of weightmap
        """

        if not isdir(self.output_dir):
            os.makedirs(self.output_dir)

        coef_img = nifti_masker.inverse_transform(predicter.coef_)
        nib.save(coef_img, os.path.abspath(self.output_dir, self.algorithm + '_weightmap.nii.gz'))


    def _save_stats_output(self, stats_output):
        """ Write stats output to csv file. 
        Args:
            stats_output: a pandas file with prediction output
        Outputs:
            predicter_stats_output.csv: Will output a csv file of stats output
        """

        if not isdir(self.output_dir):
            os.makedirs(self.output_dir)
        stats_output.to_csv(os.path.join(self.output_dir, self.algorithm + '_Stats_Output.csv'))


    def _save_plot(self, predicter):
        """ Save Plots. 
        Args:
            predicter: predicter instance
        Outputs:
            predicter_weightmap_montage.png: Will output a montage of axial slices of weightmap
            predicter_prediction.png: Will output a plot of prediction
        """

        if not isdir(self.output_dir):
            os.makedirs(self.output_dir)
        
        coef_img = nifti_masker.inverse_transform(predicter.coef_)
        overlay_img = nib.load(os.path.join(resource_dir,'MNI152_T1_2mm_brain.nii.gz'))

        fig1 = plot_stat_map(coef_img, overlay_img, title=algorithm + "weights", 
                            cut_coords=range(-40, 40, 10), display_mode='z')
        fig1.savefig(os.path.join(self.output_dir, self.algorithm + '_weightmap_axial.png'))

        if self.prediction_type == 'classification':
            if self.algorithm == 'svm':
                fig2 = _dist_from_hyperplane_plot(self,stats_output)
                fig2.savefig(os.path.join(self.output_dir, self.algorithm + 
                            '_xVal_Distance_from_Hyperplane.png'))
        elif self.prediction_type == 'prediction':
            fig2 = _scatterplot(self,stats_output)
            fig2.savefig(os.path.join(self.output_dir, self.algorithm + '_scatterplot.png'))


    def _dist_from_hyperplane_plot(self,stats_output):
        """ Save Plots. 
        Args:
            stats_output: a pandas file with prediction output
        Returns:
            fig: Will return a seaborn plot of distance from hyperplane
        """

        fig = sns.factorplot("SubID", "xval_dist_from_hyperplane", hue="Y", data=stats_output,
                            kind='point')
        plt.xlabel('Subject')
        plt.ylabel('Distance from Hyperplane')
        plt.title(self.algorithm + ' Classification')
        return fig


    def _scatterplot(self,stats_output):
        """ Save Plots. 
        Args:
        Returns:
            fig: Will return a seaborn scatterplot
        """

        fig = sns.lmplot("Y", "yfit", data=stats_out)
        plt.xlabel('Y')
        plt.ylabel('yfit')
        plt.title(self.algorithm + ' Prediction')
        return fig


