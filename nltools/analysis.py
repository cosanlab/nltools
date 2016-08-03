from __future__ import division

'''
    NeuroLearn Analysis Tools
    =========================
    These tools provide the ability to quickly run
    machine-learning analyses on imaging data
'''

# ToDo
# 1) add roc functionality for classification
# 2) add thresholding functionality
# 3) add bootstrapping functionality
# 4) add tests
# 5) add within subject checks and plots
# 6) Plot probabilities

__all__ = ['Predict','apply_mask','Roc']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
import importlib
import nibabel as nib
import sklearn
from sklearn.pipeline import Pipeline
from nilearn.input_data import NiftiMasker
import pandas as pd
import numpy as np
from nilearn.plotting import plot_stat_map
import seaborn as sns
import matplotlib.pyplot as plt
from nltools.plotting import dist_from_hyperplane_plot, scatterplot, probability_plot, roc_plot
from nltools.stats import pearson
from nltools.utils import get_resource_path
from nltools.cross_validation import set_cv
from scipy.stats import norm, binom_test
from sklearn.metrics import auc

class Predict(object):

    def __init__(self, data, Y, algorithm=None, cv_dict=None, mask=None,
                 output_dir='.', **kwargs):
        """ Initialize Predict.

        Args:
            data: nibabel data instance
            Y: vector of training labels
            subject_id: vector of labels corresponding to each subject
            algorithm: Algorithm to use for prediction.  Must be one of 'svm', 'svr',
                'linear', 'logistic', 'lasso', 'ridge', 'ridgeClassifier','randomforest',
                or 'randomforestClassifier'
            cv_dict: Type of cross_validation to use. A dictionary of
                {'type': 'kfolds', 'n_folds': n},
                {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
                {'type': 'loso', 'subject_id': holdout},
                where n = number of folds, and subject = vector of subject ids that corresponds to self.Y
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
            self.mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))

        if type(data) is list:
            data=nib.concat_images(data)

        if not isinstance(data,(nib.nifti1.Nifti1Image, nib.nifti1.Nifti1Pair)):
            raise ValueError("data is not a nibabel instance")
        self.nifti_masker = NiftiMasker(mask_img=mask)
        self.data = self.nifti_masker.fit_transform(data)

        if type(Y) is list:
            Y=np.array(Y)
        if self.data.shape[0]!= len(Y):
            raise ValueError("Y does not match the correct size of data")
        self.Y = Y

        if algorithm is not None:
            self.set_algorithm(algorithm, **kwargs)

        if cv_dict is not None:
            self.cv = set_cv(cv_dict)

    def predict(self, algorithm=None, cv_dict=None, save_images=True, save_output=True,
                save_plot=True, **kwargs):

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

        if not hasattr(self,'algorithm'):
            if algorithm is not None:
                self.set_algorithm(algorithm, **kwargs)
            else:
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
            self.cv = set_cv(cv_dict)

        dist_from_hyperplane_xval = None

        if hasattr(self, 'cv'):
            predictor_cv = self.predictor
            self.yfit_xval = self.yfit_all.copy()
            if self.prediction_type == 'classification':
                if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                    self.prob_xval = np.zeros(len(self.Y))
                else:
                    dist_from_hyperplane_xval = np.zeros(len(self.Y))
                    if self.algorithm == 'svm' and self.predictor.probability:
                        self.prob_xval = np.zeros(len(self.Y))

            for train, test in self.cv:
                predictor_cv.fit(self.data[train], self.Y[train])
                self.yfit_xval[test] = predictor_cv.predict(self.data[test])
                if self.prediction_type == 'classification':
                    if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                        self.prob_xval[test] = predictor_cv.predict_proba(self.data[test])
                    else:
                        dist_from_hyperplane_xval[test] = predictor_cv.decision_function(self.data[test])
                        if self.algorithm == 'svm' and self.predictor.probability:
                            self.prob_xval[test] = predictor_cv.predict_proba(self.data[test])

        # Save Outputs
        if save_images:
            self._save_image(predictor)

        if save_output:
            self._save_stats_output(dist_from_hyperplane_xval)

        if save_plot:
            if hasattr(self, 'cv'):
                self._save_plot(predictor_cv)
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

        # There is a newer version of this function in utils, but leaving for now as this is pretty deeply embedded in the object

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

    def _save_image(self, predictor):
        """ Write out weight map to Nifti image.

        Args:
            predictor: predictor instance

        Returns:
            predicter_weightmap.nii.gz: Will output a nifti image of weightmap

        """

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.algorithm == 'lassopcr':
            coef = np.dot(self._pca.components_.T,self._lasso.coef_)
            coef_img = self.nifti_masker.inverse_transform(np.transpose(coef))
        elif self.algorithm == 'pcr':
            coef = np.dot(self._pca.components_.T,self._regress.coef_)
            coef_img = self.nifti_masker.inverse_transform(np.transpose(coef))
        else:
            coef_img = self.nifti_masker.inverse_transform(predictor.coef_.squeeze())
        nib.save(coef_img, os.path.join(self.output_dir, self.algorithm + '_weightmap.nii.gz'))

    def _save_stats_output(self, dist_from_hyperplane_xval=None):
        """ Write stats output to csv file.

        Args:
            stats_output: a pandas file with prediction output

        Returns:
            predicter_stats_output.csv: Will output a csv file of stats output

        """

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.stats_output = pd.DataFrame({'Y': self.Y,
                                          'yfit_all': self.yfit_all})

        if hasattr(self, 'cv'):
            self.stats_output['yfit_xval'] = self.yfit_xval

        if hasattr(self, 'subject_id'):
            self.stats_output['subject_id'] = self.subject_id

        if self.prediction_type == 'classification':
            if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                self.stats_output['Probability'] = self.prob_xval
            else:
                if dist_from_hyperplane_xval is not None:
                    self.stats_output[
                        'dist_from_hyperplane_xval'] = dist_from_hyperplane_xval
                if self.algorithm == 'svm' and self.predictor.probability:
                    self.stats_output['Probability'] = self.prob_xval

        self.stats_output.to_csv(
            os.path.join(self.output_dir,
                         self.algorithm + '_Stats_Output.csv'),
            index=False)

    def _save_plot(self, predictor):
        """ Save Plots.

        Args:
            predictor: predictor instance

        Returns:
            predicter_weightmap_montage.png: Will output a montage of axial slices of weightmap
            predicter_prediction.png: Will output a plot of prediction

        """

        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if self.algorithm == 'lassopcr':
            coef = np.dot(self._pca.components_.T,self._lasso.coef_)
            coef_img = self.nifti_masker.inverse_transform(np.transpose(coef))
        elif self.algorithm == 'pcr':
            coef = np.dot(self._pca.components_.T,self._regress.coef_)
            coef_img = self.nifti_masker.inverse_transform(np.transpose(coef))
        else:
            coef_img = self.nifti_masker.inverse_transform(predictor.coef_)

        overlay_img = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz'))

        fig1 = plot_stat_map(coef_img, overlay_img, title=self.algorithm + " weights",
                            cut_coords=range(-40, 40, 10), display_mode='z')
        fig1.savefig(os.path.join(self.output_dir, self.algorithm + '_weightmap_axial.png'))

        if self.prediction_type == 'classification':
            if self.algorithm not in ['svm','ridgeClassifier','ridgeClassifierCV']:
                fig2 = probability_plot(self.stats_output)
                fig2.savefig(os.path.join(self.output_dir, self.algorithm + '_prob_plot.png'))
            else:
                fig2 = dist_from_hyperplane_plot(self.stats_output)
                fig2.savefig(os.path.join(self.output_dir, self.algorithm +
                            '_Distance_from_Hyperplane_xval.png'))
                if self.algorithm == 'svm' and self.predictor.probability:
                    fig3 = probability_plot(self.stats_output)
                    fig3.savefig(os.path.join(self.output_dir, self.algorithm + '_prob_plot.png'))

        elif self.prediction_type == 'prediction':
            fig2 = scatterplot(self.stats_output)
            fig2.savefig(os.path.join(self.output_dir, self.algorithm + '_scatterplot.png'))

def apply_mask(data=None, weight_map=None, mask=None, method='dot_product', save_output=False, output_dir='.'):
    """ Apply Nifti weight map to Nifti Images.

        Args:
            data: nibabel instance of data to be applied
            weight_map: nibabel instance of weight map
            mask: binary nibabel mask
            method: type of pattern expression (e.g,. 'dot_product','correlation')
            save_output: Boolean indicating whether or not to save output to csv file.
            output_dir: Directory to use for writing all outputs
            **kwargs: Additional parameters to pass

        Returns:
            pexp: Outputs a vector of pattern expression values

    """

    if mask is not None:
        if type(mask) is not nib.nifti1.Nifti1Image:
            raise ValueError("Mask is not a nibabel instance")
    else:
        mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'))

    if type(data) is not nib.nifti1.Nifti1Image:
        if type(data) is str:
            if os.path.isfile(data):
                data = nib.load(data)
        elif type(data) is list:
            data = nib.funcs.concat_images(data)
        else:
            raise ValueError("Data is not a nibabel instance, list of files, or a valid file name.")

    nifti_masker = NiftiMasker(mask_img=mask)
    data_masked = nifti_masker.fit_transform(data)
    if len(data_masked.shape) > 2:
        data_masked = data_masked.squeeze()

    if type(weight_map) is not nib.nifti1.Nifti1Image:
        if type(weight_map) is str:
            if os.path.isfile(weight_map):
                data = nib.load(weight_map)
        elif type(weight_map) is list:
            weight_map = nib.funcs.concat_images(weight_map)
        else:
            raise ValueError("Weight_map is not a nibabel instance, list of files, or a valid file name.")

    weight_map_masked = nifti_masker.fit_transform(weight_map)
    if len(weight_map_masked.shape) > 2:
        weight_map_masked = weight_map_masked.squeeze()

    # Calculate pattern expression
    pexp = pd.DataFrame()
    for w in range(0, weight_map_masked.shape[0]):
        if method == 'dot_product':
            pexp = pexp.append(pd.Series(np.dot(data_masked,np.transpose(weight_map_masked[w,:]))), ignore_index=True)
        elif method == 'correlation':
            pexp = pexp.append(pd.Series(pearson(data_masked,weight_map_masked[w,:])), ignore_index=True)
    pexp = pexp.T

    if save_output:
        pexp.to_csv(os.path.join(output_dir,"Pattern_Expression_" + method + ".csv"))
        # np.savetxt(os.path.join(output_dir,"Pattern_Expression_" + method + ".csv"), pexp, delimiter=",")

    return pexp


class Roc(object):

    """ Roc Class
    
    The Roc class is based on Tor Wager's Matlab roc_plot.m function and allows a user to easily run different types of 
    receiver operator characteristic curves.  For example, one might be interested in single interval or forced choice.

    Args:
        input_values: nibabel data instance
        binary_outcome: vector of training labels
        threshold_type: ['optimal_overall', 'optimal_balanced','minimum_sdt_bias']
        **kwargs: Additional keyword arguments to pass to the prediction algorithm

    """

    def __init__(self, input_values=None, binary_outcome=None, threshold_type='optimal_overall', forced_choice=False, **kwargs):
        if len(input_values) != len(binary_outcome):
            raise ValueError("Data Problem: input_value and binary_outcome are different lengths.")

        if not any(binary_outcome):
            raise ValueError("Data Problem: binary_outcome may not be boolean")

        thr_type = ['optimal_overall', 'optimal_balanced','minimum_sdt_bias']
        if threshold_type not in thr_type:
            raise ValueError("threshold_type must be ['optimal_overall', 'optimal_balanced','minimum_sdt_bias']")

        self.input_values = input_values
        self.binary_outcome = binary_outcome
        self.threshold_type = threshold_type
        self.forced_choice=forced_choice
        if isinstance(self.binary_outcome,pd.DataFrame):
            self.binary_outcome = np.array(self.binary_outcome).flatten()
        else:
            self.binary_outcome = binary_outcome

    def calculate(self, input_values=None, binary_outcome=None, criterion_values=None,
        threshold_type='optimal_overall', forced_choice=False, balanced_acc=False):
        """ Calculate Receiver Operating Characteristic plot (ROC) for single-interval
        classification.

        Args:
            input_values: nibabel data instance
            binary_outcome: vector of training labels
            criterion_values: (optional) criterion values for calculating fpr & tpr
            threshold_type: ['optimal_overall', 'optimal_balanced','minimum_sdt_bias']
            forced_choice: within-subject forced classification (bool).  Data must be
            stacked on top of each other (e.g., [1 1 1 0 0 0]).
            balanced_acc: balanced accuracy for single-interval classification (bool)
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        """

        if input_values is not None:
            self.input_values = input_values

        if binary_outcome is not None:
            self.binary_outcome = binary_outcome

        # Create Criterion Values
        if criterion_values is not None:
            self.criterion_values = criterion_values
        else:
            self.criterion_values = np.linspace(min(self.input_values), max(self.input_values), num=50*len(self.binary_outcome))

        if (forced_choice) | (self.forced_choice):
            self.forced_choice=True
            mn_scores = (self.input_values[self.binary_outcome] + self.input_values[self.binary_outcome])/2
            self.input_values[self.binary_outcome] = self.input_values[self.binary_outcome] - mn_scores;
            self.input_values[~self.binary_outcome] = self.input_values[~self.binary_outcome] - mn_scores;
            self.class_thr = 0;

        # Calculate true positive and false positive rate
        self.tpr = np.zeros(self.criterion_values.shape)
        self.fpr = np.zeros(self.criterion_values.shape)
        for i,x in enumerate(self.criterion_values):
            wh = self.input_values >= x
            self.tpr[i] = np.sum(wh[self.binary_outcome])/np.sum(self.binary_outcome)
            self.fpr[i] = np.sum(wh[~self.binary_outcome])/np.sum(~self.binary_outcome)
        self.n_true = np.sum(self.binary_outcome)
        self.n_false = np.sum(~self.binary_outcome)

        # Calculate Area Under the Curve

        # fix for AUC = 1 if no overlap - code not working (tpr_unique and fpr_unique can be different lengths)
        # fpr_unique = np.unique(self.fpr)
        # tpr_unique = np.unique(self.tpr)
        # if any((fpr_unique == 0) & (tpr_unique == 1)):
        #    self.auc = 1 # Fix for AUC = 1 if no overlap;
        # else:
        #    self.auc = auc(self.fpr, self.tpr) # Use sklearn auc otherwise
        self.auc = auc(self.fpr, self.tpr) # Use sklearn auc

        # Get criterion threshold
        if not self.forced_choice:
            self.threshold_type = threshold_type
            if threshold_type == 'optimal_balanced':
                mn = (tpr+fpr)/2
                self.class_thr = self.criterion_values[np.argmax(mn)]
            elif threshold_type == 'optimal_overall':
                n_corr_t = self.tpr*self.n_true
                n_corr_f = (1-self.fpr)*self.n_false
                sm = (n_corr_t+n_corr_f)
                self.class_thr = self.criterion_values[np.argmax(sm)]
            elif threshold_type == 'minimum_sdt_bias':
                # Calculate  MacMillan and Creelman 2005 Response Bias (c_bias)
                c_bias = ( norm.ppf(np.maximum(.0001, np.minimum(0.9999, self.tpr))) + norm.ppf(np.maximum(.0001, np.minimum(0.9999, self.fpr))) ) / float(2)
                self.class_thr = self.criterion_values[np.argmin(abs(c_bias))]

        # Calculate output
        self.false_positive = (self.input_values >= self.class_thr) & (~self.binary_outcome)
        self.false_negative = (self.input_values < self.class_thr) & (self.binary_outcome)
        self.misclass = (self.false_negative) | (self.false_positive)
        self.true_positive = (self.binary_outcome) & (~self.misclass)
        self.true_negative = (~self.binary_outcome) & (~self.misclass)
        self.sensitivity = np.sum(self.input_values[self.binary_outcome] >= self.class_thr)/self.n_true
        self.specificity = 1 - np.sum(self.input_values[~self.binary_outcome] >= self.class_thr)/self.n_false
        self.ppv = np.sum(self.true_positive)/(np.sum(self.true_positive) + np.sum(self.false_positive))
        if self.forced_choice:
            self.true_positive = self.true_positive[self.binary_outcome]
            self.true_negative = self.true_negative[~self.binary_outcome]
            self.false_negative = self.false_negative[self.binary_outcome]
            self.false_positive = self.false_positive[~self.binary_outcome]
            self.misclass = (self.false_positive) | (self.false_negative)

        # Calculate Accuracy
        if balanced_acc:
            self.accuracy = np.mean([self.sensitivity,self.specificity]) #See Brodersen, Ong, Stephan, Buhmann (2010)
        else:
            self.accuracy = 1 - np.mean(self.misclass)

        # Calculate p-Value using binomial test (can add hierarchical version of binomial test)
        self.n = len(self.misclass)
        self.accuracy_p = binom_test(int(np.sum(~self.misclass)), self.n, p=.5)
        self.accuracy_se = np.sqrt(np.mean(~self.misclass) * (np.mean(~self.misclass)) / self.n)


    def plot(self, plot_method = 'gaussian'):
        """ Create ROC Plot

        Create a specific kind of ROC curve plot, based on input values
        along a continuous distribution and a binary outcome variable (logical).

        Args:
            plot_method: type of plot ['gaussian','observed']
            binary_outcome: vector of training labels
            **kwargs: Additional keyword arguments to pass to the prediction algorithm

        Returns:
            fig
            
        """

        self.calculate() # Calculate ROC parameters

        if plot_method == 'gaussian':
            if self.forced_choice:
                diff_scores = self.input_values[self.binary_outcome] - self.input_values[~self.binary_outcome]
                mn_diff = np.mean(diff_scores)
                d = mn_diff / np.std(diff_scores)
                pooled_sd = np.std(diff_scores) / np.sqrt(2);
                d_a_model = mn_diff / pooled_sd

                x = np.arange(-3,3,.1)
                tpr_smooth = 1 - norm.cdf(x, d, 1)
                fpr_smooth = 1 - norm.cdf(x, -d, 1)
            else:
                mn_true = np.mean(self.input_values[self.binary_outcome])
                mn_false = np.mean(self.input_values[~self.binary_outcome])
                var_true = np.var(self.input_values[self.binary_outcome])
                var_false = np.var(self.input_values[~self.binary_outcome])
                pooled_sd = np.sqrt((var_true*(self.n_true-1))/(self.n_true + self.n_false - 2))
                d = (mn_true-mn_false)/pooled_sd
                z_true = mn_true/pooled_sd
                z_false = mn_false/pooled_sd

                x = np.arange(z_false-3,z_true+3,.1)
                tpr_smooth = 1-(norm.cdf(x, z_true,1))
                fpr_smooth = 1-(norm.cdf(x, z_false,1))

            fig = roc_plot(fpr_smooth,tpr_smooth)

        elif plot_method == 'observed':
            fig = roc_plot(self.fpr, self.tpr)
        else:
            raise ValueError("plot_method must be 'gaussian' or 'observed'")
        return fig

    def summary(self):
        """ Display a formatted summary of ROC analysis.

        """

        print("------------------------")
        print(".:ROC Analysis Summary:.")
        print("------------------------")
        print("{:20s}".format("Accuracy:") + "{:.2f}".format(self.accuracy))
        print("{:20s}".format("Accuracy SE:") + "{:.2f}".format(self.accuracy_se))
        print("{:20s}".format("Accuracy p-value:") + "{:.2f}".format(self.accuracy_p))
        print("{:20s}".format("Sensitivity:") + "{:.2f}".format(self.sensitivity))
        print("{:20s}".format("Specificity:") + "{:.2f}".format(self.specificity))
        print("{:20s}".format("AUC:") + "{:.2f}".format(self.auc))
        print("{:20s}".format("PPV:") + "{:.2f}".format(self.ppv))
        print("------------------------")



