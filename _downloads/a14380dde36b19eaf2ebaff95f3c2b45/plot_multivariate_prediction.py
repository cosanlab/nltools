"""
Multivariate Prediction
=======================

Running MVPA style analyses using multivariate regression is even easier and faster
than univariate methods. All you need to do is specify the algorithm and
cross-validation parameters. Currently, we have several different linear algorithms
implemented from `scikit-learn <http://scikit-learn.org/stable/>`_.

"""

#########################################################################
# Load Data
# ---------
#
# First, let's load the pain data for this example.  We need to specify the
# training levels.  We will grab the pain intensity variable from the data.X
# field.

from nltools.datasets import fetch_pain

data = fetch_pain()
data.Y = data.X['PainLevel']

#########################################################################
# Prediction with Cross-Validation
# --------------------------------
#
# We can now predict the output variable is a dictionary of the most
# useful output from the prediction analyses. The predict function runs
# the prediction multiple times. One of the iterations uses all of the
# data to calculate the 'weight_map'. The other iterations are to estimate
# the cross-validated predictive accuracy.

stats = data.predict(algorithm='ridge',
                    cv_dict={'type': 'kfolds','n_folds': 5,'stratified':data.Y})

#########################################################################
# Display the available data in the output dictionary

stats.keys()

#########################################################################
# Plot the multivariate weight map

stats['weight_map'].plot()

#########################################################################
# Return the cross-validated predicted data

stats['yfit_xval']

#########################################################################
# Algorithms
# ----------
#
# There are several types of linear algorithms implemented including:
# Support Vector Machines (svr), Principal Components Analysis (pcr), and
# penalized methods such as ridge and lasso.  These examples use 5-fold
# cross-validation holding out the same subject in each fold.

subject_id = data.X['SubjectID']
svr_stats = data.predict(algorithm='svr',
                        cv_dict={'type': 'kfolds','n_folds': 5,
                        'subject_id':subject_id}, **{'kernel':"linear"})

#########################################################################
# Lasso Regression

lasso_stats = data.predict(algorithm='lasso',
                        cv_dict={'type': 'kfolds','n_folds': 5,
                        'subject_id':subject_id}, **{'alpha':.1})

#########################################################################
# Principal Components Regression
pcr_stats = data.predict(algorithm='pcr',
                        cv_dict={'type': 'kfolds','n_folds': 5,
                        'subject_id':subject_id})

#########################################################################
# Principal Components Regression with Lasso

pcr_stats = data.predict(algorithm='lassopcr',
                        cv_dict={'type': 'kfolds','n_folds': 5,
                        'subject_id':subject_id})

#########################################################################
# Cross-Validation Schemes
# ------------------------
#
# There are several different ways to perform cross-validation.  The standard
# approach is to use k-folds, where the data is equally divided into k subsets
# and each fold serves as both training and test.
# Often we want to hold out the same subjects in each fold.
# This can be done by passing in a vector of unique subject IDs that
# correspond to the images in the data frame.

subject_id = data.X['SubjectID']
ridge_stats = data.predict(algorithm='ridge',
                        cv_dict={'type': 'kfolds','n_folds': 5,'subject_id':subject_id},
                        plot=False, **{'alpha':.1})

#########################################################################
# Sometimes we want to ensure that the training labels are balanced across
# folds.  This can be done using the stratified k-folds method.

ridge_stats = data.predict(algorithm='ridge',
                        cv_dict={'type': 'kfolds','n_folds': 5, 'stratified':data.Y},
                        plot=False, **{'alpha':.1})

#########################################################################
# Leave One Subject Out Cross-Validaiton (LOSO) is when k=n subjects.
# This can be performed by passing in a vector indicating subject id's of
# each image and using the loso flag.

ridge_stats = data.predict(algorithm='ridge',
                        cv_dict={'type': 'loso','subject_id': subject_id},
                        plot=False, **{'alpha':.1})

#########################################################################
# There are also methods to estimate the shrinkage parameter for the
# penalized methods using nested crossvalidation with the
# ridgeCV and lassoCV algorithms.

import numpy as np

ridgecv_stats = data.predict(algorithm='ridgeCV',
                        cv_dict={'type': 'kfolds','n_folds': 5, 'stratified':data.Y},
                        plot=False, **{'alphas':np.linspace(.1, 10, 5)})
