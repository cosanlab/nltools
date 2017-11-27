

.. _sphx_glr_auto_examples_02_Analysis_plot_multivariate_prediction.py:


Multivariate Prediction
=======================

Running MVPA style analyses using multivariate regression is even easier and faster
than univariate methods. All you need to do is specify the algorithm and
cross-validation parameters. Currently, we have several different linear algorithms
implemented from `scikit-learn <http://scikit-learn.org/stable/>`_.



Load Data
---------

First, let's load the pain data for this example.  We need to specify the
training levels.  We will grab the pain intensity variable from the data.X
field.



.. code-block:: python


    from nltools.datasets import fetch_pain

    data = fetch_pain()
    data.Y = data.X['PainLevel']







Prediction with Cross-Validation
--------------------------------

We can now predict the output variable is a dictionary of the most
useful output from the prediction analyses. The predict function runs
the prediction multiple times. One of the iterations uses all of the
data to calculate the 'weight_map'. The other iterations are to estimate
the cross-validated predictive accuracy.



.. code-block:: python


    stats = data.predict(algorithm='ridge',
                        cv_dict={'type': 'kfolds','n_folds': 5,'stratified':data.Y})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_001.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 3.15
    overall Correlation: 1.00
    overall CV Root Mean Squared Error: 16597188857864359936.00
    overall CV Correlation: -0.18


Display the available data in the output dictionary



.. code-block:: python


    stats.keys()







Plot the multivariate weight map



.. code-block:: python


    stats['weight_map'].plot()




.. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_003.png
    :align: center




Return the cross-validated predicted data



.. code-block:: python


    stats['yfit_xval']







Algorithms
----------

There are several types of linear algorithms implemented including:
Support Vector Machines (svr), Principal Components Analysis (pcr), and
penalized methods such as ridge and lasso.  These examples use 5-fold
cross-validation holding out the same subject in each fold.



.. code-block:: python


    subject_id = data.X['SubjectID']
    svr_stats = data.predict(algorithm='svr',
                            cv_dict={'type': 'kfolds','n_folds': 5,
                            'subject_id':subject_id}, **{'kernel':"linear"})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_004.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_005.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 0.10
    overall Correlation: 0.99
    overall CV Root Mean Squared Error: 0.88
    overall CV Correlation: 0.57


Lasso Regression



.. code-block:: python


    lasso_stats = data.predict(algorithm='lasso',
                            cv_dict={'type': 'kfolds','n_folds': 5,
                            'subject_id':subject_id}, **{'alpha':.1})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_006.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_007.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 418952087759615295488.00
    overall Correlation: 0.34
    overall CV Root Mean Squared Error: 590228642096761536512.00
    overall CV Correlation: -0.08


Principal Components Regression



.. code-block:: python

    pcr_stats = data.predict(algorithm='pcr',
                            cv_dict={'type': 'kfolds','n_folds': 5,
                            'subject_id':subject_id})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_008.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_009.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 0.00
    overall Correlation: 1.00
    overall CV Root Mean Squared Error: 1.35
    overall CV Correlation: 0.45


Principal Components Regression with Lasso



.. code-block:: python


    pcr_stats = data.predict(algorithm='lassopcr',
                            cv_dict={'type': 'kfolds','n_folds': 5,
                            'subject_id':subject_id})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_010.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_prediction_011.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 36893490133815500800.00
    overall Correlation: -0.14
    overall CV Root Mean Squared Error: 13254702038307116939583360488964096.00
    overall CV Correlation: 0.07


Cross-Validation Schemes
------------------------

There are several different ways to perform cross-validation.  The standard
approach is to use k-folds, where the data is equally divided into k subsets
and each fold serves as both training and test.
Often we want to hold out the same subjects in each fold.
This can be done by passing in a vector of unique subject IDs that
correspond to the images in the data frame.



.. code-block:: python


    subject_id = data.X['SubjectID']
    ridge_stats = data.predict(algorithm='ridge',
                            cv_dict={'type': 'kfolds','n_folds': 5,'subject_id':subject_id},
                            plot=False, **{'alpha':.1})





.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 36893488147419103232.00
    overall Correlation: nan
    overall CV Root Mean Squared Error: 15590344239783675904.00
    overall CV Correlation: 0.00


Sometimes we want to ensure that the training labels are balanced across
folds.  This can be done using the stratified k-folds method.



.. code-block:: python


    ridge_stats = data.predict(algorithm='ridge',
                            cv_dict={'type': 'kfolds','n_folds': 5, 'stratified':data.Y},
                            plot=False, **{'alpha':.1})





.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 36893488147419103232.00
    overall Correlation: nan
    overall CV Root Mean Squared Error: 23471969580059398144.00
    overall CV Correlation: 0.09


Leave One Subject Out Cross-Validaiton (LOSO) is when k=n subjects.
This can be performed by passing in a vector indicating subject id's of
each image and using the loso flag.



.. code-block:: python


    ridge_stats = data.predict(algorithm='ridge',
                            cv_dict={'type': 'loso','subject_id': subject_id},
                            plot=False, **{'alpha':.1})





.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 36893488147419103232.00
    overall Correlation: nan
    overall CV Root Mean Squared Error: 15590344239783675904.00
    overall CV Correlation: 0.00


There are also methods to estimate the shrinkage parameter for the
penalized methods using nested crossvalidation with the
ridgeCV and lassoCV algorithms.



.. code-block:: python


    import numpy as np

    ridgecv_stats = data.predict(algorithm='ridgeCV',
                            cv_dict={'type': 'kfolds','n_folds': 5, 'stratified':data.Y},
                            plot=False, **{'alphas':np.linspace(.1, 10, 5)})




.. rst-class:: sphx-glr-script-out

 Out::

    overall Root Mean Squared Error: 0.00
    overall Correlation: 1.00
    overall CV Root Mean Squared Error: 0.56
    overall CV Correlation: 0.74


**Total running time of the script:** ( 2 minutes  17.181 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_multivariate_prediction.py <plot_multivariate_prediction.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_multivariate_prediction.ipynb <plot_multivariate_prediction.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
