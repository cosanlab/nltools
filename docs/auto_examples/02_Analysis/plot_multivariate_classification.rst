

.. _sphx_glr_auto_examples_02_Analysis_plot_multivariate_classification.py:

 
Multivariate Classification
===========================

This tutorial provides an example of how to run classification analyses.



Load & Prepare Data
-------------------

First, let's load the pain data for this example.  We need to create a data 
object with high and low pain intensities.  These labels need to be specified in the
dat.Y field as a pandas dataframe. We also need to create a vector of subject ids
so that subject images can be held out together in cross-validation.



.. code-block:: python


    from nltools.datasets import fetch_pain
    import numpy as np
    import pandas as pd

    data = fetch_pain()
    high = data[np.where(data.X['PainLevel']==3)[0]]
    low = data[np.where(data.X['PainLevel']==1)[0]]
    dat = high.append(low)
    dat.Y = pd.DataFrame(np.concatenate([np.ones(len(high)),np.zeros(len(low))]))
    subject_id = np.concatenate([high.X['SubjectID'].values,low.X['SubjectID'].values])







Classification with Cross-Validation
------------------------------------

We can now train a brain model to classify the different labels specified in dat.Y.
First, we will use a support vector machine with 5 fold cross-validation in which the 
same images from each subject are held out together.  
The predict function runs the classification multiple times. One of the 
iterations uses all of the data to calculate the 'weight_map'. The other iterations 
estimate the cross-validated predictive accuracy.



.. code-block:: python


    svm_stats = dat.predict(algorithm='svm', 
    						cv_dict={'type': 'kfolds','n_folds': 5, 'subject_id':subject_id},
    						**{'kernel':"linear"})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_001.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_002.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall accuracy: 1.00
    overall CV accuracy: 0.80


SVMs can be converted to predicted probabilities using Platt Scaling



.. code-block:: python


    platt_stats = dat.predict(algorithm='svm', 
    						cv_dict={'type': 'kfolds','n_folds': 5, 'subject_id':subject_id},
        					**{'kernel':'linear','probability':True})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_003.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_004.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall accuracy: 1.00
    overall CV accuracy: 0.80


Standard OLS Logistic Regression.  



.. code-block:: python

    logistic_stats = dat.predict(algorithm='logistic', 
        					cv_dict={'type': 'kfolds','n_folds': 5, 'subject_id':subject_id})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_005.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_006.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall accuracy: 1.00
    overall CV accuracy: 0.75


Ridge classification



.. code-block:: python

    ridge_stats = dat.predict(algorithm='ridgeClassifier', 
        cv_dict={'type': 'kfolds','n_folds': 5, 'subject_id':subject_id})




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_007.png
            :scale: 47

    *

      .. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_008.png
            :scale: 47


.. rst-class:: sphx-glr-script-out

 Out::

    overall accuracy: 1.00
    overall CV accuracy: 0.79


ROC Analyses
------------

We are often interested in evaluating how well a pattern can discriminate 
between different classes of data. However, accuracy could be high because
of a highly sensitive but not specific model.  Receiver operator characteristic
curves allow us to evaluate the sensitivity and specificity of the model.  
and evaluate how well it can discriminate between high and low pain using 
We use the Roc class to initialize an Roc object and the plot() and summary() 
methods to run the analyses. We could also just run the calculate() method 
to run the analysis without plotting.



.. code-block:: python


    from nltools.analysis import Roc

    roc = Roc(input_values=svm_stats['dist_from_hyperplane_xval'], 
    		binary_outcome=svm_stats['Y'].astype(bool))
    roc.plot()
    roc.summary()




.. image:: /auto_examples/02_Analysis/images/sphx_glr_plot_multivariate_classification_009.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    ------------------------
    .:ROC Analysis Summary:.
    ------------------------
    Accuracy:           0.82
    Accuracy SE:        0.11
    Accuracy p-value:   0.00
    Sensitivity:        0.75
    Specificity:        0.89
    AUC:                0.88
    PPV:                0.88
    ------------------------


The above example uses single-interval classification, which attempts to 
determine the optimal classification interval. However, sometimes we are 
intersted in directly comparing responses to two images within the same person. 
In this situation we should use forced-choice classification, which looks at 
the relative classification accuracy between two images.  You must pass a list 
indicating the ids of each unique subject.



.. code-block:: python


    roc_fc = Roc(input_values=svm_stats['dist_from_hyperplane_xval'], 
    			binary_outcome=svm_stats['Y'].astype(bool), forced_choice=subject_id)
    roc_fc.plot()
    roc_fc.summary()




.. code-block:: pytb

    Traceback (most recent call last):
      File "/Users/Esh/anaconda/lib/python2.7/site-packages/sphinx_gallery/gen_rst.py", line 475, in execute_code_block
        exec(code_block, example_globals)
      File "<string>", line 4, in <module>
      File "/Users/Esh/Documents/Python/Cosan/nltools/nltools/analysis.py", line 187, in plot
        if self.forced_choice:
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()




**Total running time of the script:** ( 1 minutes  6.437 seconds)



.. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_multivariate_classification.py <plot_multivariate_classification.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_multivariate_classification.ipynb <plot_multivariate_classification.ipynb>`

.. rst-class:: sphx-glr-signature

    `Generated by Sphinx-Gallery <http://sphinx-gallery.readthedocs.io>`_
