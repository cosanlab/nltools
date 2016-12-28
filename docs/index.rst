NLTools
=======

.. image:: https://api.travis-ci.org/ljchang/nltools.png 
    :target: https://travis-ci.org/ljchang/nltools/

.. image:: https://coveralls.io/repos/github/ljchang/nltools/badge.svg?branch=master
    :target: https://coveralls.io/github/ljchang/nltools?branch=master

.. image:: https://readthedocs.org/projects/neurolearn/badge/?version=latest
    :target: http://neurolearn.readthedocs.io/en/latest/?badge=latest

`NLTools <https://github.com/ljchang/neurolearn>`_ is a Python package for analyzing neuroimaging data.  It is the analysis engine powering `neuro-learn <http://neuro-learn.org>`_ There are tools to perform data manipulation and analyses such as univariate GLMs, predictive multivariate modeling, and representational similarity analyses.  It is based loosely off of Tor Wager's `object-oriented Matlab toolbox <https://github.com/canlab/CanlabCore>`_ and leverages much code from `nilearn <http://nilearn.github.io/>`_ and  `scikit-learn <http://scikit-learn.org>`_

Installation
------------

1. Method 1 - Install from PyPi
  
.. code-block:: python

	pip install nltools

2. Method 2 - Install directly from github (Recommended)
  
.. code-block:: python

	pip install git+https://github.com/ljchang/neurolearn

3. Method 3 - Clone github repository

.. code-block:: python

	git clone https://github.com/ljchang/neurolearn
	python setup.py install

Dependencies
^^^^^^^^^^^^

nltools requires several dependencies.  All are available in pypi.  Can use *pip install 'package'*

 - importlib
 - nibabel>=2.0.1
 - scikit-learn>=0.17
 - nilearn>=0.2
 - pandas>=0.16
 - numpy>=1.9
 - seaborn>=0.7.0
 - matplotlib
 - scipy
 - six
 - pynv
 
Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

 - mne

Tutorials
---------

Please see our tutorials_ Gallery, which provide numerous examples for how to use the toolbox.  Here are a few examples.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Basic Data Operations">

.. only:: html

    .. figure:: /auto_examples/01_DataOperations/images/thumb/sphx_glr_plot_download_thumb.png

        :ref:`sphx_glr_auto_examples_01_DataOperations_plot_download.py`

.. raw:: html

    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Masking Examples">

.. only:: html

    .. figure:: /auto_examples/01_DataOperations/images/thumb/sphx_glr_plot_mask_thumb.png

        :ref:`sphx_glr_auto_examples_01_DataOperations_plot_mask.py`

.. raw:: html

    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Univariate Regression Analysis">

.. only:: html

    .. figure:: /auto_examples/02_Analysis/images/thumb/sphx_glr_plot_univariate_regression_thumb.png

        :ref:`sphx_glr_auto_examples_02_Analysis_plot_univariate_regression.py`

.. raw:: html

    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Multivariate Prediction Analysis">

.. only:: html

    .. figure:: /auto_examples/02_Analysis/images/thumb/sphx_glr_plot_multivariate_prediction_thumb.png

        :ref:`sphx_glr_auto_examples_02_Analysis_plot_multivariate_prediction.py`

.. raw:: html

    </div>



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Multivariate Similarity Analyses">

.. only:: html

    .. figure:: /auto_examples/02_Analysis/images/thumb/sphx_glr_plot_similarity_example_thumb.png

        :ref:`sphx_glr_auto_examples_02_Analysis_plot_similarity_example.py`

.. raw:: html

    </div>


.. raw:: html

    <div style='clear:both'></div>

.. toctree::
	:maxdepth: 1

	auto_examples/index.rst
	reference

.. _tutorials:
	auto_examples/index.html
