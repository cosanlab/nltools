:orphan:

Installation
------------

1. Method 1 - Install from PyPi

.. code-block:: python

	pip install nltools

2. Method 2 - Install directly from github (Most up to date)

.. code-block:: python

	pip install git+https://github.com/ljchang/neurolearn

3. Method 3 - Clone github repository

.. code-block:: python

	git clone https://github.com/ljchang/neurolearn
	python setup.py install

Dependencies
^^^^^^^^^^^^

nltools requires several dependencies.  All are available in pypi.  Can use *pip install 'package'*

 - nibabel>=3.0.1
 - scikit-learn>=0.21.0
 - nilearn>=0.6.0
 - pandas>=0.20
 - numpy>=1.9
 - seaborn>=0.7.0
 - matplotlib>=2.2.0
 - scipy
 - six
 - pynv
 - joblib
 - deepdish>=0.3.6

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

 - mne
 - requests
 - ipywidgets
 - networkx
