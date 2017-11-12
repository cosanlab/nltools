[![PyPI version](https://badge.fury.io/py/nltools.svg)](https://badge.fury.io/py/nltools)
[![Build Status](https://api.travis-ci.org/ljchang/nltools.png)](https://travis-ci.org/ljchang/nltools/)
[![Coverage Status](https://coveralls.io/repos/github/ljchang/nltools/badge.svg?branch=master)](https://coveralls.io/github/ljchang/nltools?branch=master)
[![Documentation Status](https://readthedocs.org/projects/neurolearn/badge/?version=latest)](http://neurolearn.readthedocs.io/en/latest/?badge=latest)

# NLTools
Python toolbox for analyzing neuroimaging data.  Compatible with both Python 2.7 and Python 3.6.  It is particularly useful for conducting multivariate analyses.  It is originally based on Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html)

### Installation
1. Method 1
  
   ```
   pip install nltools
   ```

2. Method 2 (Recommended)
  
   ```
   pip install git+https://github.com/ljchang/neurolearn
   ```

3. Method 3

   ```
   git clone https://github.com/ljchang/neurolearn
   python setup.py install
   ```

### Dependencies
nltools requires several dependencies.  All are available in pypi.  Can use `pip install 'package'`
 - nibabel>=2.0.1
 - scikit-learn>=0.19.1
 - nilearn>=0.2
 - pandas>=0.20
 - numpy>=1.9
 - seaborn>=0.7.0
 - matplotlib
 - scipy
 - six
 - pynv
 - joblib
 
### Optional Dependencies
 - mne
 - requests
 - networkx
 - ipywidgets >=5.2.2
 
### Documentation
Current Documentation can be found at [readthedocs](http://neurolearn.readthedocs.org/en/latest).  

Please see our [tutorials](http://neurolearn.readthedocs.io/en/latest/auto_examples/index.html), which provide numerous examples for how to use the toolbox.  

### Preprocessing
Please see our [cosanlab_preproc](https://github.com/cosanlab/cosanlab_preproc) library for nipype pipelines to perform preprocessing on neuroimaging data.
