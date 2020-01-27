[![Package versioning](https://img.shields.io/pypi/v/nltools.svg)](https://pypi.org/project/nltools/)
[![Build Status](https://api.travis-ci.org/cosanlab/nltools.png)](https://travis-ci.org/cosanlab/nltools/)
[![Coverage Status](https://coveralls.io/repos/github/cosanlab/nltools/badge.svg?branch=master)](https://coveralls.io/github/cosanlab/nltools?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/625677967a0749299f38c2bf8ee269c3)](https://www.codacy.com/app/ljchang/nltools?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ljchang/nltools&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/neurolearn/badge/?version=latest)](http://neurolearn.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2229813.svg)](https://doi.org/10.5281/zenodo.2229813)


# NLTools
Python toolbox for analyzing neuroimaging data. It is particularly useful for conducting multivariate analyses.  It is originally based on Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html). Nltools is compatible with Python 3.6+. Python 2.7 was only supported through 0.3.11. We will no longer be supporting Python2 starting with version 0.3.12. 

### Installation
1. Method 1

   ```
   pip install nltools
   ```

2. Method 2 (Recommended)

   ```
   pip install git+https://github.com/cosanlab/nltools
   ```

3. Method 3

   ```
   git clone https://github.com/cosanlab/nltools
   python setup.py install
   ```
   or
   ```
   pip install -e 'path_to_github_directory'
   ```

### Dependencies
nltools requires several dependencies.  All are available in pypi.  Can use `pip install 'package'`
 - nibabel>=2.0.1
 - scikit-learn>=0.19.1
 - nilearn>=0.4
 - pandas>=0.20
 - numpy>=1.9
 - seaborn>=0.7.0
 - matplotlib>=2.1
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
