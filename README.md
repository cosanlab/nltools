[![Package versioning](https://img.shields.io/pypi/v/nltools.svg)](https://pypi.org/project/nltools/)
[![Tests and Coverage](https://github.com/cosanlab/nltools/actions/workflows/tests_and_coverage.yml/badge.svg)](https://github.com/cosanlab/nltools/actions/workflows/tests_and_coverage.yml)
[![Deploy Docs and PyPI](https://github.com/cosanlab/nltools/actions/workflows/deploy_docs_pypi_onrelease.yml/badge.svg)](https://nltools.org)
[![codecov](https://codecov.io/gh/cosanlab/nltools/branch/master/graph/badge.svg)](https://codecov.io/gh/cosanlab/nltools)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/625677967a0749299f38c2bf8ee269c3)](https://www.codacy.com/app/ljchang/nltools?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ljchang/nltools&amp;utm_campaign=Badge_Grade)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2229813.svg)](https://doi.org/10.5281/zenodo.2229813)
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx%20%7C%20win-blue)


# NLTools
Python toolbox for analyzing neuroimaging data. It is particularly useful for conducting multivariate analyses.  It is originally based on Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html). Nltools is only compatible with Python 3.7+. 

## Documentation

Documentation and tutorials are available at https://nltools.org

## Installation
1. Method 1 (stable)

   ```
   pip install nltools
   ```

2. Method 2 (bleeding edge)

   ```
   pip install git+https://github.com/cosanlab/nltools
   ```

3. Method 3 (for development)

   ```
   git clone https://github.com/cosanlab/nltools
   pip install -e nltools
   ```

## Preprocessing
Nltools has minimal routines for pre-processing data. For more complete pre-processing pipelines please see our [cosanlab_preproc](https://github.com/cosanlab/cosanlab_preproc) library built with `nipype`.
