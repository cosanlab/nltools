import os
import sys

try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

__version__ = '0.2.2'

setup(
    name='nltools',
    version='0.2.2',
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    packages=['nltools'],
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='A Python package to analyze neuroimaging data',
    long_description='nltools is a collection of python tools to perform preprocessing, univariate GLMs, and predictive multivariate modeling of neuroimaging data. It is the analysis engine powering www.neuro-learn.org.',
    url='http://neurolearn.readthedocs.org/en/latest/',
    keywords = ['neuroimaging', 'preprocessing', 'analysis','machine-learning'],
    classifiers = [
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        ]
)

