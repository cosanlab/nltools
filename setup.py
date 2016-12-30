# from nltools.version import __version__
from setuptools import setup, find_packages

__version__ = '0.2.6'

# try:
#     from setuptools.core import setup
# except ImportError:
#     from distutils.core import setup

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='nltools',
    version=__version__,
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    url='http://neurolearn.readthedocs.org/en/latest/',
    install_requires=['numpy', 'scipy', 'nilearn', 'nibabel','pandas', 'six', 'importlib',
                      'seaborn', 'matplotlib', 'scikit-learn','pynv'],
    packages=find_packages(exclude=['nltools/tests']),
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='A Python package to analyze neuroimaging data',
    long_description='nltools is a collection of python tools to perform preprocessing, univariate GLMs, and predictive multivariate modeling of neuroimaging data. It is the analysis engine powering www.neuro-learn.org.',
    keywords = ['neuroimaging', 'preprocessing', 'analysis','machine-learning'],
    classifiers = [
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)

