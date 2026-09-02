# NLTools

[![PyPI version](https://img.shields.io/pypi/v/nltools.svg)](https://pypi.org/project/nltools/) [![CI](https://github.com/cosanlab/nltools/actions/workflows/ci.yml/badge.svg)](https://github.com/cosanlab/nltools/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2229813.svg)](https://zenodo.org/records/2229813) ![Python Versions](https://img.shields.io/badge/python-3.11%2B-blue) ![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20osx-blue)

[NLTools](https://github.com/cosanlab/nltools) is a Python package for analyzing neuroimaging data. It is the analysis engine powering [neuro-learn](http://neuro-learn.org). There are tools to perform data manipulation and analyses such as univariate GLMs, predictive multivariate modeling, and representational similarity analyses. It is based loosely off of Tor Wager's [object-oriented Matlab toolbox](https://github.com/canlab/CanlabCore) and leverages much code from [nilearn](http://nilearn.github.io/) and [scikit-learn](http://scikit-learn.org).

You can install it using `uv` or `pip`:

```bash
uv add nltools
```

```bash
pip install nltools
```

Watch a video in which [Dr. Eshin Jolly, PhD](https://sciminds.studio) outlines some of the design principles behind nltools at SciPy 2020:

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
    <iframe src="https://www.youtube.com/embed/1c1AnXLs7xM" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

Learn how to use nltools with the [tutorials](tutorials/index.md) on this site, or through these full courses:

- [DartBrains](https://dartbrains.org/) is an introductory neuroimaging analysis course that uses nltools.
- [Naturalistic-Data](http://naturalistic-data.org/) is a course covering advanced methods for analyzing naturalistic data. Many of its tutorials use nltools.

For a task-by-task map of which call to reach for and what will bite you, see the [User Guide](guide/index.md).

The reference is organized by what you want to do: the four [data classes](api/data/brain_data.md) (`BrainData`, `Adjacency`, `DesignMatrix`, `BrainCollection`), the public functions grouped [by task](api/tasks/loading.md), and an [A–Z index](api/algorithms.md) of `nltools.algorithms`. Upgrading from v0.5? Start with [Migrating from v0.5](migration-guide.md).
