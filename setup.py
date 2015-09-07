import os
import sys

try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

__version__ = '0.2'

setup(
    name='nltools',
    version='0.2',
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    packages=['nltools'],
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='Neurolearn: a web-enabled imaging analysis toolbox',
)

  # url='http://github.com/ljchang/neurolearn',
    # download_url = 'https://github.com/ljchang/neurolearn/tarball/%s' % __version__,
