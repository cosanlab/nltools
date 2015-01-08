try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='nltools',
    version='0.1',
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    packages=['nltools'],
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='Neurolearn: a web-enabled imaging analysis toolbox',
)