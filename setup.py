try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

setup(
    name='nltools',
    version='0.1',
    author='Luke Chang',
    author_email='luke.j.chang@dartmouth.edu',
    url='http://github.com/ljchang/neurolearn',
    download_url = 'https://github.com/ljchang/neurolearn/tarball/%s' % __version__,
    packages=['nltools'],
    package_data={'nltools': ['resources/*']},
    license='LICENSE.txt',
    description='Neurolearn: a web-enabled imaging analysis toolbox',
)