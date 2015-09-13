import imp
from setuptools import setup

version = imp.load_source('optimus.version', 'optimus/version.py')

long_description = \
    """A python package for describing and serializing feed-forward (acyclic)
signal processing graphs by decoupling topology and parameter from
implementation in JSON. Currently, the primary purpose of optimus is to extend
the functionality of Theano making it easier to build and reconstruct arbitrary
neural networks in a language-agnostic manner."""

setup(
    name='optimus',
    version=version.version,
    description='Python module for building and serializing neural networks.',
    author='Eric J. Humphrey',
    author_email='ejhumphrey@nyu.edu',
    url='http://github.com/ejhumphrey/optimus',
    download_url='http://github.com/ejhumphrey/optimus/releases',
    packages=['optimus'],
    package_data={},
    long_description=long_description,
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7"
    ],
    keywords='machine learning, neural network',
    license='ISC',
    install_requires=[
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'scikit-learn >= 0.14.0',
        'matplotlib',
        'theano >= 0.6.0'
    ]
)
