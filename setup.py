#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://hyperpoints.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='hyperpoints',
    version='0.0.1',
    description='Distributed hyperparameter optimization and reinforcement learning algorithms.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Todd Young',
    author_email='youngmt1@ornl.gov',
    url='https://github.com/yngtodd/hyperpoints',
    packages=[
        'hyperpoints',
    ],
    package_dir={'hyperpoints': 'hyperpoints'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='hyperpoints',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
