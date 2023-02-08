#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from os.path import join, dirname
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, 'dynashap', '__version__.py'), 
          'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()


def get_file_contents(filename):
    with open(join(dirname(__file__), filename)) as fp:
        return fp.read()


def get_install_requires():
    requirements = get_file_contents('requirements-dev.txt')
    install_requires = []
    for line in requirements.split('\n'):
        line = line.strip()
        if line and not line.startswith('-'):
            install_requires.append(line)
    return install_requires


setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    install_requires=get_install_requires(),
    packages=find_packages(exclude=['tests']),
    package_data={'': ['LICENSE', 'NOTICE']},
    include_package_data=True,
    license=about['__license__']
)