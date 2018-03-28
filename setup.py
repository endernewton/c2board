#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'protobuf >= 0.3.2',
    'six',
]

setup(
    name='c2board'
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
