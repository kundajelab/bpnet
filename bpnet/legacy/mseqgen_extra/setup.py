#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="mseqgen",
    version="1.0.0",
    description=("Multitask batch generation for training deeplearning"
                 "models on CHIP-seq, CHIP-exo, CHIP-nexus, ATAC-seq, "
                 "RNA-seq (or any other genomics assays that use "
                 "highthroughput sequencing)"),
    author="Zahoor Zafrulla",
    author_email="zahoor@stanford.edu",
    url="https://github.com/kundajelab/mseqgen",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "deeptools", "pyfaidx"],
    extras_require={"dev": ["pytest", "pytest-cov", "pycodestyle"]},
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    include_package_data=True,
    tests_require=["pytest", "pytest-cov", "pycodestyle"],
)
