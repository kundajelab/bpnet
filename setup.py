#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "argh",
    "attr",
    "related",
    "cloudpickle",

    "concise",
    "deepexplain",

    # ml
    "gin-config",
    "keras>=2.2.4",
    "scikit-learn",
    # "tensorflow",

    # numerics
    "h5py",
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",

    # Plotting
    "matplotlib>=3.0.2",
    "plotnine",
    "seaborn",

    # genomics
    "pybigwig",
    "pybedtools",  # remove?
    "modisco>=0.5.1.2",
    # "pyranges",

    "joblib",
    "cloudpickle>=1.0.0",  # - remove?
    "kipoi",
    "kipoi-utils>=0.3.0",
    "kipoiseq",

    "papermill",
    "jupyter_client",
    "ipykernel",
    "nbconvert",
    "vdom>=0.6",

    # utils
    "ipython",
    "tqdm",
    "pprint",

    # Remove
    "genomelake",
    "pysam",  # replace with pyfaidx
]

optional = [
    "comet_ml",
    "wandb",
    "fastparquet",
    "python-snappy",
    "ipywidgets",  # for motif simulation
]


# optional2 = [
#     "pygam",
#     "pytorch",
# ]

test_requirements = [
    "pytest",
    "virtualenv",
]

dependency_links = [
    "deepexplain @ git+https://github.com/kundajelab/DeepExplain.git@#egg=deepexplain-0.1"
]


setup(
    name="bpnet",
    version='0.0.15',
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Ziga Avsec",
    author_email="avsec@in.tum.de",
    url="https://github.com/kundajelab/bpnet",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
        "extras": optional,
    },
    license="MIT license",
    entry_points={'console_scripts': ['bpnet = bpnet.__main__:main']},
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    package_data={'bpnet': ['logging.conf']},
    include_package_data=True,
    tests_require=test_requirements
)
