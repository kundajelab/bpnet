#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="bpnet",
    version='0.4.0',
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Surag Nair, Vivekanandan Ramalingam, Zahoor Zafrulla",
    author_email="zahoor@stanford.edu",
    url="https://github.com/kundajelab/bpnet",
    packages=find_packages(exclude=["docs", "docs-build"]),
    install_requires=["tensorflow==2.4.1", 
                      "tensorflow-probability==0.12.2", 
                      "tqdm", "scikit-learn",
                      "scipy", "scikit-image", "scikit-learn", "deepdish", "pandas", "matplotlib", "plotly", 
                      "deeptools", "pyfaidx", "deeplift", "hdf5plugin"
                      ],
    extras_require={"dev": ["pytest", "pytest-cov"]},
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    include_package_data=True,
    tests_require=["pytest", "pytest-cov"],
    entry_points = {
        "console_scripts": [
            "bpnet-train = bpnet.cli.bpnettrainer:main",
            "bpnet-predict = bpnet.cli.predict:predict_main",
            "bpnet-shap = bpnet.cli.shap_scores:shap_scores_main",
            "bpnet-motif = bpnet.cli.motif_discovery:motif_discovery_main",
            "bpnet-counts-loss-weight = bpnet.cli.counts_loss_weight:counts_loss_weight_main",
            "bpnet-embeddings = bpnet.cli.embeddings:embeddings_main",
            "bpnet-outliers = bpnet.cli.outliers:outliers_main"
        ]
    }
)
