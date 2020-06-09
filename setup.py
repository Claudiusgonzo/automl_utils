# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Installs our package."""

from setuptools import setup, find_packages
from os import path


with open(path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    setup(
        name="automl-utils",
        version="0.1.0a1",
        description="Facilitating reproducible AutoML research.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Neil Tenenholtz",
        author_email="neil.tenenholtz@microsoft.com",
        url="https://github.com/microsoft/automl_utils",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Build Tools",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        keywords="PyTorch AutoML NAS reproducibility",
        packages=find_packages(exclude=["tests"]),
        python_requires="~=3.7",
        install_requires=["torch ~= 1.3", "torchvision ~= 0.5"],
        extras_require={
            "dev": [
                "black",
                "flake8",
                "flake8-blind-except",
                "flake8-bugbear",
                "flake8-builtins",
                "flake8-docstrings",
                "flake8-import-order",
                "flake8-mutable",
                "mypy",
                "pre-commit",
                "pytest",
                "pytest-cov",
                "pytest-xdist",
            ],
            "tensorboard": ["tensorflow >= 1.0"],
        },
        package_data={"automl_utils": ["py.typed"]},
        zip_safe=False,
    )
