# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import tempfile

import pytest
from torchvision.datasets import CIFAR10


@pytest.fixture(scope="session")
def cifar10_path():
    cifar_path = os.environ.get("AUTOML_UTILS_CIFAR10_PATH", None)
    if cifar_path:
        yield cifar_path
    else:
        with tempfile.TemporaryDirectory() as dirname:
            # ensure we have downloaded cifar10
            _ = CIFAR10(dirname, download=True)
            yield dirname
