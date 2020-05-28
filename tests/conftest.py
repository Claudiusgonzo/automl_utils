# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile

import pytest
from torchvision.datasets import CIFAR10


@pytest.fixture(scope="session")
def cifar_path():
    with tempfile.TemporaryDirectory() as dirname:
        # ensure we have downloaded cifar10
        _ = CIFAR10(dirname, download=True)
        yield dirname
