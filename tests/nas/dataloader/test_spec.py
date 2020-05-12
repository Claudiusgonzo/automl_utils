# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from automl_utils.nas.dataloader.spec import BatchConfig


def test_bad_batch_config():
    with pytest.raises(ValueError):
        BatchConfig(batch_size=0, input_transform=None, target_transform=None)
