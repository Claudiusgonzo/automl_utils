# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy
import pytest
import torch

from automl_utils.nas import Phase
from automl_utils.nas.dataloader import darts, Split


def reset_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)


@pytest.mark.parametrize("phase", [Phase.SEARCH, Phase.SELECT])
@pytest.mark.parametrize("split", [Split.TRAIN, Split.VAL])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_cifar_imagenet_equivalence(cifar10_path, phase, split, num_workers):
    c10 = darts.CIFAR10()
    imagenet = darts.ImageNet()

    dl_c10 = c10.get_dataloader(
        phase, split, cifar10_path, False, num_workers=num_workers, pin_memory=True, shuffle=True
    )
    dl_imagenet = imagenet.get_dataloader(
        phase, split, cifar10_path, False, num_workers=num_workers, pin_memory=True, shuffle=True
    )

    batches = []
    for dl in [dl_c10, dl_imagenet]:
        reset_seed()
        cur_batches = []
        for i, batch in enumerate(dl):
            cur_batches.append(batch)
            if i >= 4:
                break
        batches.append(cur_batches)

    for batch1, batch2 in zip(*batches):
        assert (batch1[0] == batch2[0]).all()
        assert (batch1[1] == batch2[1]).all()


@pytest.mark.parametrize("split", [Split.TRAIN, Split.VAL])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_smoke_c10_eval(cifar10_path, split, num_workers):
    c10 = darts.CIFAR10()
    dl = c10.get_dataloader(Phase.EVAL, split, cifar10_path, False, num_workers=num_workers, shuffle=True)

    for i, b in enumerate(dl):
        if i > 4:
            break
