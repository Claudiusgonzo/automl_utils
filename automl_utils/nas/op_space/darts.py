# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The op space(s) used in DARTS."""

import torch.nn as nn

from automl_utils.nas.op_space.op_space import OpContainer, OpSpace
from automl_utils.nas.op_space.vision_ops import DepthwiseSeparableConv2d, FactorizedReduce, Zero


class ImageOps(OpSpace):
    """The op space used by DARTS for CIFAR10 and ImageNet."""

    NONE = OpContainer(lambda c, stride, aff: Zero(stride))
    SKIP_CONNECT = OpContainer(lambda c, stride, aff: nn.Identity() if stride == 1 else _factorized_reduce(c, aff))

    # Pooling Operations
    MAX_POOL = OpContainer(lambda c, stride, aff: nn.MaxPool2d(3, stride, 1))
    AVG_POOL = OpContainer(lambda c, stride, aff: nn.AvgPool2d(3, stride, 1, count_include_pad=False))

    # Vanillla Depthwise Separable Convolutions (repeated twice)
    SEP_CONV_3 = OpContainer(
        lambda c, stride, aff: nn.Sequential(
            _add_relu_bn(DepthwiseSeparableConv2d(c, c, 3, stride, 1), c, aff),
            _add_relu_bn(DepthwiseSeparableConv2d(c, c, 3, 1, 1), c, aff),
        )
    )
    SEP_CONV_5 = OpContainer(
        lambda c, stride, aff: nn.Sequential(
            _add_relu_bn(DepthwiseSeparableConv2d(c, c, 5, stride, 2), c, aff),
            _add_relu_bn(DepthwiseSeparableConv2d(c, c, 5, 1, 2), c, aff),
        )
    )

    # Dilated Depthwise Separable Convolutions
    DIL_SEP_CONV_3 = OpContainer(
        lambda c, stride, aff: _add_relu_bn(DepthwiseSeparableConv2d(c, c, 3, stride, 2, 2), c, aff)
    )
    DIL_SEP_CONV_5 = OpContainer(
        lambda c, stride, aff: _add_relu_bn(DepthwiseSeparableConv2d(c, c, 5, stride, 4, 2), c, aff)
    )


def _add_relu_bn(op: nn.Module, channel: int, affine: bool) -> nn.Module:
    return nn.Sequential(nn.ReLU(), op, nn.BatchNorm2d(channel, affine=affine))


def _factorized_reduce(channel: int, affine: bool) -> nn.Module:
    return _add_relu_bn(FactorizedReduce(channel, channel), channel, affine)
