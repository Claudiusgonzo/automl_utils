# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A set of operations useful for implemented vision-based op spaces."""

import torch
from torch import nn as nn


class Zero(nn.Module):
    """Zeros the input at a desired stride."""

    def __init__(self, stride: int):
        """Zeros the input at a desired stride.

        Parameters
        ----------
        stride: int
            The stride along which the input should be zeroed (stride=1 results an all-0 tensor)
        """
        super().__init__()
        self._stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the zeroed input."""
        template = x if self._stride == 1 else x[:, :, :: self._stride, :: self._stride]
        return torch.zeros_like(template)


class FactorizedReduce(nn.Module):
    """A spatial reduction operation that concats two strided, offset, pointwise convs along the channel dim."""

    def __init__(self, c_in: int, c_out: int):
        """Creates the FactorizedReduce operation.

        Parameters
        ----------
        c_in: int
            The number of input channels
        c_out: int
            The number of output channels
        affine: bool
            Whether the final batchnorm should include affine parameters
        """
        super().__init__()
        if c_out % 2 != 0:
            raise ValueError("Number of output channels must be even.")
        self._conv1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self._conv2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the FactorizedReduce."""
        if x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0:
            raise ValueError("Spatial dimensions must be even.")
        return torch.cat([self._conv1(x), self._conv2(x[:, :, 1:, 1:])], dim=1)


class DepthwiseSeparableConv2d(nn.Module):
    """A depthwise separable 2d convolution."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int, padding: int, dilation: int = 1):
        """Creates a depthwise separable 2d convolution.

        Parameters
        ----------
        c_in: int
            The number of input channels
        c_out: int
            The number of output channels
        kernel: int
            The kernel size of the spatial conv
        stride: int
            The stride of the spatial conv
        padding: int
            The padding of the spatial conv
        dilation: int, optional
            The dilation of the spatial conv. Defaults to 1.
        """
        super().__init__()
        self._op = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel, stride, padding, dilation=dilation, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the depthwise separable conv."""
        return self._op(x)
