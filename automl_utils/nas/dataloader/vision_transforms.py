# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Miscellaneous transformations which can be applied to datasets."""

from typing import Tuple

import numpy as np
import torch


def get_std_shape(img: torch.Tensor, channel_first: bool) -> Tuple[int, int, int]:
    """Returns the channel count and size of the two spatial dimensions in a consistent order.

    Parameters
    ----------
    img: np.ndarray
        The image whose shape will be returned
    channel_first: bool
        Whether the first dimension is a channel (True) or spatial (False) dimension

    Returns
    -------
    (int, int, int)
        The number of channels, first spatial dimension (H), and second spatial dimension (W)
    """
    if channel_first:
        return img.shape[0], img.shape[1], img.shape[2]
    else:
        return img.shape[2], img.shape[0], img.shape[1]


class Cutout:
    """Applies Cutout to the supplied tensor."""

    def __init__(self, length: int, num_cuts: int = 1, cut_val: float = 0.0, channel_first: bool = True):
        """Creates a cutout transform.

        Parameters
        ----------
        length: int
            The length of the square to
        num_cuts
        cut_val
        channel_first
        """
        self._length = length
        self._num_cuts = num_cuts
        self._channel_first = channel_first
        self._cut_val = cut_val

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Applies cutout to the given image."""
        _, height, width = get_std_shape(img, self._channel_first)
        mask = np.ones((height, width), np.float32)

        for _ in range(self._num_cuts):
            y = np.random.randint(height)
            x = np.random.randint(width)

            x1 = np.clip(x - self._length // 2, 0, width)
            x2 = np.clip(x + self._length // 2, 0, width)
            y1 = np.clip(y - self._length // 2, 0, height)
            y2 = np.clip(y + self._length // 2, 0, height)

            mask[y1:y2, x1:x2] = self._cut_val

        mask = mask[None, :, :] if self._channel_first else mask[:, :, None]
        return img * torch.from_numpy(mask)
