# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task-specific sets of operations should be derived from `OpSpace` with entries whose values are `OpContainer`s."""

from __future__ import annotations

import enum
from typing import Any, Callable

import torch.nn as nn


class OpContainer:
    """Container for a callable that produces an nn.Module (used as the value of an OpSpace entry).

    Note: This is needed as one cannot create an Enum whose values are functions.

    Attributes:
    -----------
    function: Callable[..., nn.Module]
        A callable which creates a new instance of the specified operation

    """

    def __init__(self, function: Callable[..., nn.Module]):
        """Creates a new operation type.

        Parameters
        ----------
        function: Callable[..., nn.Module]
            A callable which creates a new instance of the specified operation

        """
        self.function = function

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module:
        """Creates an instance of the operation.

        Parameters
        ----------
        args: list
            The non-keyword args
        kwargs: dict
            The keyword args

        Returns
        -------
        nn.Module
            The instantiated operation

        """
        return self.function(*args, **kwargs)


class OpSpace(enum.Enum):
    """The base class from which all spaces of cell operations are derived.

    Examples:
        >>> class MyOpSpace(OpSpace):
        ...     CONV2D = OpContainer(lambda *args, **kwargs: nn.Conv2d(*args, **kwargs))
        ...     MAXPOOL3 = OpContainer(lambda *args, **kwargs: nn.MaxPool2d(3))
        ...     AVGPOOL3 = OpContainer(lambda *args, **kwargs: nn.AvgPool2d(3))

    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Creates a new operation in the OpSpace and sets its index.

        Parameters
        ----------
        args: list
            The non-keywords args
        kwargs: dict
            The keywords args
        """
        # try calling the parent constructor if necessary
        try:
            super().__init__(*args, **kwargs)  # type: ignore
        except TypeError:
            # no parent
            pass

        # set the order in the `OpSpace` (i.e., self._index = N-1 for the Nth operation in the `OpSpace`)
        self._index = len(self.__class__.__members__)

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module:
        """Creates the operation by invoking the `OpContainer`.

        Parameters
        ----------
        args: list
            The non-keyword args passed to the `OpContainer` __init__ method
        kwargs: dict
            The keyword args passed to the `OpContainer` __init__ method

        Returns
        -------
        nn.Module
            The instantiated operation

        """
        return self.value(*args, **kwargs)

    @classmethod
    def get_by_index(cls, idx: int) -> OpSpace:
        """Gets the enum entry by index.

        Parameters
        ----------
        idx: int
            The index of the desired operation

        Returns
        -------
        OpSpace
            The enum entry containing the desired `OpContainer`

        """
        return list(cls)[idx]

    @property
    def index(self) -> int:
        """Gets the index of a given entry in the `OpSpace`.

        Returns
        -------
        int
            The entry's index

        """
        return self._index
