# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A simple enumeration of NAS stages."""

import enum


class Phase(enum.Enum):
    """An enum signaling the stages of model training.

    Valid options include:
        SEARCH - a process which produces one of more candidate architectures
        SELECT - a process which selects the desired architecture for evaluation
        EVAL - a process which produces the final trained model

    Note: Not all NAS algorithms will implement all three phases.
    """

    SEARCH: str = "search"
    SELECT: str = "select"
    EVAL: str = "eval"
