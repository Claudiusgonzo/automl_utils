# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tooling used to create reference implementations of dataloaders from published works."""

import abc
import dataclasses as dc
import enum
from typing import Any, Callable, Mapping, Optional, Tuple

import torch.utils.data as tud

from automl_utils.nas.phase import Phase


class Split(enum.Enum):
    """An enum signaling whether the dataset represents training or validation data.

    Valid options include TRAIN and VAL.
    """

    TRAIN: str = "train"
    VAL: str = "val"


@dc.dataclass
class BatchConfig:
    """How individual samples should be transformed and aggregated into a batch.

    Attributes:
        batch_size: int
            How many samples should be aggregated into a single batch.
        input_transform: Optional[Callable]
            How the inputs should be transformed. None corresponds to no transformation.
        target_transform: Optional[Callable]
            How the targets should be transformed. None corresponds to no transformation.
    """

    batch_size: int
    input_transform: Optional[Callable]
    target_transform: Optional[Callable]

    def __post_init__(self) -> None:
        """Validates that the batch_size is strictly positive."""
        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0. Received value of {self.batch_size}.")


class DataloaderSpec(abc.ABC):
    """The base class from which all reference implementation data strategies should be derived."""

    def __init__(
        self,
        train_splits: Optional[Mapping[Phase, Optional[float]]] = None,
        configs: Optional[Mapping[Tuple[Phase, Split], BatchConfig]] = None,
    ):
        """Creates a new `DataloaderSpec`.

        Parameters
        ----------
        train_splits: Optional[Mapping[Phase, float]], optional
            Overrides the reference implementation split of the training dataset for train/val in the various phases.
            Defaults to None, which signals the split from the reference implementation should be used.
        configs: Optional[Mapping[Tuple[Phase, Split], BatchConfig]], optional
            Overrides the reference implementation batch sizes and input/target transforms for dataloaders used for
            training and validation data in the search and genotype evaluation phases. Any tuple not specified will
            default to those of the reference implementation. Defaults to None (no override of any dataloader config).
        """
        train_splits = train_splits or {}
        self._train_split_map = {}
        for phase in Phase:
            if phase in train_splits:
                cur_split = train_splits[phase]
                if cur_split is not None and (cur_split < 0 or cur_split > 1):
                    raise ValueError(f"split must be None or in [0, 1], received {cur_split} for phase '{phase.name}'")
            else:
                cur_split = self.get_default_train_split(phase)
            self._train_split_map[phase] = cur_split

        self._config_map = {}
        configs = configs or {}
        for p in Phase:
            for s in Split:
                if (p, s) in configs:
                    self._config_map[(p, s)] = configs[(p, s)]
                else:
                    self._config_map[(p, s)] = self.get_default_config(p, s)

    @abc.abstractmethod
    def load_dataset(
        self,
        phase: Phase,
        split: Split,
        path: str,
        input_transform: Optional[Callable],
        target_transform: Optional[Callable],
        train_split: Optional[float] = None,
        download: bool = False,
    ) -> Optional[tud.Dataset]:
        """A method, to be overridden in derived classes, that produces the requested dataset.

        Parameters
        ----------
        phase: Phase
            The phase in which the dataloader is to be used.
        split: Split
            The split of the data being requested.
        path: str
            The path where the data may be found (or should be stored if `download`=True and it does not yet exist).
        input_transform: Optional[Callable]
            How the input data should be transformed. If None, no transform should be applied.
        target_transform: Optional[Callable]
            How the target data should be transformed. If None, no transform should be applied.
        train_split: Optional[float], optional
            How the training set should be split in the search/select phase for the train/val dataloaders. Defaults to
            None, but must be in the range [0, 1] inclusive if specified.
        download: bool, optional
            Whether the data should be downloaded if it does not yet exist at the specified `path`. Defaults to False.

        Returns
        -------
        Optional[torch.utils.data.Dataset]
            The corresponding Dataset (or None if the requested split would result in a null Dataloader)
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_default_config(phase: Phase, split: Split) -> BatchConfig:
        """Returns the `BatchConfig` used to reproduce the dataloader from the reference implementation."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_default_train_split(phase: Phase) -> Optional[float]:
        """Returns the fraction of the dataset used for training during `phase` in the reference implementation.

        Parameters
        ----------
        phase: Phase
            The phase for which the dataset is to be used

        Returns
        -------
        float or None
            A float representing the fraction of the dataset to use or None if a dedicated training set is available
        """
        raise NotImplementedError

    def get_config(self, phase: Phase, split: Split) -> BatchConfig:
        """Returns the `BatchConfig` to be used, including any overrides, in dataloader creation."""
        return self._config_map[(phase, split)]

    def get_train_split(self, phase: Phase) -> Optional[float]:
        """Returns the fraction of the dataset used for training during `phase`, including any overrides.

        Parameters
        ----------
        phase: Phase
            The phase for which the dataset is to be used

        Returns
        -------
        float or None
            A float representing the fraction of the dataset to use or None if a dedicated training set is available
        """
        return self._train_split_map[phase]

    def get_dataloader(
        self, phase: Phase, split: Split, path: str, download: bool = False, **kwargs: Any
    ) -> Optional[tud.DataLoader]:
        """Returns the dataloader that follows the reference implementation except for user-supplied overrides.

        Parameters
        ----------
        phase: Phase
            The phase in which the dataloader is to be used.
        split: Split
            The split of the data being requested.
        path: str
            The path where the data may be found (or should be stored if `download`=True and it does not yet exist).
        download: bool, optional
            Whether the dataset should be downloaded if it does not yet exist at the supplied path. Defaults to False.
        kwargs
            Any additional kwargs to be passed to the dataloader upon creation.

        Returns
        -------
        Optional[tud.DataLoader]
            The requested dataloader (or None if the requested configuration would results in an empty dataloader).
        """
        config = self.get_config(phase, split)
        ts = self.get_train_split(phase)
        dset = self.load_dataset(
            phase, split, path, config.input_transform, config.target_transform, download=download, train_split=ts,
        )
        if not dset:
            return None

        if "batch_size" in kwargs and kwargs["batch_size"] != config.batch_size:
            raise ValueError(
                f'`batch_size` specified in kwargs ({kwargs["batch_size"]}) does not match '
                f"`BatchConfig.batch_size` ({config.batch_size})."
            )

        kwargs = {"batch_size": config.batch_size, **(kwargs or {})}
        return tud.DataLoader(dset, **kwargs)
