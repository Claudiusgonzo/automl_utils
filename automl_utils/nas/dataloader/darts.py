# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reference implementations of DARTS dataloaders."""

import os.path as osp
from typing import Callable, Optional

import torch.utils.data as tud
import torchvision.datasets as dsets
import torchvision.transforms as tfs

from automl_utils.nas.dataloader.spec import BatchConfig, DataloaderSpec, Split
from automl_utils.nas.dataloader.vision_transforms import Cutout
from automl_utils.nas.phase import Phase


def _cifar_tf(valid: bool, cutout: bool) -> Callable:
    assert not (valid and cutout), "Invalid configuration of val + cutout"

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    tf = tfs.Compose([tfs.ToTensor(), tfs.Normalize(CIFAR_MEAN, CIFAR_STD)])
    if not valid:
        tf = tfs.Compose([tfs.RandomCrop(32, padding=4), tfs.RandomHorizontalFlip(), tf])

    if cutout:
        tf = tfs.Compose([tf, Cutout(16)])

    return tf


def _imagenet_tf(valid: bool) -> Callable:
    normalize = tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if valid:
        return tfs.Compose([tfs.Resize(256), tfs.CenterCrop(224), tfs.ToTensor(), normalize])
    else:
        return tfs.Compose(
            [
                tfs.RandomResizedCrop(224),
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                tfs.ToTensor(),
                normalize,
            ]
        )


class CIFAR10(DataloaderSpec):
    """Generates search and genotype eval dataloaders for CIFAR10 as done in DARTS."""

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
        """Loads CIFAR10 from disk and optionally downloads it if it's not present at the specified path.

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
            How the training set should be split in the search phase for the train/val dataloaders. Defaults to None,
            but must be in the range [0, 1] inclusive if `phase` == Phase.SEARCH.
        download: bool, optional
            Whether the data should be downloaded if it does not yet exist at the specified `path`. Defaults to False.

        Returns
        -------
        Optional[torch.utils.data.Dataset]
            The CIFAR10 datset (or None if the requested split would result in an empty dataset).
        """
        if phase != Phase.SEARCH and train_split is not None:
            raise ValueError("DARTS only splits the training dataset in the search phase.")
        elif phase == Phase.SEARCH:
            if train_split is None:
                raise ValueError("`train_split` cannot be None for the search phase")
            elif train_split < 0 or train_split > 1:
                raise ValueError("`train_split` must be in [0, 1] for the search phase")

        if phase == Phase.SEARCH or split == Split.TRAIN:
            # darts uses only the training dataset in the search phase (it splits it) or for training genotypes
            dset = dsets.CIFAR10(path, True, input_transform, target_transform, download)
        else:
            # darts uses the full validation set as the val set for Phase.SELECT and Phase.EVAL
            dset = dsets.CIFAR10(path, False, input_transform, target_transform, download)

        if phase == Phase.SEARCH:
            # in search, darts splits the "true training set" into a search-train set and a search-val sset.
            idx = int(train_split * len(dset))  # type: ignore
            dset = tud.Subset(dset, list(range(0, idx) if split == Split.TRAIN else range(idx, len(dset))))

        return dset if dset else None

    @staticmethod
    def get_default_config(phase: Phase, split: Split) -> BatchConfig:
        """Returns the `BatchConfig` corresponding to the dataloaders used in DARTS."""
        if phase == Phase.SEARCH:
            return BatchConfig(batch_size=64, input_transform=_cifar_tf(False, False), target_transform=None)
        elif phase in (Phase.SELECT, Phase.EVAL) and split == Split.TRAIN:
            return BatchConfig(batch_size=96, input_transform=_cifar_tf(False, True), target_transform=None)
        elif phase in (Phase.SELECT, Phase.EVAL) and split == Split.VAL:
            return BatchConfig(batch_size=96, input_transform=_cifar_tf(True, False), target_transform=None)
        else:
            raise ValueError("Invalid combination of phase and split")

    @staticmethod
    def get_default_train_split(phase: Phase) -> Optional[float]:
        """Returns the split of the training dataset used during architecture search in DARTS."""
        if phase == Phase.SEARCH:
            return 0.5
        return None


class ImageNet(CIFAR10):
    """Generates search and genotype eval dataloaders for ImageNet as done in DARTS.

    Note: DARTS transfers an architecture learned on CIFAR10 to ImageNet for genotype evaluation.
    """

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
        """Loads CIFAR10/ImageNet from disk and optionally downloads it if it's not present at the specified path.

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
        search_split: Optional[float], optional
            How the training set should be split in the search phase for the train/val dataloaders. Defaults to None,
            but must be in the range [0, 1] inclusive if `phase` == Phase.SEARCH.
        download: bool, optional
            Whether the data should be downloaded if it does not yet exist at the specified `path`. Defaults to False.

        Returns
        -------
        Optional[torch.utils.data.Dataset]
            The CIFAR10/ImageNet datset (or None if the requested split would result in an empty dataset).
        """
        if phase in (Phase.SEARCH, Phase.SELECT):
            return super().load_dataset(phase, split, path, input_transform, target_transform, train_split, download)
        else:
            if download is True:
                raise ValueError("`download=True` is not compatible with ImageNet.")

            if split == Split.TRAIN:
                split_str = "train"
            elif split == Split.VAL:
                split_str = "val"
            else:
                raise ValueError(f"Unknown value of `split` ({split})")

            return dsets.ImageFolder(
                osp.join(path, split_str),
                self.get_config(phase, split).input_transform,
                self.get_config(phase, split).target_transform,
            )

    @staticmethod
    def get_default_config(phase: Phase, split: Split) -> BatchConfig:
        """Returns the `BatchConfig` corresponding to the dataloaders used in DARTS."""
        if phase in (Phase.SEARCH, Phase.SELECT):
            return CIFAR10.get_default_config(phase, split)
        elif phase == Phase.EVAL and split == Split.TRAIN:
            return BatchConfig(batch_size=128, input_transform=_imagenet_tf(False), target_transform=None)
        elif phase == Phase.EVAL and split == Split.VAL:
            return BatchConfig(batch_size=128, input_transform=_imagenet_tf(True), target_transform=None)
        else:
            raise ValueError("Invalid combination of phase and split")
