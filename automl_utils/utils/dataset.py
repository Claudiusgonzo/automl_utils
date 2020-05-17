# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for constructing and transforming datasets."""

from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset


def make_dict_dataset(dset: Dataset, names: Sequence[str]) -> Dataset:
    """Converts a dataset returning a tuple per entry to one which returns a dict per entry.

    Parameters
    ----------
    dset: Dataset
        A dataset which returns a tuple of values
    names: Sequence[str]
        A sequence of strings which serve as keys in the converted dict-returning dataset

    Returns
    -------
    Dataset

    """

    class DictDatasetWrapper(Dataset):
        def __init__(self, dset: Dataset, names: Sequence[str]):
            assert len(dset[0]) == len(names)
            self._dset = dset
            self._names = names

        def __len__(self) -> int:
            return len(self._dset)

        def __getitem__(self, item: int) -> Mapping[str, Any]:
            return {n: x for n, x in zip(self._names, self._dset[item])}

    return DictDatasetWrapper(dset, names)
