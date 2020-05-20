# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for constructing and transforming datasets."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset


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

        def __getitem__(self, item: int) -> Dict[str, Any]:
            return {n: x for n, x in zip(self._names, self._dset[item])}

    return DictDatasetWrapper(dset, names)


class PrefetchToGPU:
    """An iterable that wraps a dataloader to ensure batches are pre-fetched to the GPU.

    Note: It assumes the batch will be used in the current stream by invoking `record_stream` on the tensors.
    """

    def __init__(
        self,
        dl: DataLoader,
        device: Union[str, torch.device],
        num_prefetch: int = 1,
        input_key: Hashable = 0,
        label_key: Hashable = 1,
    ):
        """Creates a iterable which pre-fetches batches to the GPU.

        Parameters
        ----------
        dl: DataLoader
            The dataloader whose batches should be pre-fetched
        device: str or torch.device
            The device to which the batches will be pre-fetched
        num_prefetch: int
            The number of batches to pre-fetch
        input_key: Hashable
            Passed to the batch's __getitem__ method to obtain the model's input data
        label_key: Hashable
            Passed to the batch's __getitem__ method to obtain the corresponding labels for the input data

        """
        device = device if isinstance(device, torch.device) else torch.device(device)
        if not dl.pin_memory:
            raise ValueError("`pin_memory` must be enabled for asynchronous prefetching to the GPU.")
        if device.type != "cuda":
            raise ValueError("Prefetching only supported to cuda devices.")
        if num_prefetch <= 0 or num_prefetch != int(num_prefetch):
            raise ValueError("`num_prefetch` must be a positive integer.")

        self._dl = dl
        self._device = device
        self._num_prefetch = num_prefetch
        self._input_key = input_key
        self._label_key = label_key

    def __iter__(self) -> _PrefetchToGpuIter:
        """Returns an iterator which pre-fetches batches to the device."""
        return _PrefetchToGpuIter(self._dl, self._device, self._num_prefetch, self._input_key, self._label_key)

    def __len__(self) -> int:
        """Returns the length of the dataloader."""
        return len(self._dl)


class _PrefetchToGpuIter:
    def __init__(
        self,
        dl: DataLoader,
        device: Union[str, torch.device],
        num_prefetch: int = 1,
        input_key: Hashable = 0,
        label_key: Hashable = 1,
    ):
        self._dl = dl
        self._num_prefetch = num_prefetch
        self._input_key = input_key
        self._label_key = label_key

        self._device = device if isinstance(device, torch.device) else torch.device(device)
        # pin to the correct device
        with torch.cuda.device(self._device):  # type: ignore
            self._dl_iter = iter(dl)
        self._stream = torch.cuda.Stream(self._device)  # type: ignore

        self._data: List[Tuple[Optional[BatchLike], Optional[torch.cuda.Event]]] = []  # type: ignore
        self._event_cache = [torch.cuda.Event() for _ in range(self._num_prefetch)]  # type: ignore
        self._prefetch(self._num_prefetch)

    def _prefetch(self, n: int = 1) -> None:
        """Asynchronously prefetches n batches from the dataloader to the GPU.

        Parameters
        ----------
        n: int, optional
            The number of batches to prefetch. Defaults to 1.

        Returns
        -------
        None

        """
        with torch.cuda.stream(self._stream):  # type: ignore
            for _ in range(n):
                try:
                    # fetch the next batch from the dataloader
                    # note: if the dataloader has yet to process the next batch, this step will block
                    batch = next(self._dl_iter)

                    # asynchronously move the batch to the device while retaining the type
                    t = type(batch)
                    if isinstance(batch, Sequence):
                        batch = t([elem.to(device=self._device, non_blocking=True) for elem in batch])
                    elif isinstance(batch, Mapping):
                        batch = t({k: v.to(device=self._device, non_blocking=True) for k, v in batch.items()})
                    else:
                        raise ValueError(f"`_PrefetchToGpuIter` does not support batch type {t}")

                    # record an event so __next__ knows when the transfer of the batch is complete
                    if self._event_cache:
                        event = self._event_cache.pop(0)
                    else:
                        event = torch.cuda.Event()  # type: ignore
                    event = self._stream.record_event(event)
                    self._data.append((batch, event))
                except StopIteration:
                    # we intentionally discard the event as the iterator will be recreated for the next epoch
                    self._data.append((None, None))

    def __iter__(self) -> _PrefetchToGpuIter:
        return self

    def __next__(self) -> BatchLike:
        # get the next pre-fetched batch (note: it may not yet be fully copied to the GPU)
        batch, event = self._data.pop(0)

        if batch is None or event is None:
            # we have exhausted all of our prefetched batches
            # this means we have reached the end of the epoch
            assert batch is None and event is None
            raise StopIteration

        # ensure the batch has been fully copied to the GPU before allowing work on the current stream to continue
        # NOTE: we assume the batch will be used in the current stream from which the data is fetched
        cur_stream = torch.cuda.current_stream()  # type: ignore
        cur_stream.wait_event(event)

        # add event back to the cache for later re-use
        self._event_cache.append(event)

        # mark that the batch is being used in a different stream
        # this is done to ensure the memory is not reallocated while still in use in the current stream
        # See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.record_stream for more details
        if isinstance(batch, Sequence):
            for elem in batch:
                elem.record_stream(cur_stream)  # type: ignore
        elif isinstance(batch, Mapping):
            for elem in batch.values():
                elem.record_stream(cur_stream)  # type: ignore
        else:
            raise ValueError(f"`_PrefetchToGpuIter` does not support batch type {type(batch)}")

        # prefetch the next batch
        self._prefetch()
        return batch


BatchLike = Union[Sequence[torch.Tensor], Dict[Hashable, torch.Tensor]]
DataLoaderLike = Union[DataLoader, PrefetchToGPU]
