import numpy as np
from kipoi_utils.external.torch.sampler import BatchSampler
import collections
from kipoi_utils.data_utils import (numpy_collate, numpy_collate_concat, get_dataset_item,
                                    DataloaderIterable, batch_gen, get_dataset_lens, iterable_cycle)
from copy import deepcopy
from bpnet.utils import flatten, unflatten
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from kipoi.writers import HDF5BatchWriter
from kipoi.readers import HDF5Reader

from kipoi.data import BaseDataLoader
try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    # use the Kipoi dataloader as a fall-back strategy
    from kipoi.data import DataLoader
import abc


def to_numpy(data):
    # import torch
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, collections.Mapping):
        return {key: to_numpy(data[key]) for key in data}
    elif isinstance(data, collections.Sequence):
        if isinstance(data[0], str):
            return data
        else:
            return [to_numpy(sample) for sample in data]
    else:
        raise ValueError("Leafs of the nested structure need to be numpy arrays")


class Dataset(BaseDataLoader):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return one sample

        index: {0, ..., len(self)-1}
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Return the number of all samples
        """
        raise NotImplementedError

    def _batch_iterable(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iteratrable

        See batch_iter docs

        Returns:
            Iterable
        """
        dl = DataLoader(self,
                        batch_size=batch_size,
                        # collate_fn=numpy_collate,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        **kwargs)
        return dl

    def batch_iter(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If False and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)

        Returns:
            iterator
        """
        dl = self._batch_iterable(batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  **kwargs)
        return (to_numpy(batch) for batch in dl)

    def batch_train_iter(self, cycle=True, **kwargs):
        """Returns samples directly useful for training the model:
        (x["inputs"],x["targets"])
        Args:
          cycle: when True, the returned iterator will run indefinitely go through the dataset
            Use True with `fit_generator` in Keras.
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        if cycle:
            return ((to_numpy(x["inputs"]), to_numpy(x["targets"]))
                    for x in iterable_cycle(self._batch_iterable(**kwargs)))
        else:
            return ((x["inputs"], x["targets"]) for x in self.batch_iter(**kwargs))

    def batch_predict_iter(self, **kwargs):
        """Returns samples directly useful for prediction x["inputs"]
        Args:
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        return (x["inputs"] for x in self.batch_iter(**kwargs))

    def load_all(self, batch_size=32, **kwargs):
        """Load the whole dataset into memory
        Arguments:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
        """
        from copy import deepcopy
        return numpy_collate_concat([deepcopy(x)
                                     for x in tqdm(self.batch_iter(batch_size,
                                                                   **kwargs),
                                                   total=len(self) // batch_size)])


def nested_numpy_minibatch(data, batch_size=1):
    lens = get_dataset_lens(data)
    if isinstance(lens, collections.Mapping):
        ln = [v for v in lens.values()][0]
    elif isinstance(lens, collections.Sequence):
        ln = lens[0]
    else:
        ln = lens

    for idx in BatchSampler(range(ln),
                            batch_size=batch_size,
                            drop_last=False):
        yield get_dataset_item(data, idx)


class NumpyDataset(Dataset):
    """Data-structure of arbitrarily nested arrays
       with the same first axis
    """

    def __init__(self, data, attrs=None):
        """

        Args:
          data: any arbitrarily nested dict/list of np.arrays
            with the same first axis size
          attrs: optional dictionary of attributes
        """
        self.data = data
        if attrs is None:
            self.attrs = OrderedDict()
        else:
            self.attrs = attrs

        self._validate()

    def _validate(self):
        # Make sure the first axis is the same
        # for k,v in flatten(data).items():
        assert len(set(self.get_lens())) == 1

    def get_lens(self):
        return list(flatten(self.dapply(len)).values())

    def __len__(self):
        return self.get_lens()[0]

    def __getitem__(self, idx):
        def get_item(arr, idx):
            return arr[idx]
        return self.dapply(get_item, idx=idx)

    def loc(self, idx):
        return super().__init__(self[idx], attrs=deepcopy(self.attrs))

    def flatten(self):
        return super().__init__(flatten(self.data), attrs=deepcopy(self.attrs))

    def unflatten(self):
        return super().__init__(unflatten(self.data), attrs=deepcopy(self.attrs))

    def shapes(self):
        from pprint import pprint

        def get_shape(arr):
            return str(arr.shape)

        out = self.dapply(get_shape)
        pprint(out)

    def dapply(self, fn, *args, **kwargs):
        """Apply a function to each element in the list

        Returns a nested dictionary
        """
        def _dapply(data, fn, *args, **kwargs):
            if type(data).__module__ == 'numpy':
                return fn(data, *args, **kwargs)
            elif isinstance(data, collections.Mapping):
                return {key: _dapply(data[key], fn, *args, **kwargs) for key in data}
            elif isinstance(data, collections.Sequence):
                return [_dapply(sample, fn, *args, **kwargs) for sample in data]
            else:
                raise ValueError("Leafs of the nested structure need to be numpy arrays")

        return _dapply(self.data, fn, *args, **kwargs)

    def sapply(self, fn, *args, **kwargs):
        """Same as dapply but returns NumpyDataset
        """
        return super().__init__(self.dapply(fn, *args, **kwargs), deepcopy(self.attrs))

    def aggregate(self, fn=np.mean, axis=0):
        """Aggregate across all tracks

        Args:
          idx: subset index
        """
        return self.dapply(fn, axis=axis)

    def shuffle(self):
        """Permute the order of seqlets
        """
        idx = pd.Series(np.arange(len(self))).sample(frac=1).values
        return self.loc(idx)

    def split(self, i):
        """Split the Dataset at a certain index
        """
        return self.loc(np.arange(i)), self.loc(np.arange(i, len(self)))

    def append(self, datax):
        """Append two datasets
        """
        return super().__init__(data=numpy_collate_concat([self.data, datax.data]),
                                attrs=deepcopy(self.attrs))

    def save(self, file_path, **kwargs):
        """Save the dataset to an hdf5 file
        """
        obj = HDF5BatchWriter(file_path=file_path, **kwargs)
        obj.batch_write(self.data)
        # Store the attrs
        for k, v in self.attrs.items():
            obj.f.attrs[k] = v
        obj.close()

    @classmethod
    def load(cls, file_path):
        """Load the dataset from an hdf5 dataset
        """
        with HDF5Reader(file_path) as obj:
            data = obj.load_all()
            attrs = OrderedDict(obj.f.attrs)
        return cls(data, attrs)

    @classmethod
    def concat(cls, objects):
        return cls(data=numpy_collate_concat(objects), attrs=None)
