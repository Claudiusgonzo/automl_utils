# Reference Dataloaders
This subpackage provides a set of reference dataloaders, which may be used to compare a given NAS algorithm against
others holding the data pre-processing and data augmentation fixed.

## Usage
To use an existing reference dataloader, simply import the class corresponding to the algorithm and dataset of interest,
instantiate it, and invoke `.get_dataloader` with the appropriate arguments. Should you wish to override how the data 
split in the search phase or the batching/pre-processing/augmentation, this may be done when instantiating the object
via a supplied mapping. Any argument not set will implicitly retain its default value. See below for an example.

```python
from automl_utils.nas.dataloader import BatchConfig, Phase, Split
from automl_utils.nas.dataloader import darts

c10 = darts.CIFAR10(
    search_split=0.8,
    config_map={(Phase.SEARCH, Split.EVAL): BatchConfig(batch_size=128, input_transform=None, target_transform=None)},
)

dl_train = c10.get_dataloader(Phase.SEARCH, Split.TRAIN, '/my/data/path', kwargs={'pin_memory': True, 'num_workers': 4})
dl_eval = c10.get_dataloader(Phase.SEARCH, Split.EVAL, '/my/data/path', kwargs={'pin_memory': True, 'num_workers': 4})
``` 

## Creating a New Reference Implementation
New reference implementations should inherit and implement the `DataloaderSpec` interface. For most use cases, the
supplied default `get_dataloader` method should be sufficient which therefore reduces the task to the implementation of
the following abstract methods:

```python
from automl_utils.nas.dataloader.spec import DataloaderSpec

class MyDatasetImpl(DataloaderSpec):
    def load_dataset(...):
        # load the dataset from the specified path with the specified input and target transforms
        pass
    
    @staticmethod
    def get_default_config(...):
        # for the given (phase, split) tuple, construct and return a `BatchConfig` object corresponding to how the data
        # should be transformed
        pass
    
    @staticmethod
    def get_default_search_split():
        # returns the default percentage (as a fraction between 0 and 1) of the training data that should be used for
        # training set during architecture search
        pass
```

Should the default `get_dataloader` implementation not be sufficient, it may be overridden in derived classes. If doing
so, please remember to use the `get_config` method and `search_split` property, which respect overrides supplied by the
user in the `__init__` method, to fetch dataset and dataloader configuration parameters. 