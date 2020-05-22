# Reference Op Spaces
This subpackage provides a set of reference op spaces, which may be used to compare a given NAS algorithm against
others holding the model and searched operations fixed.

## Usage
Op spaces are implemented as a subclass of a python `Enum`, thereby conveying all properties that enums offer, namely
* Retrieve by name
* Strict typing
* Iteration
* ...

Additionally, the OpSpace base class allows for a given operation to be fetched by its numerical location in the space
(as if it were a list) using the `get_by_index` method as well as querying an individual operation's index via the
`index` property.

All operations are `Callable`s which return instantiated operations when invoked.


## Creating a Reference Implementation
New reference implementations should inherit from the `OpSpace` class. To define operations, create an entry in the
`Enum` with an appropriate name and a value of the form `OpContainer(...)` where the argument can be any callable
(lambda, function, or class). The `OpContainer` wrapper is necessary as any function bound to a name in an `Enum` is
implicitly treated as an instance method rather than an `Enum` value. See below for a sample implementation:

```python
import torch.nn as nn
from automl_utils.nas.op_space import OpContainer, OpSpace

class MyOpSpace(OpSpace):
    CONV2D = OpContainer(lambda *args, **kwargs: nn.Conv2d(*args, **kwargs))
    MAXPOOL3 = OpContainer(lambda *args, **kwargs: nn.MaxPool2d(3))
    AVGPOOL3 = OpContainer(lambda *args, **kwargs: nn.AvgPool2d(3))

mp = MyOpSpace.MAXPOOL3() 
ap = MyOpSpace.AVGPOOL3()
```
