import numpy as np
from .tensor import Tensor
from collections.abc import Iterable

class Zeros(Tensor):
    def __init__(self, shape: Iterable, need: bool = False):
        assert isinstance(shape, Iterable), "`shape` is not iterable"
        self.shape = tuple(shape)
        super().__init__(np.zeros(shape), need)

class Randn(Tensor):
    def __init__(self, shape: Iterable, need: bool = False):
        assert isinstance(shape, Iterable), "`shape` is not iterable"
        self.shape = tuple(shape)
        super().__init__(np.random.randn(*shape), need)
