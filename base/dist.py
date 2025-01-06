import numpy as np
from .tensor import Tensor
from collections.abc import Iterable

class Zeros(Tensor):
    def __init__(self, shape: Iterable, need: bool = False):
        assert isinstance(shape, Iterable), "`shape` is not iterable"
        self.shape = tuple(shape)
        super().__init__(np.zeros(shape), need=need)

class Randn(Tensor):
    def __init__(self, shape: Iterable, need: bool = False):
        assert isinstance(shape, Iterable), "`shape` is not iterable"
        self.shape = tuple(shape)
        super().__init__(np.random.randn(*shape), need=need)

class Uniform(Tensor):
    def __init__(self, low: float = 0.0, high: float = 1.0, shape: Iterable = None, need: bool = False):
        assert isinstance(shape, Iterable), "`shape` is not iterable"
        self.shape = tuple(shape)
        super().__init__(np.random.uniform(low=low, high=high, size=shape), need=need)
