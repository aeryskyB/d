import torch
from base.tensor import Tensor
from base.ops import Conv2d
from base.dist import Randn
from time import perf_counter

a1 = torch.randn((4, 16, 16))
c1 = torch.nn.Conv2d(4, 10, 3)
start = perf_counter()
b1 = c1(a1)
end = perf_counter()
print(f'torch time: {end - start:.3}')

a2 = Randn((4, 16, 16))
c2 = Conv2d(4, 10, 3)
start = perf_counter()
b2 = c2(a2)
end = perf_counter()
print(f'd time: {end - start:.3} (expecting worse perf)')

assert a1.shape == a2.shape and c1.weight.shape == c2.weight.shape, b1.shape == b2.shape
