import torch
from base.tensor import Tensor
from base.compose import Conv2d
from base.dist import Randn
from time import perf_counter

# WARNING: need better tests

print("simple convolution...")

a1 = torch.randn((4, 16, 16))
c1 = torch.nn.Conv2d(4, 10, 3)
start = perf_counter()
b1 = c1(a1)
end = perf_counter()
print(f"torch time: {end - start:.3}s")

a2 = Randn((4, 16, 16))
c2 = Conv2d(4, 10, 3)
start = perf_counter()
b2 = c2(a2)
end = perf_counter()
print(f"d time: {end - start:.3}s (expecting worse perf)")

assert a1.shape == a2.shape and c1.weight.shape == c2.weight.shape, b1.shape == b2.shape

##############################################

print("\ngrouped convolution...")
c1 = torch.nn.Conv2d(4, 10, 3, groups=2)
start = perf_counter()
b1 = c1(a1)
end = perf_counter()
print(f"torch time: {end - start:.3}s")

c2 = Conv2d(4, 10, 3, groups=2)
start = perf_counter()
b2 = c2(a2)
end = perf_counter()
print(f"d time: {end - start:.3}s")

assert a1.shape == a2.shape and c1.weight.shape == c2.weight.shape, b1.shape == b2.shape

##############################################

print("\ndilated convolution...")
c1 = torch.nn.Conv2d(4, 10, 3, dilation=2)
start = perf_counter()
b1 = c1(a1)
end = perf_counter()
print(f"torch time: {end - start:.3}s")

c2 = Conv2d(4, 10, 3, dilation=2)
start = perf_counter()
b2 = c2(a2)
end = perf_counter()
print(f"d time: {end - start:.3}s")

assert a1.shape == a2.shape and c1.weight.shape == c2.weight.shape, b1.shape == b2.shape

##############################################

# for batches
a1 = torch.randn((1_000, 3, 16, 16))
c1 = torch.nn.Conv2d(3, 10, 3)

print("\n(big) batched convolution...")

start = perf_counter()
b1 = c1(a1)
end = perf_counter()
print(f"torch time: {end - start:.3}s")

a2 = Randn((1_000, 3, 16, 16))
c2 = Conv2d(3, 10, 3)
start = perf_counter()
b2 = c2(a2)
end = perf_counter()
print(f"d time (shameful perfomance TT): {end - start:.3}s")

assert a1.shape == a2.shape and c1.weight.shape == c2.weight.shape, b1.shape == b2.shape
