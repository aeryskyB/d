import numpy as np
from base.tensor import Tensor

a = Tensor(np.arange(9).reshape((3, 3)))
b = Tensor(np.eye(3))
c = Tensor(np.arange(3).reshape((3, 1)))

print(a)
print(a + 1)
print(1 + a)
print(a - 1)
print(a * 2)
print(a / 2)
print(a ** 3)
print(-a)
print(a + b)
print(a * b)
print(a @ c)
print(a[0])
print(a[0, -1])

a[0, 0] = 9
print(a)
