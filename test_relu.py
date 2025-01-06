import numpy as np
from base.ops import relu
from base.tensor import Tensor

a = Tensor(np.array([[ 1.5, -2., 3.],
                     [-2.5, 3., 2.]]),
           need=True)
b = relu(a)
c = b.sum()
assert np.allclose(c.numpy(), 9.5)

y = 20
l = (y - c)**2
assert np.allclose(l.numpy(), 10.5**2)

l.back()
assert np.allclose(c._grad_acc.numpy(), -10.5*2)
assert np.allclose(b._grad_acc.numpy(), np.array([[1., 1., 1.], [1., 1., 1.]]) * (-10.5*2))
assert np.allclose(a._grad_acc.numpy(), np.array([[1., 0., 1.], [0., 1., 1.]]) * (-10.5*2))
