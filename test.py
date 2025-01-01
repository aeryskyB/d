import numpy as np
from base.tensor import Tensor

a = Tensor(np.arange(9).reshape((3, 3)))
b = Tensor(np.eye(3))
c = Tensor(np.arange(3).reshape((3, 1)))
d = Tensor(a)

assert (a.numpy() == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])).all()
assert ((a+1).numpy() == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()
assert ((1+a).numpy() == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()
assert ((a-1).numpy() == np.array([[-1, 0, 1], [2, 3, 4], [5, 6, 7]])).all()
assert ((1-a).numpy() == np.array([[1, 0, -1], [-2, -3, -4], [-5, -6, -7]])).all()
assert ((a*2).numpy() == np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])).all()
assert ((2*a).numpy() == np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])).all()
assert ((a/2).numpy() == np.array([[0., 0.5, 1.], [1.5, 2., 2.5], [3., 3.5, 4.]])).all()
assert ((a**3).numpy() == np.array([[0, 1, 8], [27, 64, 125], [216, 343, 512]])).all()
assert ((-a).numpy() == np.array([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]])).all()
assert ((a+b).numpy() == np.array([[1, 1, 2], [3, 5, 5], [6, 7, 9]])).all()
assert ((a*b).numpy() == np.array([[0, 0, 0], [0, 4, 0], [0, 0, 8]])).all()
assert ((a@c).numpy() == np.array([[5], [14], [23]])).all()

assert (d==a).numpy().all()
assert ((1==a).numpy() == np.array([[False, True, False], [False, False, False], [False, False, False]])).all()
assert ((a==1).numpy() == np.array([[False, True, False], [False, False, False], [False, False, False]])).all()
assert ((1!=a).numpy() == np.array([[True, False, True], [True, True, True], [True, True, True]])).all()
assert ((a!=1).numpy() == np.array([[True, False, True], [True, True, True], [True, True, True]])).all()
assert ((a<1).numpy() == np.array([[True, False, False], [False, False, False], [False, False, False]])).all()
assert ((1<=a).numpy() == ~np.array([[True, False, False], [False, False, False], [False, False, False]])).all()
assert ((1<a).numpy() == np.array([[False, False, True], [True, True, True], [True, True, True]])).all()
assert ((a<=1).numpy() == ~np.array([[False, False, True], [True, True, True], [True, True, True]])).all()

assert (a[0].numpy() == np.array([0, 1, 2])).all()
assert a[0, -1] == 2

a[0, 0] = 9
assert a[0, 0] == 9
