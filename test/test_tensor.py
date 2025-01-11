import unittest
import numpy as np
from base.tensor import Tensor

a = Tensor(np.arange(9).reshape((3, 3)))
b = Tensor(np.eye(3))
c = Tensor(np.arange(3).reshape((3, 1)))
d = Tensor(a)

class TestBasicTensor(unittest.TestCase):

    def test_numpy(self):
        assert (a.numpy() == np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])).all()

    def test_add_scalar(self):
        assert ((a+1).numpy() == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()
        assert ((1+a).numpy() == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).all()

    def test_sub_scalar(self):
        assert ((a-1).numpy() == np.array([[-1, 0, 1], [2, 3, 4], [5, 6, 7]])).all(), f"{a-1}"
        assert ((1-a).numpy() == np.array([[1, 0, -1], [-2, -3, -4], [-5, -6, -7]])).all()

    def test_mul_scalar(self):
        assert ((a*2).numpy() == np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])).all()
        assert ((2*a).numpy() == np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])).all()

    def test_div_scalar(self):
        assert ((a/2).numpy() == np.array([[0., 0.5, 1.], [1.5, 2., 2.5], [3., 3.5, 4.]])).all()

    def test_pow_scalar(self):
        assert ((a**3).numpy() == np.array([[0, 1, 8], [27, 64, 125], [216, 343, 512]])).all()

    def test_negation(self):
        assert ((-a).numpy() == np.array([[0, -1, -2], [-3, -4, -5], [-6, -7, -8]])).all()

    def test_add_vec(self):
        assert ((a+b).numpy() == np.array([[1, 1, 2], [3, 5, 5], [6, 7, 9]])).all()

    def test_mult_vec(self):
        assert ((a*b).numpy() == np.array([[0, 0, 0], [0, 4, 0], [0, 0, 8]])).all()

    def test_matmul(self):
        assert ((a@c).numpy() == np.array([[5], [14], [23]])).all()

    def test_eq(self):
        assert (d==a).numpy().all()
        assert ((1==a).numpy() == np.array([[False, True, False], [False, False, False], [False, False, False]])).all()
        assert ((a==1).numpy() == np.array([[False, True, False], [False, False, False], [False, False, False]])).all()

    def test_neq(self):
        assert ((1!=a).numpy() == np.array([[True, False, True], [True, True, True], [True, True, True]])).all()
        assert ((a!=1).numpy() == np.array([[True, False, True], [True, True, True], [True, True, True]])).all()

    def test_lt(self):
        assert ((a<1).numpy() == np.array([[True, False, False], [False, False, False], [False, False, False]])).all()
        assert ((1<a).numpy() == np.array([[False, False, True], [True, True, True], [True, True, True]])).all()

    def test_leq(self):
        assert ((1<=a).numpy() == ~np.array([[True, False, False], [False, False, False], [False, False, False]])).all()
        assert ((a<=1).numpy() == ~np.array([[False, False, True], [True, True, True], [True, True, True]])).all()

    def test_getitem(self):
        assert (a[0].numpy() == np.array([0, 1, 2])).all()
        assert a[0, -1] == 2

        t = Tensor([1., 2., 3.])
        assert t[0] == 1.

    def test_setitem(self):
        tmp = a[0, 0]
        a[0, 0] = 9
        assert a[0, 0] == 9
        a[0, 0] = tmp
        
    def test_transpose(self):
        a = Tensor([[1, 2]])
        aT = a.transpose()
        b = aT @ a
        assert aT.shape == (2, 1)
        assert b.shape == (2, 2)
        assert np.allclose(b.numpy(), np.array([[1, 2], [2, 4]]))

    def test_grad(self):
        # TODO: try other grads

        A = Tensor([[1, 2], [3, 4]], need=True)
        B = Tensor([[1], [2]], need=True)
        C = A @ B
        D = C.sum()

        assert (C.numpy() == np.array([[5], [11]])).all()
        assert D == 16

        assert (A._grad_acc.numpy() == np.array([[1., 1.], [1., 1.]])).all()
        assert (B._grad_acc.numpy() == np.array([[1.], [1.]])).all()

        D.back()
        assert (A._grad_acc.numpy() == np.array([[1., 2.], [1., 2.]])).all()
        assert (B._grad_acc.numpy() == np.array([[4.], [6.]])).all()

        D.reset_grad()
        assert (A._grad_acc.numpy() == np.array([[1., 1.], [1., 1.]])).all()
        assert (B._grad_acc.numpy() == np.array([[1.], [1.]])).all()

if __name__ == "__main__":
    unittest.main()
