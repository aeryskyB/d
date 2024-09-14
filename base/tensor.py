from __future__ import annotations
import numpy as np
from typing import List, Union

class Tensor:

    def __init__(self, val: Union[np.ndarray, List], requires_grad: bool = False) -> None:
        if isinstance(val, list):
            val = np.array(val)
        self.val = val
        self.need = requires_grad
        if self.need:
            self.grad = None
            self.grad_stack = None

    def __repr__(self) -> str:
        r = 'Tensor'
        s = self.val.__str__().split('\n')
        fin = r + '(' + ('\n' + ' '*(len(r) + 1)).join(s) + ')' + '\n'
        return fin

    def __str__(self) -> str:
        return f'{self.val.__str__()}'

    def __add__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            assert not (self.need ^ other.need), 'both of the operands\' gradient requirements should be same'
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val + opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.add, self, other]

        return out

    def __radd__(self, other) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            assert not (self.need ^ other.need), 'both of the operands\' gradient requirements should be same'
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val - opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.subtract, self, other]

        return out

    def __rsub__(self, other) -> Tensor:
        return self.__sub__(other)

    def __mul__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            assert not (self.need ^ other.need), 'both of the operands\' gradient requirements should be same'
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val * opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.multiply, self, other]

        return out

    def __rmul__(self, other) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            assert not (self.need ^ other.need), 'both of the operands\' gradient requirements should be same'
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val / opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.divide, self, other]

        return out

    def __rtruediv__(self, other) -> Tensor:
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.negative, self]

        return out

    def __pow__(self, ex: Union[int, float]) -> Tensor:
        out = Tensor(self.val ** ex, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.power, self, ex]

        return out

    def __rpow__(self, ex) -> Tensor:
        return self.__pow__(ex)

    def __matmul__(self, other: Tensor) -> Tensor:
        assert not (self.need ^ other.need), 'both of the operands\' gradient requirements should be same'

        out = Tensor(self.val @ other.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.matmul, self, other]

        return out

    def __getitem__(self, idx):
        return self.val.__getitem__(idx)

    def __setitem__(self, idx, val):
        return self.val.__setitem__(idx, val)

    '''
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Tensor:
        inputs = [np.asarray(x) for x in inputs]
        result = getattr(ufunc, method)(*inputs, **kwargs)
        out = Tensor(result)
        if self.need:
            out.grad = [ufunc, self]
        return out
    '''

    def exp(self, **kwargs) -> Tensor:
        out = Tensor(np.exp(self.val, **kwargs), requires_grad=self.need)

        if self.need:
            out.grad_stack = [np.exp, self]

        return out

    def log(self, **kwargs) -> Tensor:
        out = Tensor(np.log(self.val, **kwargs), requires_grad=self.need)

        if self.need:
            out.grad.stack = [np.log, self]

        return out

    def sum(self, **kwargs) -> Tensor:
        return Tensor(np.sum(self.val, **kwargs))
