from __future__ import annotations
import numpy as np
from typing import List, Union

class Tensor:

    def __init__(self, v: Union[Tensor, np.ndarray, List], requires_grad: bool = False) -> None:
        if isinstance(v, list):
            val = np.array(v)
        elif isinstance(v, Tensor):
            val = v.val.copy()
        elif isinstance(v, np.ndarray):
            val = v
        else:
            raise TypeError("Only Tensor, np.ndarray, list objects are allowed")
        self.val = val
        self.need = requires_grad
        if self.need:
            self.grad = None
            self.grad_stack = None
            self._grad_acc = Tensor(np.ones(self.val.shape))

    def __repr__(self) -> str:
        r = 'Tensor'
        s = self.val.__str__().split('\n')
        fin = r + '(' + ('\n' + ' '*(len(r) + 1)).join(s) + ')'
        return fin

    def __str__(self) -> str:
        return f'{self.val.__str__()}'

    def __add__(self, other) -> Tensor:
        if isinstance(other, Tensor):
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
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val - opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.subtract, self, other]

        return out

    def _rsub(self, other) -> Tensor:
        if isinstance(other, Tensor):
            opd = other.val
        else:
            opd = other

        out = Tensor(opd - self.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.subtract, other, self]

        return out

    def __rsub__(self, other) -> Tensor:
        return self._rsub(other)

    def __mul__(self, other) -> Tensor:
        if isinstance(other, Tensor):
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
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val / opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.divide, self, other]

        return out

    def _rtruediv(self, other) -> Tensor:
        if isinstance(other, Tensor):
            opd = other.val
        else:
            opd = other

        out = Tensor(opd / self.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.divide, other, self]

        return out

    def __rtruediv__(self, other) -> Tensor:
        return self._rtruediv(other)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.negative, self]

        return out

    def __pow__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            opd = other.val
        else:
            opd = other

        out = Tensor(self.val ** opd, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.power, self, other]

        return out

    def _rpow(self, other) -> Tensor:
        if isinstance(other, Tensor):
            opd = other.val
        else:
            opd = other

        out = Tensor(opd ** self.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.power, other, self]

        return out

    def __rpow__(self, other) -> Tensor:
        return self._rpow(other)

    def __eq__(self, other: Union[List, np.ndarray, Tensor]) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val == other.val)
        else:
            out = Tensor(self.val == other)
        return out

    def __ne__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val != other.val)
        else:
            out = Tensor(self.val != other)
        return out

    def __lt__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val < other.val)
        else:
            out = Tensor(self.val < other)
        return out

    def __le__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val <= other.val)
        else:
            out = Tensor(self.val <= other)
        return out

    def __gt__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val > other.val)
        else:
            out = Tensor(self.val > other)
        return out

    def __ge__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            out = Tensor(self.val >= other.val)
        else:
            out = Tensor(self.val >= other)
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        out = Tensor(self.val @ other.val, requires_grad = self.need)

        if self.need:
            out.grad_stack = [np.matmul, self, other]

        return out

    def __getitem__(self, idx):
        return self.val.__getitem__(idx)

    def __setitem__(self, idx, val):
        return self.val.__setitem__(idx, val)

    def copy(self, requires_grad=True) -> Tensor:
        return Tensor(self.val.copy(), requires_grad=requires_grad)

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

    def back(self, f=None) -> None:
        if self.need:
            if f: self._grad_acc *= f
            if not self.grad_stack: return
            elif self.grad_stack[0] == np.add:
                if isinstance(self.grad_stack[1], Tensor): (self.grad_stack[1]).back(f=self._grad_acc)
                if isinstance(self.grad_stack[2], Tensor): (self.grad_stack[2]).back(f=self._grad_acc)
            elif self.grad_stack[0] == np.subtract:
                if isinstance(self.grad_stack[1], Tensor): (self.grad_stack[1]).back(f=self._grad_acc)
                if isinstance(self.grad_stack[2], Tensor): (self.grad_stack[2]).back(f=-self._grad_acc)
            elif self.grad_stack[0] == np.multiply:
                if isinstance(self.grad_stack[1], Tensor):
                    (self.grad_stack[1]).back(f=self._grad_acc*self.grad_stack[2])
                if isinstance(self.grad_stack[2], Tensor):
                    (self.grad_stack[2]).back(f=self._grad_acc*self.grad_stack[1])
            elif self.grad_stack[0] == np.divide:
                if isinstance(self.grad_stack[1], Tensor):
                    (self.grad_stack[1]).back(f=self._grad_acc/self.grad_stack[2])
                if isinstance(self.grad_stack[2], Tensor):
                    (self.grad_stack[2]).back(f=-self._grad_acc*self.grad_stack[1]/(self.grad_stack[2])**2)
            elif self.grad_stack[0] == np.negative:
                (self.grad_stack[1]).back(f=-self._grad_acc)
            elif self.grad_stack[0] == np.power:
                if isinstance(self.grad_stack[1], Tensor):
                    (self.grad_stack[1]).back(
                        f=self._grad_acc*self.grad_stack[2]*(self.grad_stack[1])**(self.grad_stack[2]-1)
                    )
                if isinstance(self.grad_stack[2], Tensor):
                    (self.grad_stack[2]).back(f=self._grad_acc*self*np.log(self.grad_stack[1].val))
            elif self.grad_stack[0] == np.exp:
                (self.grad_stack[1]).back(f=self._grad_acc*self)
            elif self.grad_stack[0] == np.log:
                (self.grad_stack[1]).back(f=self._grad_acc/self.grad_stack[1])

