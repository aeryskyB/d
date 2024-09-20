from __future__ import annotations
import numpy as np
from typing import List, Union

## for keeping track
# l = []

class Tensor:
    def __init__(self, v, need: bool = False) -> None:
        if isinstance(v, Tensor):
            val = v.val.copy()
        else:
            val = np.array(v)

        self.val = val
        self.need = need

        if self.need:
            self.grad_stack = None
            self._grad_acc = Tensor(np.ones(self.val.shape))

        # l.append(self)

    def __repr__(self) -> str:
        r = '\nTensor'

        s = self.val.__str__().split('\n')

        fin = r + '(' + ('\n' + ' '*(len(r))).join(s)
        fin += f', need={self.need}' if self.need else ''
        fin += ')\n'

        return fin

    def __str__(self) -> str:
        fin = f'{self.val.__str__()}'
        fin += f', need={self.need}' if self.need else ''

        return fin

    def __add__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            o_need = other.need
            opd = other.val
            out = Tensor(self.val + opd, need = self.need or other.need)
        else:
            o_need = False
            opd = other
            out = Tensor(self.val + opd, need = self.need)

        if self.need or o_need:
            out.grad_stack = [np.add, self, other]

        return out

    def __radd__(self, other) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            o_need = other.need
            opd = other.val
            out = Tensor(self.val - opd, need = self.need or other.need)
        else:
            o_need = False
            opd = other
            out = Tensor(self.val - opd, need = self.need)

        if self.need or o_need:
            out.grad_stack = [np.subtract, self, other]

        return out

    def _rsub(self, other) -> Tensor:
        out = Tensor(other - self.val, need = self.need)

        if self.need:
            out.grad_stack = [np.subtract, other, self]

        return out

    def __rsub__(self, other) -> Tensor:
        return self._rsub(other)

    def __mul__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            o_need = other.need
            opd = other.val
            out = Tensor(self.val * opd, need = self.need or other.need)
        else:
            o_need = False
            opd = other
            out = Tensor(self.val * opd, need = self.need)

        if self.need or o_need:
            out.grad_stack = [np.multiply, self, other]

        return out

    def __rmul__(self, other) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            o_need = other.need
            opd = other.val
            out = Tensor(self.val / opd, need = self.need or other.need)
        else:
            o_need = False
            opd = other
            out = Tensor(self.val / opd, need = self.need)

        if self.need or o_need:
            out.grad_stack = [np.divide, self, other]

        return out

    def _rtruediv(self, other) -> Tensor:
        out = Tensor(other / self.val, need = self.need)

        if self.need:
            out.grad_stack = [np.divide, other, self]

        return out

    def __rtruediv__(self, other) -> Tensor:
        return self._rtruediv(other)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.val, need = self.need)

        if self.need:
            out.grad_stack = [np.negative, self]

        return out

    def __pow__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            o_need = other.need
            opd = other.val
            out = Tensor(self.val ** opd, need = self.need or other.need)
        else:
            o_need = False
            opd = other
            out = Tensor(self.val ** opd, need = self.need)

        if self.need or o_need:
            out.grad_stack = [np.power, self, other]

        return out

    def _rpow(self, other) -> Tensor:
        if isinstance(other, Tensor):
            opd = other.val
            out = Tensor(self.val ** opd, need = self.need or other.need)
        else:
            opd = other
            out = Tensor(opd ** self.val, need = self.need or other.need)

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
        assert isinstance(other, Tensor), "only tensor matmuls supported for now"
        out = Tensor(self.val @ other.val, need = self.need or other.need)

        if self.need or isinstance(other, Tensor):
            out.grad_stack = [np.matmul, self, other]

        return out

    def __getitem__(self, idx):
        return self.val.__getitem__(idx)

    def __setitem__(self, idx, val):
        return self.val.__setitem__(idx, val)

    def len(self):
        return len(self.val)

    def copy(self, need=True) -> Tensor:
        return Tensor(self.val.copy(), need=need)

    def exp(self, **kwargs) -> Tensor:
        out = Tensor(np.exp(self.val, **kwargs), need=self.need)

        if self.need:
            out.grad_stack = [np.exp, self]

        return out

    def log(self, **kwargs) -> Tensor:
        out = Tensor(np.log(self.val, **kwargs), need=self.need)

        if self.need:
            out.grad_stack = [np.log, self]

        return out

    def sum(self, **kwargs) -> Tensor:
        out_val = np.array(np.sum(self.val, **kwargs))
        out = Tensor(out_val, need=self.need)

        if self.need:
            out.grad_stack = [np.sum, self]

        return out

    def transpose(self) -> Tensor:
        out = Tensor(self.val.T, need=self.need)

        if self.need:
            out.grad_stack = [np.transpose, self]

        return out

    def reshape(self, shape=None) -> Tensor:
        out = Tensor(self.val.reshape(shape), need=self.need)

        if self.need:
            out.grad_stack = [np.reshape, self, self.val.shape]

        return out

    def back(self, f=None) -> None:
        if self.need:
            if f: self._grad_acc = self._grad_acc * f
            ## debugging
            # print(f'g: {self._grad_acc}')
            # print(f's: {self.grad_stack}')

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
            elif self.grad_stack[0] == np.transpose:
                (self.grad_stack[1]).back(f=self._grad_acc.transpose())
            elif self.grad_stack[0] == np.reshape:
                (self.grad_stack[1]).back(f=self._grad_acc.reshape(self.grad_stack[2]))
            elif self.grad_stack[0] == np.matmul:
                (self.grad_stack[1]).back(f=self._grad_acc@self.grad_stack[2].transpose())
                (self.grad_stack[2]).back(f=self.grad_stack[1].transpose()@self._grad_acc)
            elif self.grad_stack[0] == np.sum:
                (self.grad_stack[1]).back(f=self._grad_acc)

    def reset_grad(self) -> None:
        if not self.need: return

        self._grad_acc = self._grad_acc*0 + 1

        if not self.grad_stack: return

        for t in self.grad_stack:
            if isinstance(t, Tensor) and t.need:
                t.reset_grad()

    def update_incr(self, delta) -> None:
        if isinstance(delta, Tensor): delta = delta.val
        self.val = self.val + delta

    def update_decr(self, delta) -> None:
        if isinstance(delta, Tensor): delta = delta.val
        self.val = self.val - delta

    def update_mult(self, delta) -> None:
        if isinstance(delta, Tensor): delta = delta.val
        self.val = self.val * delta

    def update_div(self, delta) -> None:
        if isinstance(delta, Tensor): delta = delta.val
        self.val = self.val / delta

