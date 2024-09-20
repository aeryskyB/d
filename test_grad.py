from base.tensor import Tensor
import numpy as np

# scalar test (visual): https://cs231n.github.io/optimization-2/
fwd = [Tensor([2.0], need=True), 
       Tensor([-1.0], need=True),
       Tensor([-3.0], need=True), 
       Tensor([-2.0], need=True),
       Tensor([-3.0], need=True)]

fwd.append(fwd[0] * fwd[1])
fwd.append(fwd[2] * fwd[3])
fwd.append(fwd[-2] + fwd[-1])
fwd.append(fwd[-1] + fwd[4])
fwd.append(-fwd[-1])
fwd.append((fwd[-1]).exp())
fwd.append(fwd[-1] + 1)
fwd.append(1/fwd[-1])

(fwd[-1]).back()
bwd = [t._grad_acc for t in fwd]

print(fwd)
print()
print(bwd)
