# d
This is my toy deep learning library 🤍  

Have a nice day!

# Usage
[`test_grad_d.py`](./test_grad_d.py):
```python
from base.tensor import Tensor, l

x = Tensor([[1], [2]])
y = Tensor([[1], [0]])

W = Tensor([[1, 2], [3, 4]], need=True)

lr = 4e-2

for i in range(10):
    y_ = W @ x
    y_d = (y - y_)
    y_d_s = y_d ** 2
    y_d_ms = y_d_s / y_.len()
    loss = y_d_ms.sum()
    print(loss)
    loss.back()
    W.update_decr(lr * W._grad_acc)
    loss.reset_grad()

print(W)
```

# Inspirations
1. tinygrad
2. jax
3. pytorch