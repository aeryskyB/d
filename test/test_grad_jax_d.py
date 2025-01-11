import numpy as np
import jax.numpy as jnp
from jax import grad
from base.tensor import Tensor

lr = 4e-2

#################################################

x = Tensor([[1], [2]])
y = Tensor([[1], [0]])

W1 = Tensor([[1, 2], [3, 4]], need=True)


for i in range(10):
    y_ = W1 @ x
    y_d = (y - y_)
    y_d_s = y_d ** 2
    y_d_ms = y_d_s / y_.len()
    loss = y_d_ms.sum()
    # print(loss)
    loss.back()
    W1.update_decr(lr * W1._grad_acc)
    loss.reset_grad()

#################################################

x = jnp.array([[1.], [2.]])
y = jnp.array([[1.], [0.]])

W2 = jnp.array([[1., 2.], [3., 4.]])

def predict(W):
    y_ = W @ x
    return y_

def loss(W):
    y_ = predict(W)
    y_d = jnp.subtract(y, y_)
    y_d_s = jnp.power(y_d, 2)
    y_d_ms = jnp.true_divide(y_d_s, len(y_))
    loss = jnp.sum(y_d_ms)
    return loss

for i in range(10):
    l = loss(W2)
    # print(l)
    W_grad = grad(loss, argnums=0)(W2)
    W2 = W2 - lr * W_grad

#################################################

assert np.allclose(W1.numpy(), W2.__array__())
