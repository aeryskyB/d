import jax.numpy as jnp
from jax import grad

x = jnp.array([[1.], [2.]])
y = jnp.array([[1.], [0.]])

W = jnp.array([[1., 2.], [3., 4.]])
lr = 4e-2

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
    l = loss(W)
    print(l)
    W_grad = grad(loss, argnums=0)(W)
    W = W - lr * W_grad

print(W)
