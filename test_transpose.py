from base.tensor import Tensor

a = Tensor([[1, 2]])
print(a.transpose() @ a)
