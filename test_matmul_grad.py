from base.tensor import Tensor

A = Tensor([[1, 2], [3, 4]], need=True)
B = Tensor([[1], [2]], need=True)
C = A @ B
D = C.sum()

print('-'*30)
print('A =')
print(A)
print('-'*30)
print('B =')
print(B)
print('-'*30)
print('C = AB')
print(C)
print('-'*30)
print('D = C.sum()')
print(D)

print('-'*30)
print("Before backprop:\n")
print("Accumulated gradient at A")
print(A._grad_acc)
print("Accumulated gradient at B")
print(B._grad_acc)

print('-'*30)
D.back()
print("After backprop:\n")
print("Accumulated gradient at A")
print(A._grad_acc)
print("Accumulated gradient at B")
print(B._grad_acc)

print('-'*30)
D.reset_grad()
print("After cleaning accumulated gradients:\n")
print("Accumulated gradient at A")
print(A._grad_acc)
print("Accumulated gradient at B")
print(B._grad_acc)

