import eintensor.backend as be
import numpy as np
from itertools import product
from eintensor.tensor import Tensor, einsum, relu
from copy import deepcopy

np.set_printoptions(precision=2)
I = lambda k: Tensor(be.ones(k))

# Nin, B -> Nout, B
F1 = lambda X, W, B: relu(einsum("ij,jk->ik", W, X) + einsum("i,k->ik", B, I(X.shape[-1])))
F2 = lambda X, W, B: einsum("ij,jk->ik", W, X) + einsum("i,k->ik", B, I(X.shape[-1]))
F = lambda X, W1, W2, B1, B2: F2(F1(X, W1, B1), W2, B2)
loss_fn = lambda X, W1, W2, B1, B2: einsum("ij,ij->", F(X, W1, W2, B1, B2), F(X, W1, W2, B1, B2))

X = Tensor(be.random((4, 16)))
W1 = Tensor(be.random((6, 4)))
B1 = Tensor(be.random(6))
W2 = Tensor(be.random((3, 6)))
B2 = Tensor(be.random(3))

loss = loss_fn(X, W1, W2, B1, B2)
print("Autograd loss: ", loss.eval())

loss.grad()
print("dL/dX =\n", X.partial, "\n")
print("dL/dW1=\n", W1.partial, "\n")
print("dL/dW2=\n", W2.partial, "\n")
print("dL/dB1=\n", B1.partial, "\n")
print("dL/dB2=\n", B2.partial, "\n")
# loss.null()

eps = 1e-6
for name, param in zip(("X", "W1", "W2", "B1", "B2"), (X, W1, W2, B1, B2)):
    dLdP = be.zeros(param.shape)
    params = {"X": X, "W1": W1, "W2": W2, "B1": B1, "B2": B2}
    for i in product(*[range(n) for n in param.shape]):
        M = be.zeros(param.shape)
        M[i] = 1
        p1 = deepcopy(params)
        p2 = deepcopy(params)
        p1[name] = param + eps * Tensor(M)
        p2[name] = param - eps * Tensor(M)
        dLdP[i] = loss_fn(**p1).eval() - loss_fn(**p2).eval()
        dLdP[i] /= 2 * eps

    print(dLdP, "\n")
    # if not be.allclose(param.partial, dLdP, atol=eps):
    # print(param.partial, "\n")
    # print(dLdP == param.partial)
