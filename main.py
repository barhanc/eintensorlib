import backend as be
from tensor import Tensor, einsum, relu

from itertools import product
from copy import deepcopy
from time import perf_counter

# be.set_printoptions(precision=8)
be.random.seed(0)
I = lambda shape: Tensor(be.ones(shape, dtype=float))

# Nin, B -> Nout, B
F1 = lambda X, W, B: relu(einsum("ij,jk->ik", W, X) + einsum("i,k->ik", B, I(X.shape[-1])))
F2 = lambda X, W, B: einsum("ij,jk->ik", W, X) + einsum("i,k->ik", B, I(X.shape[-1]))
F = lambda X, W1, W2, B1, B2: F2(F1(X, W1, B1), W2, B2)
loss_fn = lambda X, W1, W2, B1, B2: einsum(
    "ij,ij->", F(X, W1, W2, B1, B2) ** 2.0, I((W2.shape[0], X.shape[1]))
)

X = Tensor(be.random.normal(size=(100, 64)))
W1 = Tensor(be.random.normal(size=(128, 100)))
B1 = Tensor(be.random.normal(size=(128,)))
W2 = Tensor(be.random.normal(size=(64, 128)))
B2 = Tensor(be.random.normal(size=(64,)))

time_s = perf_counter()
loss = loss_fn(X, W1, W2, B1, B2)
time_e = perf_counter()
print(f"Forward pass: {1_000 * (time_e - time_s):.2f}ms", "Loss: ", loss.eval())

time_s = perf_counter()
loss.grad()
time_e = perf_counter()
print(f"Backward pass: {1_000 * (time_e - time_s):.2f}ms")
# print("dL/dX =\n", X.partial, "\n")
# print("dL/dW1=\n", W1.partial, "\n")
# print("dL/dW2=\n", W2.partial, "\n")
# print("dL/dB1=\n", B1.partial, "\n")
# print("dL/dB2=\n", B2.partial, "\n")
# loss.null()
print("")

# print("Check numeric gradients . . .")
# eps = 1e-4
# for name, param in zip(("X", "W1", "W2", "B1", "B2"), (X, W1, W2, B1, B2)):
#     dLdP = be.zeros(param.shape, dtype=float)
#     params = {"X": X, "W1": W1, "W2": W2, "B1": B1, "B2": B2}
#     M = be.zeros(param.shape, dtype=float)
#     for i in product(*[range(n) for n in param.shape]):
#         M[i] = 1
#         p1 = deepcopy(params)
#         p2 = deepcopy(params)
#         p1[name] = Tensor((param.eval() + eps * M))
#         p2[name] = Tensor((param.eval() - eps * M))

#         dLdP[i] = loss_fn(**p1).eval() - loss_fn(**p2).eval()
#         dLdP[i] /= 2 * eps
#         M[i] = 0
#     # print(dLdP)
#     print(be.allclose(param.partial, dLdP, atol=eps), "\n")
#     # print(be.abs(param.partial - dLdP))
#     # print(param.partial, "\n")
