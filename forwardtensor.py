import numpy as np  # for now
from typing import Callable, Final
from itertools import product
from functools import reduce
from string import ascii_letters


# Forward differentiation scheme. Not really useful as caluclating derivatives of all parameters is
# O(n*p) where p is the number of parameters and n is the size of the computation graph. You can
# however easily calculate a tensor derivative od arbitrary order.


class Tensor:
    def __init__(
        self,
        data: np.ndarray | None = None,
        shape: tuple[int] | None = None,
        eval: Callable | None = None,
        grad: Callable | None = None,
        clear: Callable | None = None,
    ):
        if data is not None:
            self.data = data
            self.shape: Final = data.shape
            self.isvar: Final = True

            def _grad(X: Tensor):
                assert X.isvar, "You can only differentiate by independent variable Tensors"
                return delta(self.shape) if X is self else const(0.0, X.shape + self.shape)

            self.eval = lambda: self.data
            self.grad = _grad
            self.clear = lambda: None

        if data is None:
            self.data = None
            self.shape: Final = shape
            self.isvar: Final = False

            def _grad(X: Tensor):
                assert X.isvar, "You can only differentiate by independent variable Tensors"
                return grad(X)

            def _eval():
                if self.data is None:
                    print("Calculating")
                    self.data = eval()
                return self.data

            def _clear():
                clear()
                self.data = None

            self.eval = _eval
            self.grad = _grad
            self.clear = _clear

    def __str__(self) -> str:
        if self.data is not None:
            return str(self.data)
        return "Tensor not evaluated. Call .eval() method first."

    def __add__(self, other):
        if isinstance(other, Tensor):
            return add(self, other)
        raise TypeError("Unsupported operand type(s) for +")

    def __mul__(self, other):
        if isinstance(other, float):
            return mul(self, other)
        raise TypeError("Unsupported operand type(s) for *")

    def __rmul__(self, other):
        return mul(self, other)


# Tensor initialization
def const(value: float, shape: tuple[int]):
    return Tensor(data=value * np.ones(shape))


def randf(shape):
    return Tensor(data=np.random.rand(*shape))


def delta(shape):
    dim = len(shape)
    out = np.zeros(shape + shape)
    for x in product(*[range(n) for n in shape + shape]):
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return Tensor(data=out)


def randf(shape):
    return Tensor(data=np.random.rand(*shape))


# UnaryOps
def mul(A: Tensor, a: float):
    return Tensor(
        shape=A.shape,
        eval=lambda: a * A.eval(),
        grad=lambda X: a * A.grad(X),
        clear=A.clear,
    )


# BinaryOps
def add(A: Tensor, B: Tensor):
    assert A.shape == B.shape, "You can only add tensors of the same shape."

    def _clear():
        A.clear()
        B.clear()

    return Tensor(
        shape=A.shape,
        eval=lambda: A.eval() + B.eval(),
        grad=lambda X: A.grad(X) + B.grad(X),
        clear=_clear,
    )


def einsum(expr: str, A: Tensor, B: Tensor):
    indices = expr.split(",")
    indices = indices[:-1] + indices[-1].split("->")
    letters = set(ascii_letters) - set("".join(indices))

    def _grad(X: Tensor):
        n = len(X.shape)
        assert len(letters) >= n, "Not enough letters :("

        prefix = "".join(list(letters)[:n])
        L = einsum(f"{prefix+indices[0]},{indices[1]}->{prefix+indices[2]}", A.grad(X), B)
        R = einsum(f"{indices[0]},{prefix+indices[1]}->{prefix+indices[2]}", A, B.grad(X))
        return L + R

    def _clear():
        A.clear()
        B.clear()

    shape = []
    for x in indices[-1]:
        for i, y in enumerate(indices[0]):
            if x == y:
                shape.append(A.shape[i])
        for i, y in enumerate(indices[1]):
            if x == y:
                shape.append(B.shape[i])

    return Tensor(
        shape=tuple(shape),
        eval=lambda: np.einsum(expr, A.eval(), B.eval()),
        grad=_grad,
        clear=_clear,
    )


# Reshape
def reshape(expr: str, A: Tensor):
    pass


if __name__ == "__main__":
    W1 = Tensor(data=np.array([[0.5, 0.3], [1.2, 1.1]]))
    print("var W1 = \n", W1, "\n")

    X = Tensor(data=np.array([0.2, 0.1]))
    print("var X = \n", X, "\n")

    Y = einsum("ik,k->i", W1, X)
    print("xpr Y = W1.ik * X.k =\n", Y.eval(), "\n")

    W2 = Tensor(data=np.array([[2.0, 0.0], [0.0, 2.0]]))
    print("var W2 = \n", W2, "\n")

    F = einsum("ik,k->i", W2, Y)
    print("xpr F = W2.ik * Y.k\n", F.eval(), "\n")

    print("dF/dW2 = ", F.grad(W2).eval(), "\n")
    print("dF/dW1 = ", F.grad(W1).eval(), "\n")
