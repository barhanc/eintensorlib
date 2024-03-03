# import backend as be
import backend_einsum as be
from abc import abstractmethod, ABC
from typing import Final, Callable
from string import ascii_letters
from copy import copy
from itertools import combinations


class Expression(ABC):
    data: be.Data_t
    shape: Final[tuple[int]]

    @abstractmethod
    def eval(self) -> be.Data_t:
        pass

    @abstractmethod
    def grad(self, seed: be.Data_t) -> None:
        pass

    @abstractmethod
    def null(self) -> None:
        pass

    def __add__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("Unsupported operand type(s) for +")
        return Add(self, other)

    def __mul__(self, other):
        if not isinstance(other, float):
            raise TypeError("Unsupported operand type(s) for *")
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("Unsupported operand type(s) for -")
        return self + -1 * other

    def __pow__(self, other):
        if not isinstance(other, float):
            raise TypeError("Unsupported operand type(s) for **")
        return Pow(self, other)


class Tensor(Expression):
    def __init__(self, data: be.Data_t):
        self.data: be.Data_t = data
        self.shape = data.shape
        self.partial: be.Data_t = None

    def eval(self):
        # print("leaf")
        return self.data

    def grad(self, seed: be.Data_t = None):
        seed = seed if seed is not None else be.delta(self.shape)
        self.partial = self.partial + seed if self.partial is not None else seed

    def null(self):
        self.partial = None


# -----------------------------------
# Binary Ops
# -----------------------------------


class Add(Expression):
    def __init__(self, A: Expression, B: Expression):
        self.A = A
        self.B = B

        assert A.shape == B.shape, "You can only add tensors of the same shape"
        self.data: be.Data_t = None
        self.shape = A.shape

    def eval(self):
        self.data = self.A.eval() + self.B.eval() if self.data is None else self.data
        return self.data

    def grad(self, seed: be.Data_t = None):
        seed = seed if seed is not None else be.delta(self.shape)
        self.A.grad(seed)
        self.B.grad(seed)

    def null(self):
        self.data = None
        self.A.null()
        self.B.null()


class Einsum(Expression):
    def __init__(self, indices: str, A: Expression, B: Expression):
        self.A = A
        self.B = B
        self.indices: str = indices

        self.data: be.Data_t = None
        self.shape = get_shape(indices, A.shape, B.shape)

    def eval(self):
        self.data = (
            be.einsum(self.indices, self.A.eval(), self.B.eval())
            if self.data is None
            else self.data
        )
        return self.data

    # TODO: It can be made faster with delta reduction
    def grad(self, seed: be.Data_t = None):
        seed = seed if seed is not None else be.delta(self.shape)
        alphabet = set(ascii_letters) - set(self.indices)
        ij, k = self.indices.split("->")

        i, j = ij.split(",")
        a, b = make_indices(alphabet, len(seed.shape) - len(self.shape), len(self.A.shape))
        print(f"{k+a},{b+i},{j}->{b+a}")

        j_ = j.translate(str.maketrans({k: v for k, v in zip(i, b)}))
        k_ = k.translate(str.maketrans({k: v for k, v in zip(i, b)}))
        equal = ["".join(nb for nb, ni in zip(b, i) if ni == n) for n in set(i)]
        print(f"{k_+a},{equal},{j_}->{b+a}")

        new_seed = be.einsum(f"{k+a},{b+i},{j}->{b+a}", seed, be.Delta(self.A.shape), self.B.eval())
        self.A.grad(new_seed)

        i, j = ij.split(",")
        a, b = make_indices(alphabet, len(seed.shape) - len(self.shape), len(self.B.shape))
        print(f"{k+a},{i},{b+j}->{b+a}")
        new_seed = be.einsum(f"{k+a},{i},{b+j}->{b+a}", seed, self.A.eval(), be.Delta(self.B.shape))
        self.B.grad(new_seed)

    def null(self):
        self.data = None
        self.A.null()
        self.B.null()


# -----------------------------------
# Unary Ops
# -----------------------------------


class Mul(Expression):
    def __init__(self, A: Expression, scalar: float):
        self.A = A
        self.scalar = scalar

        self.data: be.Data_t = None
        self.shape = A.shape

    def eval(self):
        self.data = self.scalar * self.A.eval() if self.data is None else self.data
        return self.data

    def grad(self, seed: be.Data_t = None):
        seed = seed if seed is not None else be.delta(self.shape)
        self.A.grad(self.scalar * seed)

    def null(self):
        self.data = None
        self.A.null()


class Function1D(Expression):
    f: Callable[[float], float]
    df: Callable[[float], float]

    def __init__(self, A: Expression):
        self.A = A
        self.data: be.Data_t = None
        self.shape = A.shape

    def eval(self):
        self.data = self.f(self.A.eval()) if self.data is None else self.data
        return self.data

    def grad(self, seed: be.Data_t = None):
        seed = seed if seed is not None else be.delta(self.shape)
        a, b = make_indices(set(ascii_letters), len(seed.shape) - len(self.shape), len(self.shape))
        new_seed = be.einsum(f"{b+a},{b}->{b+a}", seed, self.df(self.A.eval()))
        self.A.grad(new_seed)

    def null(self):
        self.data = None
        self.A.null()


class Pow(Function1D):
    def __init__(self, A: Expression, e: float):
        self.f = lambda x: x**e
        self.df = lambda x: e * x ** (e - 1)
        super().__init__(A)


class ReLU(Function1D):
    def __init__(self, A: Expression):
        self.f = lambda x: be.max(0, x)
        self.df = lambda x: be.heaviside(x, 0)
        super().__init__(A)


# -----------------------------------
# Reshape Ops
# -----------------------------------


class Reshape(Expression):
    pass


# -----------------------------------
# Utils
# -----------------------------------


# TODO: Make it nicer
def get_shape(indices: str, sh1: tuple[int], sh2: tuple[int]):
    xys, zs = indices.split("->")
    xs, ys = xys.split(",")
    shape = [None for _ in zs]
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            if x == z and shape[i] is None:
                shape[i] = sh1[j]
            elif x == z:
                assert shape[i] == sh1[j], f"Not matching dimensions {j}"
        for j, y in enumerate(ys):
            if y == z and shape[i] is None:
                shape[i] = sh2[j]
            elif y == z:
                assert shape[i] == sh2[j], f"Not matching dimensions {j}"

    for s in shape:
        assert s is not None, "One of the axis is None"

    return tuple(shape)


def make_indices(alphabet: set[str], *ns):
    alphabet = copy(alphabet)
    return ["".join(alphabet.pop() for _ in range(n)) for n in ns]
