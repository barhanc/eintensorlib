import symbolic

# =========================================================
# Backend specific functions
# =========================================================
import numpy as np
import cupy as cp
import math

Data_t = cp.ndarray


def array(data) -> Data_t:
    return cp.array(data, dtype=float)


def ein_eval(expr: str, *Ts: Data_t, bounds: dict[str, int], threads_per_block=128) -> Data_t:
    program = symbolic.prog_cuda(expr, bounds, [T.shape for T in Ts])
    print(program)
    program_gpu = cp.RawKernel(program, "ein")
    out_shape = symbolic.get_output_shape(expr, bounds)
    size = math.prod(out_shape)
    out_gpu = cp.zeros(size, dtype=float)

    grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
    block_size = (threads_per_block, 1, 1)

    program_gpu(grid_size, block_size, (*[T.flatten("F") for T in Ts], out_gpu))
    return out_gpu.reshape(out_shape, order="F")


def delta(shape: tuple[int]) -> Data_t:
    args_a = symbolic.make_indices(len(shape))
    args_b = symbolic.make_indices(len(shape), set(args_a))
    e = f"T({','.join(args_a)},{','.join(args_b)})<-{symbolic.delta_prod(args_a, args_b)}"
    return ein_eval(e, bounds=dict(zip(args_a + args_b, shape + shape)))


# =========================================================
# Backend agnostic classes abstractions
# =========================================================
from typing import Callable
from abc import ABC, abstractmethod


class Expression(ABC):
    data: Data_t
    shape: tuple[int]

    @abstractmethod
    def eval(self) -> Data_t:
        pass

    @abstractmethod
    def grad(self, seed: Data_t) -> None:
        pass

    @abstractmethod
    def null(self) -> None:
        pass

    def __add__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("Unsupported operand type(s) for +")
        return Add(self, other)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError("Unsupported operand type(s) for *")
        return Mul(self, other)

    def __rmul__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError("Unsupported operand type(s) for *")
        return Mul(self, other)

    def __sub__(self, other):
        if not isinstance(other, Expression):
            raise TypeError("Unsupported operand type(s) for -")
        return self + -1 * other

    def __pow__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError("Unsupported operand type(s) for **")
        return Pow(self, other)


class Tensor(Expression):
    def __init__(self, data) -> None:
        self.data: Data_t = array(data)
        self.shape: tuple[int] = self.data.shape
        self.partial: Data_t = None

    def eval(self) -> Data_t:
        return self.data

    def grad(self, seed: Data_t = None) -> None:
        seed = delta(self.shape) if seed is None else seed
        self.partial = seed if self.partial is None else self.partial + seed

    def null(self) -> None:
        self.partial = None


# ---------------------------------------------------------
# Tensor addition
# ---------------------------------------------------------
class Add(Expression):
    def __init__(self, T1: Expression, T2: Expression):
        assert T1.shape == T2.shape
        self.T1, self.T2 = T1, T2
        self.data: Data_t = None
        self.shape = T1.shape

    def eval(self) -> Data_t:
        self.data = self.T1.eval() + self.T2.eval() if self.data is None else self.data
        return self.data

    def grad(self, seed: Data_t = None) -> None:
        seed = delta(self.shape) if seed is None else seed
        self.T1.grad(seed)
        self.T2.grad(seed)

    def null(self) -> None:
        self.data = None
        self.T1.null()
        self.T2.null()


# ---------------------------------------------------------
# Multiplication by a scalar
# ---------------------------------------------------------
class Mul(Expression):
    def __init__(self, T: Expression, a: float | int):
        self.T, self.a = T, a
        self.data: Data_t = None
        self.shape = T.shape

    def eval(self):
        self.data = self.a * self.T.eval() if self.data is None else self.data
        return self.data

    def grad(self, seed: Data_t = None):
        seed = delta(self.shape) if seed is None else seed
        self.T.grad(self.a * seed)

    def null(self):
        self.data = None
        self.T.null()


# ---------------------------------------------------------
# Einstein's summation notation
# ---------------------------------------------------------
class Ein(Expression):
    def __init__(self, expr: str, *Ts: Expression, bounds: dict[str, int]) -> None:
        self.Ts: tuple[Expression] = Ts
        self.expr: str = expr
        self.data: Data_t = None
        self.shape: tuple[int] = symbolic.get_output_shape(expr, bounds)
        self.bounds: dict[str] = bounds

    def eval(self) -> Data_t:
        if self.data is None:
            self.data = ein_eval(self.expr, *[T.eval() for T in self.Ts], bounds=self.bounds)
        return self.data

    # TODO: Delta reductions
    def grad(self, seed: Data_t = None) -> None:
        seed = delta(self.shape) if seed is None else seed

        for id, (T, (grad_args, free_args, grad_expr)) in enumerate(zip(self.Ts, symbolic.derive(self.expr))):
            _, rhs = grad_expr.split("<-")
            used_args = set(grad_args) | set(free_args) | set(self.bounds.keys())
            args = symbolic.make_indices(len(seed.shape) - len(free_args), used_args)

            expr = f"T({','.join(grad_args.split(','))},{','.join(args)})"
            expr += f" <- T({','.join(free_args)},{','.join(args)}){'*'+rhs if len(rhs) > 0 else ''}"
            bounds = {
                **self.bounds,
                **dict(zip(grad_args.split(","), T.shape)),
                **dict(zip(args, seed.shape[-len(args) :])),
            }
            print(expr, bounds)

            new_seed = ein_eval(expr, seed, *[T.eval() for T in self.Ts[:id] + self.Ts[id + 1 :]], bounds=bounds)
            T.grad(new_seed)

    def null(self) -> None:
        self.data = None
        for T in self.Ts:
            T.null()


# ---------------------------------------------------------
# Element-wise function
# ---------------------------------------------------------
class Elf(Expression):
    f: Callable[[Data_t], Data_t]
    df: Callable[[Data_t], Data_t]

    def __init__(self, T: Expression):
        self.T = T
        self.data: Data_t = None
        self.shape = T.shape

    def eval(self) -> Data_t:
        self.data = self.f(self.T.eval()) if self.data is None else self.data
        return self.data

    def grad(self, seed: Data_t = None) -> None:
        seed = delta(self.shape) if seed is None else seed
        i = ",".join(symbolic.make_indices(len(seed.shape) - len(self.shape)))
        j = ",".join(symbolic.make_indices(len(self.shape), set(i)))

        expr = f"T({j},{i}) <- T({j},{i}) * T({j})"
        bounds = {
            **dict(zip(j.split(","), self.shape)),
            **dict(zip(i.split(","), seed.shape[-len(i.split(",")) :])),
        }
        new_seed = ein_eval(expr, seed, self.df(self.T.eval()), bounds=bounds)
        self.T.grad(new_seed)

    def null(self):
        self.data = None
        self.T.null()


# =============================================================================
# =============================================================================


class Pow(Elf):
    def __init__(self, T: Expression, e: float | int):
        self.f = lambda x: x**e
        self.df = lambda x: e * x ** (e - 1)
        super().__init__(T)


class Exp(Elf):
    def __init__(self, T: Expression):
        self.f = lambda x: cp.exp(x)
        self.df = lambda x: cp.exp(x)
        super().__init__(T)


class Log(Elf):
    def __init__(self, T: Expression):
        self.f = lambda x: cp.log(x)
        self.df = lambda x: 1 / x
        super().__init__(T)


class ReLU(Elf):
    def __init__(self, T: Expression):
        self.f = lambda x: cp.maximum(0, x)
        self.df = lambda x: cp.heaviside(x, 0)
        super().__init__(T)


if __name__ == "__main__":
    A = Tensor(np.ones((4000, 4000)))
    X = Tensor(np.ones(4000))
    C = Tensor([2])
    Y = Ein("T(i) <- T(i,j) * T(j) * T(0)", A, X, C, bounds={"i": 4000, "j": 4000})
    l = Ein("T(i) <- T(j)", Y**2, bounds={"i": 1, "j": 4000})
    print(l.eval())
    l.grad()
    # print(X.partial)
    print(A.partial[:, :, 0])
    print(C.partial)
    # X = Tensor([1])
    # Y = 0.5 * X**2
    # print(Y.eval())
    # Y.grad()
    # print(X.partial)
    # Y.null()
    # print(X.partial, Y.data)
    # print(Y.eval())
    # print(X.partial)
