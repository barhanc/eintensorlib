# =========================================================
# Backend specific functions
# =========================================================
import cupy as cp
import symbolic
import math

Data_t = cp.ndarray


def array(data) -> Data_t:
    return cp.array(data)


def ein_eval(expr: str, *Ts: Data_t, bounds: dict[str, int], threads_per_block=128) -> Data_t:
    program = symbolic.prog_cuda(expr, bounds, [T.shape for T in Ts])
    # print(program)
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
# Backend agnostic classes/functions
# =========================================================
# Maybe not. Lets just stick with Add + Ein and *explicit* declaration of bounds of indices(sic!).
# This way the whole implementation with CUDA backend can be one ~250loc file and we can eaily
# support arthimetic expressions for indices e.g. "T(a,b,c) <- T(5*a + 2*i, 5*b + 2*j, k) *
# T(i,j,k,c)" and in this way also reshape operations

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

    # def __pow__(self, other):
    #     if not isinstance(other, float):
    #         raise TypeError("Unsupported operand type(s) for **")
    #     return Pow(self, other)


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
        self.T = T
        self.a = a
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
        return ein_eval(self.expr, *[T.eval() for T in self.Ts], self.bounds)

    def grad(self, seed: Data_t = None) -> None:
        seed = delta(self.shape) if seed is None else seed
        Tid = 0
        for T, (grad_args, free_args, grad_expr) in zip(self.Ts, symbolic.derive(self.expr)):
            _, rhs = grad_expr.split("<-")
            new_free_args = symbolic.make_indices(
                len(seed.shape) - len(free_args),
                set(grad_args) | set(free_args) | set(self.bounds.keys()),
            )
            expr = f"T({','.join(grad_args)},{','.join(new_free_args)}) <- T({','.join(free_args)},{','.join(new_free_args)}){'*'+rhs if len(rhs)>0 else ''}"
            bounds = {
                **self.bounds,
                **dict(zip(grad_args, T.shape)),
                **dict(zip(new_free_args, seed.shape[-len(new_free_args) :])),
            }
            T.grad(ein_eval(expr, seed, *(self.Ts[:Tid] + self.Ts[Tid + 1 :]), bounds))
            Tid += 1

    def null(self) -> None:
        self.data = None
        for T in self.Ts:
            T.null()


# ---------------------------------------------------------
# Element-wise function
# ---------------------------------------------------------
class Elf(Expression):
    f: Callable
    df: Callable
    pass


if __name__ == "__main__":
    print(delta((5,)))
