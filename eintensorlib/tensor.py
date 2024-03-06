# =========================================================
# Backend specific
# =========================================================
import cupy as cp
import numpy as np


Data_t = cp.ndarray


def array(data) -> Data_t:
    return cp.array(data)


def einEval(expr: str, *Ts: Data_t, bounds: dict[str, int]):
    pass


# =========================================================
# Backend agnostic
# =========================================================
from typing import Callable
from abc import ABC, abstractmethod

# Maybe not. Lets just stick with Add + Ein and *explicit* declaration of bounds of indices(sic!).
# This way the whole implementation with CUDA backend can be one ~250loc file and we can eaily
# support arthimetic expressions for indices e.g. "T(a,b,c) <- T(5*a + 2*i, 5*b + 2*j, k) *
# T(i,j,k,c)"


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

    # def __add__(self, other):
    #     if not isinstance(other, Expression):
    #         raise TypeError("Unsupported operand type(s) for +")
    #     return Add(self, other)

    # def __mul__(self, other):
    #     if not isinstance(other, float):
    #         raise TypeError("Unsupported operand type(s) for *")
    #     return Mul(self, other)

    # def __rmul__(self, other):
    #     return Mul(self, other)

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

    def grad(self, seed: Data_t) -> None:
        pass

    def null(self) -> None:
        pass


# Addition
# ---------------------------
class Add(Expression):
    def __init__(self, T1: Expression, T2: Expression):
        assert T1.shape == T2.shape
        self.T1, self.T2 = T1, T2
        self.shape = T1.shape

    def eval(self) -> Data_t:
        return self.T1.eval() + self.T2.eval()

    def grad(self, seed: Data_t) -> None:
        pass

    def null(self) -> None:
        pass


# Einstein's summation notation
# ---------------------------
class Ein(Expression):
    def __init__(self, expr: str, *Ts: Expression, bounds: dict[str, int]) -> None:
        self.Ts: tuple[Expression] = Ts
        self.expr: str = expr
        self.data: Data_t = None
        self.shape: tuple[int] = None
        self.bounds: dict[str] = bounds

    def eval(self) -> Data_t:
        pass

    def grad(self, seed: Data_t) -> None:
        pass

    def null(self) -> None:
        pass


# Elementwise function
# ---------------------------
class Elf(Expression):
    f: Callable
    df: Callable
    pass
