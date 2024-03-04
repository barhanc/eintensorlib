# =========================================================
# Backend specific
# =========================================================
import cupy as cp
import numpy as np


Data_t = cp.ndarray


def trans(data: float | int | list | np.ndarray) -> Data_t:
    return cp.array(data)


def einEval(expr: str, *Ts: Data_t):
    pass


# =========================================================
# Backend agnostic
# =========================================================
# Y = ReLU( Ein("T(i,j) <- T(i,k) * T(k,j) + T(i)", W, X, B) )
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

    # def __mul__(self, other): if not isinstance(other, float): raise TypeError("Unsupported
    #     operand type(s) for *") return Mul(self, other)

    # def __rmul__(self, other): return Mul(self, other)

    # def __pow__(self, other): if not isinstance(other, float): raise TypeError("Unsupported
    #     operand type(s) for **") return Pow(self, other)


class Tensor(Expression):
    def __init__(self, data) -> None:
        self.data: Data_t = trans(data)
        self.shape: tuple[int] = self.data.shape
        self.partial: Data_t = None

    def eval(self) -> Data_t:
        return self.data

    def grad(self, seed: Data_t) -> None:
        pass

    def null(self) -> None:
        pass


# Einstein's notation
# ---------------------------
class Ein(Expression):
    def __init__(self, expr: str, *Ts: Expression) -> None:
        self.Ts: tuple[Expression] = Ts
        self.expr: str = expr
        self.data: Data_t = None
        self.shape: tuple[int] = None

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
