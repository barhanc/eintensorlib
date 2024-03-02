import cupy as cp
import numpy as np
import math

from typing import Final, Callable
from abc import ABC, abstractmethod
from itertools import product
from functools import reduce

Data_t = np.ndarray


class LazyData_t(ABC):
    shape: Final[tuple[int]]

    @abstractmethod
    def expr(self, indices: list[str]) -> str:
        pass

    @abstractmethod
    def eval(self) -> Data_t:
        pass


class Delta(LazyData_t):
    def __init__(self, shape):
        self.shape = shape + shape

    def expr(self, indices):
        assert (n := len(indices)) == len(self.shape)
        if self.shape == ():
            return "1"
        double_indices = zip(indices[: n // 2], indices[n // 2 :])
        return f"""({" && ".join(f"( {i} == {j} )" for i, j in double_indices)} ? 1 : 0)"""

    def eval(self):
        if self.shape == ():
            return cp.array(1, dtype=cp.float32)

        args = [
            f"(item % {math.prod(self.shape[:i])}) / {math.prod(self.shape[:i-1])}"
            for i in range(1, len(self.shape) + 1)
        ]
        prog = f"""
extern "C"
__global__ void delta(float * C, const int size){{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;   
  if ( item < size ) C[item] = {self.expr(args)};
}}
"""
        delta = cp.RawKernel(prog, "delta")
        size = math.prod(self.shape)
        c_gpu = cp.zeros(size, dtype=cp.float32)

        threads_per_block = 1024
        grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
        block_size = (threads_per_block, 1, 1)

        delta(grid_size, block_size, (c_gpu, size))
        return c_gpu.reshape(self.shape)


class Ones(LazyData_t):
    def __init__(self, shape):
        self.shape = shape

    def expr(self, indices):
        assert len(indices) == len(self.shape)
        return "1"

    def eval(self):
        return cp.ones(self.shape, dtype=cp.float32)


def myeinsum(indices: str, *Xs):
    pass


zeros = np.zeros
ones = np.ones
array = np.array
random = np.random.rand
allclose = np.allclose

heaviside = np.heaviside
max = np.maximum
einsum = np.einsum


def delta(shape):
    dim = len(shape)
    out = np.ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out
