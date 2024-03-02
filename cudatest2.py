import math
import cupy as cp
import numpy as np
from time import perf_counter
from copy import copy
from abc import ABC, abstractmethod
from typing import Final
from itertools import product
from functools import reduce


def make_indices(alphabet: set[str], *ns):
    alphabet = copy(alphabet)
    return ["".join(alphabet.pop() for _ in range(n)) for n in ns]


Data_t = np.ndarray


class LazyData_t(ABC):
    shape: Final[tuple[int]]
    expr: str

    @abstractmethod
    def eval(self):
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
        args = [
            f"(item % {math.prod(self.shape[:i])}) / {math.prod(self.shape[:i-1])}"
            for i in range(1, len(self.shape) + 1)
        ]
        prog = f"""
extern "C"
__global__ void delta(float *C, const int size){{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;   
    if ( item < size ) C[item] = {self.expr(args)};
}}
"""
        delta_gpu = cp.RawKernel(prog, "delta")
        size = math.prod(self.shape)
        c_gpu = cp.zeros(size, dtype=cp.float32)

        threads_per_block = 1024
        grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
        block_size = (threads_per_block, 1, 1)

        delta_gpu(grid_size, block_size, (c_gpu, size))
        return c_gpu.reshape(self.shape)


def delta1(shape):
    dim = len(shape)
    out = np.ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out


if __name__ == "__main__":
    shape = ()
    delta = Delta(shape)
    # print(delta.prog)
    time_s = perf_counter()
    d1 = delta.eval()
    time_e = perf_counter()
    print(f"{1_000 * (time_e - time_s):.3f}ms")

    print(d1)

    # time_s = perf_counter()
    # d2 = delta1(shape)
    # time_e = perf_counter()
    # print(f"{1_000 * (time_e - time_s):.3f}ms")

    # assert np.allclose(d1, d2)
