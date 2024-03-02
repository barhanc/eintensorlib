import math
import cupy as cp
import numpy as np
from time import perf_counter
from string import ascii_letters
from copy import copy
from abc import ABC, abstractmethod
from typing import Final
from itertools import product
from functools import reduce
import backend as be


def make_indices(alphabet: set[str], *ns):
    alphabet = copy(alphabet)
    return ["".join(alphabet.pop() for _ in range(n)) for n in ns]


class CudaTensorTemplate(ABC):
    shape: Final[tuple[int]]
    expr: str

    @abstractmethod
    def eval(self) -> be.Delta:
        pass


class Delta(CudaTensorTemplate):
    def __init__(self, shape):
        self.shape = shape + shape
        if shape == ():
            self.expr = lambda: """1"""
        else:

            def _expr(indices):
                assert len(indices) == len(self.shape)
                return f"""({" && ".join(f"( {i} == {j} )" for i, j in zip(indices[: len(indices) // 2], indices[len(indices) // 2 :])
            )} ? 1 : 0)"""

            self.expr = _expr

    def eval(self) -> be.Delta:
        if self.shape == ():
            return np.array(1.0)

        prog = f"""
extern "C"
__global__ void delta(float * C, const int size)
{{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;   

  if ( item < size )
  {{
      C[item] = {self.expr([f"(item % {math.prod(self.shape[:i])}) / {math.prod(self.shape[:i-1])}" for i in range(1,len(self.shape) + 1)])};
  }}
}}
"""
        delta = cp.RawKernel(prog, "delta")
        size = math.prod(self.shape)
        c_gpu = cp.zeros(size, dtype=cp.float32)

        threads_per_block = 1024
        grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
        block_size = (threads_per_block, 1, 1)

        delta(grid_size, block_size, (c_gpu, size))
        c_cpu = cp.asnumpy(c_gpu).reshape(self.shape)
        return c_cpu


def delta1(shape):
    dim = len(shape)
    out = np.ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out


if __name__ == "__main__":
    shape = ()
    delta = Delta(shape)
    time_s = perf_counter()
    delta.eval()
    time_e = perf_counter()
    print(f"{1_000 * (time_e - time_s):.3f}ms")

    time_s = perf_counter()
    delta1(shape)
    time_e = perf_counter()
    print(f"{1_000 * (time_e - time_s):.3f}ms")
