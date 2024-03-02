import cupy as cp
import numpy as np
import math

from typing import Final
from functools import reduce
from itertools import product
from abc import ABC, abstractmethod

Data_t = np.ndarray


class CudaTensorTemplate(ABC):
    shape: Final[tuple[int]]
    prog: str

    @abstractmethod
    def eval(self):
        pass


zeros = np.zeros
ones = np.ones
array = np.array
random = lambda shape: np.random.normal(0.0, 1.0, shape)
allclose = np.allclose

heaviside = np.heaviside
max = np.maximum
einsum = np.einsum


# TODO: This implementation is too slow. Delta shouldn't be realized unless in seed. It should be a
# function N^k -> {0,1} used in einsum (How to make einsum which supports functions <- custom cuda
# kernel?)
def delta(shape):
    dim = len(shape)
    out = ones(shape + shape)
    for x in product(*[range(n) for n in shape + shape]) if dim > 0 else []:
        out[x] = 1 if reduce(lambda a, b: a and b, [x[i] == x[dim + i] for i in range(dim)]) else 0
    return out


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

    def eval(self) -> Data_t:
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