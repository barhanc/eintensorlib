import cupy as cp
import numpy as np
import math

from typing import Final, Callable, Iterable
from abc import ABC, abstractmethod
from itertools import product
from functools import reduce

Data_t = cp.ndarray


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


class Ones(LazyData_t):
    def __init__(self, shape):
        self.shape = shape

    def expr(self, indices):
        assert len(indices) == len(self.shape)
        return "1"

    def eval(self):
        return cp.ones(self.shape, dtype=cp.float32)


def get_shape(indices: str, *Ts: Data_t | LazyData_t):
    sum_inds, free_inds = indices.split("->")
    sum_inds = sum_inds.split(",")
    assert len(sum_inds) == len(Ts), "Number of index expressions doesn't match number of operands"
    shape = [None for _ in free_inds]
    for i, free_indx in enumerate(free_inds):
        for j, T in enumerate(Ts):
            for k, sum_indx in sum_inds[j]:
                if sum_indx == free_indx and shape[i] is None:
                    shape[i] = T.shape[k]
                elif sum_indx == free_indx:
                    assert shape[i] == T.shape[k], "Axis dimensions do not match"
    for s in shape:
        assert s is not None, "One of the axis is None"
    return shape


def myeinsum(indices: str, *Ts: Data_t | LazyData_t):
    shape = get_shape(indices, *Ts)
    sum_inds, free_inds = indices.split("->")
    sum_inds = sum_inds.split(",")

    fun_sign = ", ".join(f"float *T{i}" for i, T in enumerate(Ts) if isinstance(T, Data_t))
    free_inds = f"int {', '.join(free_inds.split())} = "
    free_inds += f"""{", ".join([f"(item % {math.prod(shape[:i])}) / {math.prod(shape[:i-1])}"
            for i in range(1, len(shape) + 1)
        ])}"""
    free_inds = free_inds if shape != () else ""

    prog = f"""
extern "C"
__global__ void einsum({fun_sign}, float *Res, const int size){{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    {free_inds};
    if ( item < size ){{
      
   }}
}}
"""

    shape = None

    einsum_gpu = cp.RawKernel(prog, "vector_add")
    size = math.prod(shape)
    result = cp.zeros(shape if shape == () else size, dtype=cp.float32)

    threads_per_block = 1024
    grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
    block_size = (threads_per_block, 1, 1)


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
