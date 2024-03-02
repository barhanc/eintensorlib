import math
import cupy as cp
from time import perf_counter
from typing import Final, Callable, Iterable
from abc import ABC, abstractmethod


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


def get_bounds(indices: str, *Ts: Data_t | LazyData_t) -> dict[str, int]:
    bounds = {indx: None for indx in set(indices) - set(",->")}
    lhs_expr = indices.split("->")[0].split(",")
    assert len(lhs_expr) == len(Ts), "Number of LHS expressions doesn't match number of operands"

    for indices, T in zip(lhs_expr, Ts):
        for i, index in enumerate(indices):
            if bounds[index] is None:
                bounds[index] = T.shape[i]
            else:
                assert bounds[index] == T.shape[i], f"Bounds at index {index} do not match"

    for bound in bounds.items():
        assert bound is not None, "One of the bounds is None"
    return bounds


def einsum(indices: str, *Ts: Data_t | LazyData_t):
    bounds = get_bounds(indices, *Ts)
    rhs_inds = indices.split("->")[1]
    lhs_inds = indices.split("->")[0].split(",")
    shape = tuple(bounds[i] for i in rhs_inds)

    print(lhs_inds, rhs_inds, bounds, shape)

    fun_sign = ", ".join(f"float *T{i}" for i, T in enumerate(Ts) if isinstance(T, Data_t))
    values = [
        f"(item % {math.prod(shape[:i])}) / {math.prod(shape[:i-1])}"
        for i in range(1, len(shape) + 1)
    ]

    free_inds_init = (
        ""
        if shape == ()
        else ", ".join(f"{rhs_idx} = {value}" for rhs_idx, value in zip(rhs_inds, values))
    )

    for_loops = "\n".join(
        f"for(long {dummy_idx}=0; {dummy_idx}<{bounds[dummy_idx]}; {dummy_idx}++){{"
        for dummy_idx in (set("".join(lhs_inds)) - set(rhs_inds))
    )
    lin_idx = lambda indices: "+ ".join(
        f"{i}*{math.prod([bounds[j] for j in indices[:n]])}" for n, i in enumerate(indices)
    )
    ein_expr = " * ".join(
        (
            f"T{n}[{lin_idx(lhs_inds[n]) if lhs_inds[n] != '' else 0}]"
            if isinstance(T, Data_t)
            else T.expr(lhs_inds[n])
        )
        for n, T in enumerate(Ts)
    )

    prog = f"""
extern "C"
__global__ void einsum({fun_sign}, float *Res, const long size){{
long item = (blockIdx.x * blockDim.x) + threadIdx.x;
long {free_inds_init};
float acc = 0;

if ( item < size ){{
{for_loops}
    acc += {ein_expr};
{"}"*len(set("".join(lhs_inds)) - set(rhs_inds))}
}}

Res[item] = acc;
}}
"""

    print(prog)
    einsum_gpu = cp.RawKernel(prog, "einsum")
    size = math.prod(shape)
    result_gpu = cp.zeros(shape if shape == () else size, dtype=cp.float32)

    threads_per_block = 1024
    grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
    block_size = (threads_per_block, 1, 1)
    einsum_gpu(grid_size, block_size, (*[T for T in Ts if isinstance(T, Data_t)], result_gpu, size))

    return result_gpu.reshape(shape)


if __name__ == "__main__":
    T1 = cp.ones(10, dtype=cp.float32)
    a = cp.array(2, dtype=cp.float32)
    delta = Delta((10,))

    time_s = perf_counter()
    result = einsum(",ij,j->i", a, delta, T1)
    time_e = perf_counter()
    print(f"{1_000 *(time_e - time_s):.2f}ms")
    print(result)
    print(cp.einsum(",ij,j->i", a, delta.eval(), T1))
