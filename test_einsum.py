import math
import cupy as cp
import numpy as np
from backend import Data_t, LazyData_t, Delta
from time import perf_counter


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

    fun_sign = ", ".join(f"float *T{i}" for i, T in enumerate(Ts) if isinstance(T, Data_t))
    declar_free_inds = f"long {', '.join(rhs_inds)} = " if shape != () else ""
    assign_free_inds = (
        ", ".join(
            f"(item % {math.prod(shape[:i])}) / {math.prod(shape[:i-1])}"
            for i in range(1, len(shape) + 1)
        )
        if shape != ()
        else ""
    )
    for_loops = "\n".join(
        f"for(long {dummy_idx}=0; {dummy_idx}<{bounds[dummy_idx]}; {dummy_idx}++){{"
        for dummy_idx in (set("".join(lhs_inds)) - set(rhs_inds))
    )
    lin_idx = lambda indices: "+ ".join(
        f"{i}*{math.prod([bounds[j] for j in indices[:n]])}" for n, i in enumerate(indices)
    )
    ein_expr = " * ".join(
        f"T{n}[{lin_idx(lhs_inds[n])}]" if isinstance(T, Data_t) else T.expr(lhs_inds[n])
        for n, T in enumerate(Ts)
    )

    prog = f"""
extern "C"
__global__ void einsum({fun_sign}, float *Res, const long size){{
long item = (blockIdx.x * blockDim.x) + threadIdx.x;
{declar_free_inds}{assign_free_inds};
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
    T1 = cp.ones(100_000, dtype=cp.float32)
    delta = Delta((100_000,))

    time_s = perf_counter()
    result = einsum("ij,j->i", delta, T1)
    time_e = perf_counter()
    print(f"{1_000 *(time_e - time_s):.2f}ms")
    print(np.allclose(result, T1))
